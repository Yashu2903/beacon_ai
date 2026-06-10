from collections import Counter, defaultdict
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.step import Step
from app.models.source_evidence import SourceEvidence
from app.schemas.extraction_validation import (
    DocumentStepStructure,
    ExtractionValidationResult,
    StepValidationIssue,
)


TRUSTED_STEP_REGION_TYPES = {
    "step",
    "assembly_step",
    "primary_step",
    "step_region",
}

IGNORED_REGION_TYPES = {
    "parts_inventory",
    "tool_inventory",
    "cover",
    "back_matter",
    "informational",
    "warning",
    "safety",
}


MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK = 0.60


def _coerce_step_number(value) -> int | None:
    if value is None:
        return None

    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        return None

    if 1 <= number <= 50:
        return number

    return None


def _is_trusted_step_region(metadata: dict) -> bool:
    region_type = metadata.get("region_type")

    if region_type in IGNORED_REGION_TYPES:
        return False

    if region_type is None:
        # Older extractor versions may not populate region_type.
        return True

    return region_type in TRUSTED_STEP_REGION_TYPES


def get_visible_step_numbers_by_page(
    db: Session,
    document_id: UUID,
) -> dict[int, list[int]]:
    """
    Return trusted visible manual step numbers by page.

    This intentionally does NOT scan every text span for numbers because that
    creates false positives from part numbers, quantities, page numbers, dates,
    and document IDs.

    It only trusts diagram_region metadata step_number values.
    """
    evidence_rows = (
        db.query(SourceEvidence)
        .filter(SourceEvidence.document_id == document_id)
        .filter(SourceEvidence.evidence_type == "diagram_region")
        .all()
    )

    numbers_by_page: dict[int, set[int]] = defaultdict(set)

    for evidence in evidence_rows:
        metadata = evidence.metadata_json or {}

        if not _is_trusted_step_region(metadata):
            continue

        step_number = _coerce_step_number(metadata.get("step_number"))

        if step_number is not None:
            numbers_by_page[evidence.page_number].add(step_number)

    return {
        page: sorted(numbers)
        for page, numbers in numbers_by_page.items()
        if numbers
    }


def _get_step_ids_by_number(steps: list[Step]) -> dict[int, list[str]]:
    step_ids_by_number: dict[int, list[str]] = defaultdict(list)

    for step in steps:
        if step.step_number is not None:
            step_ids_by_number[step.step_number].append(str(step.id))

    return dict(step_ids_by_number)


def _get_pages_for_visible_number(
    visible_by_page: dict[int, list[int]],
    step_number: int,
) -> list[int]:
    return [
        page
        for page, numbers in visible_by_page.items()
        if step_number in numbers
    ]


def _visible_step_coverage(
    visible_numbers: set[int],
    extracted_numbers: set[int],
) -> float:
    """
    Estimate whether visible-step metadata is complete enough to compare against.

    If visible_numbers is empty or covers only a small fraction of extracted steps,
    we should not generate unexpected_step_number warnings for every extracted step.
    """
    if not extracted_numbers:
        return 0.0

    if not visible_numbers:
        return 0.0

    matched = len(visible_numbers & extracted_numbers)

    return matched / len(extracted_numbers)


def _compute_sequence_gaps(extracted_numbers: list[int]) -> list[int]:
    """
    Detect missing numbers inside the extracted step sequence.

    Example:
    [1,2,3,4,5,6,7,9,10] -> missing [8]

    This does not rely on OCR or diagram metadata.
    It is useful when visible step detection is incomplete.
    """
    if not extracted_numbers:
        return []

    unique_numbers = sorted(set(extracted_numbers))

    if len(unique_numbers) < 3:
        return []

    min_number = min(unique_numbers)
    max_number = max(unique_numbers)

    expected = set(range(min_number, max_number + 1))
    actual = set(unique_numbers)

    return sorted(expected - actual)


def _build_duplicate_step_issues(
    steps: list[Step],
    duplicate_numbers: list[int],
) -> list[StepValidationIssue]:
    issues: list[StepValidationIssue] = []

    step_ids_by_number = _get_step_ids_by_number(steps)

    for number in duplicate_numbers:
        related_step_ids = step_ids_by_number.get(number, [])

        related_pages = sorted(
            {
                step.source_page_number
                for step in steps
                if step.step_number == number and step.source_page_number is not None
            }
        )

        issues.append(
            StepValidationIssue(
                issue_type="duplicate_step_number",
                severity="error",
                page_number=related_pages[0] if related_pages else None,
                step_number=number,
                message=(
                    f"Step number {number} appears multiple times in extracted steps."
                ),
                suggested_action=(
                    "Merge duplicate entries if they describe one visible manual step, "
                    "or mark the incorrect duplicate as rejected/needs_attention."
                ),
                related_step_ids=related_step_ids,
                metadata={
                    "candidate_pages": related_pages,
                    "duplicate_count": len(related_step_ids),
                },
            )
        )

    return issues


def _build_missing_visible_step_issues(
    visible_by_page: dict[int, list[int]],
    missing_visible_numbers: list[int],
) -> list[StepValidationIssue]:
    issues: list[StepValidationIssue] = []

    for number in missing_visible_numbers:
        candidate_pages = _get_pages_for_visible_number(
            visible_by_page=visible_by_page,
            step_number=number,
        )

        issues.append(
            StepValidationIssue(
                issue_type="missing_visible_step",
                severity="error",
                page_number=candidate_pages[0] if candidate_pages else None,
                step_number=number,
                message=(
                    f"Trusted visible manual step {number} appears in diagram evidence, "
                    "but no extracted step has this step number."
                ),
                suggested_action=(
                    "Re-run extraction for the candidate page or manually create the missing step."
                ),
                related_step_ids=[],
                metadata={
                    "candidate_pages": candidate_pages,
                },
            )
        )

    return issues


def _build_unexpected_step_issues(
    steps: list[Step],
    unexpected_numbers: list[int],
) -> list[StepValidationIssue]:
    issues: list[StepValidationIssue] = []

    step_ids_by_number = _get_step_ids_by_number(steps)

    for number in unexpected_numbers:
        related_steps = [
            step
            for step in steps
            if step.step_number == number
        ]

        related_step_ids = step_ids_by_number.get(number, [])

        related_pages = sorted(
            {
                step.source_page_number
                for step in related_steps
                if step.source_page_number is not None
            }
        )

        issues.append(
            StepValidationIssue(
                issue_type="unexpected_step_number",
                severity="warning",
                page_number=related_pages[0] if related_pages else None,
                step_number=number,
                message=(
                    f"Extracted step number {number} was not found among trusted "
                    "visible step numbers from diagram evidence."
                ),
                suggested_action=(
                    "Verify this is not an informational, inventory, or variant page "
                    "incorrectly converted into a numbered assembly step."
                ),
                related_step_ids=related_step_ids,
                metadata={
                    "candidate_pages": related_pages,
                },
            )
        )

    return issues


def _build_sequence_gap_issues(
    steps: list[Step],
    sequence_gaps: list[int],
) -> list[StepValidationIssue]:
    """
    Flag missing numbers inside the extracted sequence.

    This is a warning, not an error, because some manuals contain informational
    or variant pages that may make the visible sequence non-obvious.
    """
    issues: list[StepValidationIssue] = []

    if not sequence_gaps:
        return issues

    extracted_by_page = {
        step.step_number: step.source_page_number
        for step in steps
        if step.step_number is not None
    }

    for missing_number in sequence_gaps:
        previous_page = extracted_by_page.get(missing_number - 1)
        next_page = extracted_by_page.get(missing_number + 1)

        candidate_pages = sorted(
            {
                page
                for page in [previous_page, next_page]
                if page is not None
            }
        )

        issues.append(
            StepValidationIssue(
                issue_type="sequence_gap",
                severity="warning",
                page_number=candidate_pages[0] if candidate_pages else None,
                step_number=missing_number,
                message=(
                    f"Extracted step sequence appears to skip step {missing_number}."
                ),
                suggested_action=(
                    "Check nearby source pages. If the manual has a real visible step "
                    "with this number, re-run extraction or create it manually. If this "
                    "is an informational/variant page, mark the warning as reviewed."
                ),
                related_step_ids=[],
                metadata={
                    "candidate_pages": candidate_pages,
                    "reason": "missing number inside extracted sequence",
                },
            )
        )

    return issues


def _build_visible_detection_quality_issue(
    visible_numbers: set[int],
    extracted_numbers: set[int],
    coverage: float,
) -> list[StepValidationIssue]:
    """
    Add one diagnostic warning when visible step detection coverage is poor.

    This prevents the API from spamming every extracted step as unexpected just
    because the OCR/diagram metadata did not capture step numbers reliably.
    """
    if not extracted_numbers:
        return []

    if coverage >= MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK:
        return []

    return [
        StepValidationIssue(
            issue_type="visible_step_detection_low_coverage",
            severity="warning",
            page_number=None,
            step_number=None,
            message=(
                "Trusted visible-step detection coverage is low. "
                "Skipping broad unexpected-step validation to avoid false positives."
            ),
            suggested_action=(
                "Improve diagram/OCR step-number metadata extraction, or review "
                "sequence-gap and duplicate-step warnings manually."
            ),
            related_step_ids=[],
            metadata={
                "visible_step_numbers_detected": sorted(visible_numbers),
                "extracted_step_numbers": sorted(extracted_numbers),
                "coverage": coverage,
                "threshold": MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK,
            },
        )
    ]


def _build_quality_issues(steps: list[Step]) -> list[StepValidationIssue]:
    issues: list[StepValidationIssue] = []

    for step in steps:
        confidence = float(step.confidence or 0)

        if confidence < 0.70:
            issues.append(
                StepValidationIssue(
                    issue_type="low_confidence_step",
                    severity="warning",
                    page_number=step.source_page_number,
                    step_number=step.step_number,
                    message=(
                        f"Step {step.step_number} has low confidence {confidence:.2f}."
                    ),
                    suggested_action="Review this step manually before storyboarding.",
                    related_step_ids=[str(step.id)],
                    metadata={
                        "confidence": confidence,
                    },
                )
            )

        if step.review_status == "needs_attention":
            issues.append(
                StepValidationIssue(
                    issue_type="step_needs_attention",
                    severity="warning",
                    page_number=step.source_page_number,
                    step_number=step.step_number,
                    message=step.reviewer_notes or "Step is marked needs_attention.",
                    suggested_action="Resolve reviewer notes before Gate 1 approval.",
                    related_step_ids=[str(step.id)],
                    metadata={},
                )
            )

        if not step.action or not step.action.strip():
            issues.append(
                StepValidationIssue(
                    issue_type="empty_step_action",
                    severity="error",
                    page_number=step.source_page_number,
                    step_number=step.step_number,
                    message="Extracted step has an empty action.",
                    suggested_action="Edit or reject this step before Gate 1 approval.",
                    related_step_ids=[str(step.id)],
                    metadata={},
                )
            )

    return issues


def validate_extracted_steps_for_job(
    db: Session,
    job_id: UUID,
) -> ExtractionValidationResult:
    """
    Validate extracted LLM/heuristic steps against manual structure signals.

    This does not call an LLM.
    This does not mutate the database.
    This only returns validation issues for review/debugging.

    Important:
    If trusted visible-step metadata has poor coverage, broad unexpected-step
    warnings are suppressed to avoid noisy false positives.
    """
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise ValueError(f"Job not found: {job_id}")

    steps = (
        db.query(Step)
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .all()
    )

    visible_by_page = get_visible_step_numbers_by_page(
        db=db,
        document_id=job.document_id,
    )

    extracted_numbers = [
        step.step_number
        for step in steps
        if step.step_number is not None
    ]

    extracted_counter = Counter(extracted_numbers)

    duplicate_numbers = sorted(
        number
        for number, count in extracted_counter.items()
        if count > 1
    )

    visible_numbers = sorted(
        {
            number
            for numbers in visible_by_page.values()
            for number in numbers
        }
    )

    extracted_number_set = set(extracted_numbers)
    visible_number_set = set(visible_numbers)

    coverage = _visible_step_coverage(
        visible_numbers=visible_number_set,
        extracted_numbers=extracted_number_set,
    )

    missing_visible_numbers: list[int] = []
    unexpected_numbers: list[int] = []

    if coverage >= MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK:
        missing_visible_numbers = sorted(
            visible_number_set - extracted_number_set
        )

        unexpected_numbers = sorted(
            extracted_number_set - visible_number_set
        )

    sequence_gaps = _compute_sequence_gaps(extracted_numbers)

    issues: list[StepValidationIssue] = []

    issues.extend(
        _build_duplicate_step_issues(
            steps=steps,
            duplicate_numbers=duplicate_numbers,
        )
    )

    issues.extend(
        _build_visible_detection_quality_issue(
            visible_numbers=visible_number_set,
            extracted_numbers=extracted_number_set,
            coverage=coverage,
        )
    )

    issues.extend(
        _build_missing_visible_step_issues(
            visible_by_page=visible_by_page,
            missing_visible_numbers=missing_visible_numbers,
        )
    )

    issues.extend(
        _build_unexpected_step_issues(
            steps=steps,
            unexpected_numbers=unexpected_numbers,
        )
    )

    issues.extend(
        _build_sequence_gap_issues(
            steps=steps,
            sequence_gaps=sequence_gaps,
        )
    )

    issues.extend(
        _build_quality_issues(steps=steps)
    )

    structure = DocumentStepStructure(
        visible_step_numbers_by_page=visible_by_page,
        extracted_step_numbers=extracted_numbers,
        duplicate_step_numbers=duplicate_numbers,
        missing_visible_step_numbers=missing_visible_numbers,
        unexpected_step_numbers=unexpected_numbers,
    )

    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")

    return ExtractionValidationResult(
        job_id=str(job.id),
        document_id=str(job.document_id),
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        structure=structure,
        issues=issues,
    )