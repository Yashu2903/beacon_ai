from collections import Counter, defaultdict
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.step import Step
from app.models.source_evidence import SourceEvidence
from app.models.manual_page_structure import ManualPageStructure, ManualPageType
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

MIN_VISIBLE_STEP_COVERAGE_FOR_VISIBLE_CHECK = 0.45
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
        return True

    return region_type in TRUSTED_STEP_REGION_TYPES


def _get_manual_page_structures_by_page(
    db: Session,
    document_id: UUID,
) -> dict[int, ManualPageStructure]:
    rows = (
        db.query(ManualPageStructure)
        .filter(ManualPageStructure.document_id == document_id)
        .order_by(ManualPageStructure.page_number.asc())
        .all()
    )

    return {row.page_number: row for row in rows}


def _get_visible_step_numbers_from_manual_structure(
    structures_by_page: dict[int, ManualPageStructure],
) -> dict[int, list[int]]:
    visible_by_page: dict[int, list[int]] = {}

    for page_number, structure in structures_by_page.items():
        numbers: list[int] = []

        for value in structure.visible_step_numbers or []:
            number = _coerce_step_number(value)

            if number is not None:
                numbers.append(number)

        if numbers:
            visible_by_page[page_number] = sorted(set(numbers))

    return visible_by_page


def _get_visible_step_numbers_from_diagram_metadata(
    db: Session,
    document_id: UUID,
) -> dict[int, list[int]]:
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


def get_visible_step_numbers_by_page(
    db: Session,
    document_id: UUID,
) -> dict[int, list[int]]:
    """
    Primary source:
    - ManualPageStructure.visible_step_numbers

    Fallback:
    - diagram_region.metadata_json.step_number

    We intentionally do not scan all text spans because that created false
    positives from quantities, dates, page numbers, and part numbers.
    """
    structures_by_page = _get_manual_page_structures_by_page(
        db=db,
        document_id=document_id,
    )

    visible_from_structure = _get_visible_step_numbers_from_manual_structure(
        structures_by_page=structures_by_page,
    )

    if visible_from_structure:
        return visible_from_structure

    return _get_visible_step_numbers_from_diagram_metadata(
        db=db,
        document_id=document_id,
    )


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
    if not extracted_numbers:
        return 0.0

    if not visible_numbers:
        return 0.0

    matched = len(visible_numbers & extracted_numbers)

    return matched / len(extracted_numbers)


def _compute_sequence_gaps(numbers: list[int]) -> list[int]:
    if not numbers:
        return []

    unique_numbers = sorted(set(numbers))

    if len(unique_numbers) < 3:
        return []

    min_number = min(unique_numbers)
    max_number = max(unique_numbers)

    expected = set(range(min_number, max_number + 1))
    actual = set(unique_numbers)

    return sorted(expected - actual)


def _get_step_page_type(
    step: Step,
    structures_by_page: dict[int, ManualPageStructure],
) -> ManualPageType | None:
    if step.source_page_number is None:
        return None

    structure = structures_by_page.get(step.source_page_number)

    if structure is None:
        return None

    return structure.page_type


def _is_assembly_page_type(page_type: ManualPageType | None) -> bool:
    return page_type in {
        ManualPageType.assembly_step,
        ManualPageType.mixed_inventory_and_step,
    }


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
                message=f"Step number {number} appears multiple times in extracted steps.",
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


def _build_manual_structure_quality_issues(
    structures_by_page: dict[int, ManualPageStructure],
    visible_by_page: dict[int, list[int]],
) -> tuple[list[StepValidationIssue], bool]:
    """
    Validate the manual structure detector itself.

    Returns:
    - issues
    - structure_is_suspicious

    If structure is suspicious, we still use it for missing-visible checks,
    but suppress broad unexpected-step checks to avoid false positives.
    """
    issues: list[StepValidationIssue] = []
    structure_is_suspicious = False

    if not structures_by_page:
        issues.append(
            StepValidationIssue(
                issue_type="manual_structure_missing",
                severity="warning",
                page_number=None,
                step_number=None,
                message="No ManualPageStructure rows found for this document.",
                suggested_action=(
                    "Run POST /documents/{document_id}/structure/detect before validation."
                ),
                related_step_ids=[],
                metadata={},
            )
        )

        return issues, True

    visible_occurrences: dict[int, list[int]] = defaultdict(list)

    for page_number, numbers in visible_by_page.items():
        for number in numbers:
            visible_occurrences[number].append(page_number)

    duplicate_visible_numbers = sorted(
        number
        for number, pages in visible_occurrences.items()
        if len(pages) > 1
    )

    for number in duplicate_visible_numbers:
        structure_is_suspicious = True

        issues.append(
            StepValidationIssue(
                issue_type="duplicate_visible_step_number",
                severity="warning",
                page_number=visible_occurrences[number][0],
                step_number=number,
                message=f"Manual structure detected visible step {number} on multiple pages.",
                suggested_action=(
                    "Inspect the detected page structure. This may indicate OCR drift "
                    "or an incorrect repaired visible step."
                ),
                related_step_ids=[],
                metadata={
                    "candidate_pages": visible_occurrences[number],
                },
            )
        )

    visible_numbers = sorted(
        {
            number
            for numbers in visible_by_page.values()
            for number in numbers
        }
    )

    if visible_numbers:
        min_visible = min(visible_numbers)

        if min_visible != 1:
            structure_is_suspicious = True

            issues.append(
                StepValidationIssue(
                    issue_type="manual_structure_missing_start_step",
                    severity="warning",
                    page_number=None,
                    step_number=1,
                    message=f"Manual structure starts at visible step {min_visible}, not step 1.",
                    suggested_action=(
                        "Inspect early assembly pages. The detector may have missed "
                        "the first visible steps or misread a diagram artifact."
                    ),
                    related_step_ids=[],
                    metadata={
                        "min_visible_step": min_visible,
                        "visible_step_numbers": visible_numbers,
                    },
                )
            )

        manual_sequence_gaps = _compute_sequence_gaps(visible_numbers)

        for missing_number in manual_sequence_gaps:
            structure_is_suspicious = True

            candidate_pages = _get_pages_for_visible_number(
                visible_by_page=visible_by_page,
                step_number=missing_number - 1,
            ) + _get_pages_for_visible_number(
                visible_by_page=visible_by_page,
                step_number=missing_number + 1,
            )

            issues.append(
                StepValidationIssue(
                    issue_type="manual_structure_sequence_gap",
                    severity="warning",
                    page_number=candidate_pages[0] if candidate_pages else None,
                    step_number=missing_number,
                    message=(
                        f"Manual structure visible-step sequence appears to skip "
                        f"step {missing_number}."
                    ),
                    suggested_action=(
                        "Inspect nearby pages. This may be a real missing structure "
                        "detection or a page/step repair drift."
                    ),
                    related_step_ids=[],
                    metadata={
                        "candidate_pages": sorted(set(candidate_pages)),
                        "visible_step_numbers": visible_numbers,
                    },
                )
            )

    for structure in structures_by_page.values():
        metadata = structure.metadata_json or {}

        if metadata.get("repair_applied") is True:
            issues.append(
                StepValidationIssue(
                    issue_type="manual_structure_repaired_page",
                    severity="info",
                    page_number=structure.page_number,
                    step_number=None,
                    message=(
                        f"Manual structure for page {structure.page_number} includes "
                        "heuristically repaired step numbers."
                    ),
                    suggested_action=(
                        "Use this page as a helpful signal, but keep confidence lower "
                        "than directly OCR-detected step numbers."
                    ),
                    related_step_ids=[],
                    metadata={
                        "visible_step_numbers": structure.visible_step_numbers,
                        "confidence": structure.confidence,
                        "repair_types": metadata.get("repair_types", []),
                        "repair_notes": metadata.get("repair_notes", []),
                    },
                )
            )

        if float(structure.confidence or 0.0) < 0.65 and structure.visible_step_numbers:
            issues.append(
                StepValidationIssue(
                    issue_type="manual_structure_low_confidence_visible_step",
                    severity="warning",
                    page_number=structure.page_number,
                    step_number=None,
                    message=(
                        f"Manual structure detected visible steps on page "
                        f"{structure.page_number}, but confidence is low."
                    ),
                    suggested_action=(
                        "Review this page before using it as strong validation evidence."
                    ),
                    related_step_ids=[],
                    metadata={
                        "visible_step_numbers": structure.visible_step_numbers,
                        "confidence": structure.confidence,
                    },
                )
            )

    return issues, structure_is_suspicious


def _build_missing_visible_step_issues(
    visible_by_page: dict[int, list[int]],
    missing_visible_numbers: list[int],
    structures_by_page: dict[int, ManualPageStructure],
    structure_is_suspicious: bool,
) -> list[StepValidationIssue]:
    issues: list[StepValidationIssue] = []

    for number in missing_visible_numbers:
        candidate_pages = _get_pages_for_visible_number(
            visible_by_page=visible_by_page,
            step_number=number,
        )

        confidences = []
        page_types = []

        for page in candidate_pages:
            structure = structures_by_page.get(page)

            if structure is not None:
                confidences.append(float(structure.confidence or 0.0))
                page_types.append(str(structure.page_type.value))

        severity = "warning" if structure_is_suspicious else "error"

        # If the page has good confidence, still treat it as error even if
        # another part of the structure is suspicious.
        if confidences and max(confidences) >= 0.70:
            severity = "error"

        issues.append(
            StepValidationIssue(
                issue_type="missing_visible_step",
                severity=severity,
                page_number=candidate_pages[0] if candidate_pages else None,
                step_number=number,
                message=(
                    f"Manual structure detected visible step {number}, "
                    "but no extracted step has this step number."
                ),
                suggested_action=(
                    "Re-run extraction for the candidate page or manually create "
                    "the missing step if the visible step is real."
                ),
                related_step_ids=[],
                metadata={
                    "candidate_pages": candidate_pages,
                    "structure_confidences": confidences,
                    "page_types": page_types,
                    "structure_is_suspicious": structure_is_suspicious,
                },
            )
        )

    return issues


def _filter_unexpected_numbers(
    steps: list[Step],
    unexpected_numbers: list[int],
    structures_by_page: dict[int, ManualPageStructure],
    max_visible_number: int | None,
) -> list[int]:
    """
    Remove acceptable unexpected numbers before reporting.

    Example:
    Manual structure detected visible steps only up to 11, but extracted step 12
    comes from an assembly page. This may mean structure detection missed the
    final visible number on that same assembly page, so we should not report it
    as unexpected.

    But if step 14 or 15 comes from an informational page, keep it unexpected.
    """
    filtered: list[int] = []

    for number in unexpected_numbers:
        related_steps = [
            step
            for step in steps
            if step.step_number == number
        ]

        should_suppress = False

        if max_visible_number is not None and number == max_visible_number + 1:
            if related_steps and all(
                _is_assembly_page_type(
                    _get_step_page_type(step, structures_by_page)
                )
                for step in related_steps
            ):
                should_suppress = True

        if not should_suppress:
            filtered.append(number)

    return filtered


def _build_unexpected_step_issues(
    steps: list[Step],
    unexpected_numbers: list[int],
    structures_by_page: dict[int, ManualPageStructure],
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

        page_types = []

        for step in related_steps:
            page_type = _get_step_page_type(
                step=step,
                structures_by_page=structures_by_page,
            )

            if page_type is not None:
                page_types.append(str(page_type.value))

        issues.append(
            StepValidationIssue(
                issue_type="unexpected_step_number",
                severity="warning",
                page_number=related_pages[0] if related_pages else None,
                step_number=number,
                message=(
                    f"Extracted step number {number} was not found among "
                    "manual-structure visible step numbers."
                ),
                suggested_action=(
                    "Verify this is not an informational, inventory, or variant page "
                    "incorrectly converted into a numbered assembly step."
                ),
                related_step_ids=related_step_ids,
                metadata={
                    "candidate_pages": related_pages,
                    "page_types": page_types,
                },
            )
        )

    return issues


def _build_sequence_gap_issues(
    steps: list[Step],
    sequence_gaps: list[int],
) -> list[StepValidationIssue]:
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
                    "with this number, re-run extraction or create it manually."
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
    used_manual_structure: bool,
) -> list[StepValidationIssue]:
    if not extracted_numbers:
        return []

    if coverage >= MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK:
        return []

    source = "ManualPageStructure" if used_manual_structure else "diagram metadata"

    return [
        StepValidationIssue(
            issue_type="visible_step_detection_low_coverage",
            severity="warning",
            page_number=None,
            step_number=None,
            message=(
                f"Trusted visible-step detection coverage is low using {source}. "
                "Skipping broad unexpected-step validation to avoid false positives."
            ),
            suggested_action=(
                "Improve manual structure detection or review sequence-gap and "
                "duplicate-step warnings manually."
            ),
            related_step_ids=[],
            metadata={
                "visible_step_numbers_detected": sorted(visible_numbers),
                "extracted_step_numbers": sorted(extracted_numbers),
                "coverage": coverage,
                "threshold": MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK,
                "source": source,
            },
        )
    ]


def _build_step_on_non_assembly_page_issues(
    steps: list[Step],
    structures_by_page: dict[int, ManualPageStructure],
) -> list[StepValidationIssue]:
    issues: list[StepValidationIssue] = []

    non_assembly_types = {
        ManualPageType.cover,
        ManualPageType.parts_inventory,
        ManualPageType.informational,
        ManualPageType.back_matter,
    }

    for step in steps:
        if step.source_page_number is None:
            continue

        structure = structures_by_page.get(step.source_page_number)

        if structure is None:
            continue

        if structure.page_type not in non_assembly_types:
            continue

        severity = "warning"

        if structure.page_type in {
            ManualPageType.cover,
            ManualPageType.back_matter,
            ManualPageType.parts_inventory,
        }:
            severity = "error"

        issues.append(
            StepValidationIssue(
                issue_type="step_on_non_assembly_page",
                severity=severity,
                page_number=step.source_page_number,
                step_number=step.step_number,
                message=(
                    f"Extracted step {step.step_number} comes from page "
                    f"{step.source_page_number}, which manual structure classifies "
                    f"as {structure.page_type.value}."
                ),
                suggested_action=(
                    "Verify this step. It may be an informational/variant page "
                    "incorrectly converted into a numbered assembly step."
                ),
                related_step_ids=[str(step.id)],
                metadata={
                    "page_type": structure.page_type.value,
                    "visible_step_numbers_on_page": structure.visible_step_numbers,
                    "structure_confidence": structure.confidence,
                },
            )
        )

    return issues


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
                    message=f"Step {step.step_number} has low confidence {confidence:.2f}.",
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
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise ValueError(f"Job not found: {job_id}")

    steps = (
        db.query(Step)
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .all()
    )

    structures_by_page = _get_manual_page_structures_by_page(
        db=db,
        document_id=job.document_id,
    )

    used_manual_structure = bool(structures_by_page)

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

    manual_structure_issues, structure_is_suspicious = (
        _build_manual_structure_quality_issues(
            structures_by_page=structures_by_page,
            visible_by_page=visible_by_page,
        )
    )

    missing_visible_numbers: list[int] = []
    unexpected_numbers: list[int] = []

    if coverage >= MIN_VISIBLE_STEP_COVERAGE_FOR_VISIBLE_CHECK:
        missing_visible_numbers = sorted(
            visible_number_set - extracted_number_set
        )

    max_visible_number = max(visible_number_set) if visible_number_set else None

    # Only do broad unexpected-step validation when structure coverage is good
    # and the structure itself does not look drifted.
    if (
        coverage >= MIN_VISIBLE_STEP_COVERAGE_FOR_UNEXPECTED_CHECK
        and not structure_is_suspicious
    ):
        raw_unexpected_numbers = sorted(
            extracted_number_set - visible_number_set
        )

        unexpected_numbers = _filter_unexpected_numbers(
            steps=steps,
            unexpected_numbers=raw_unexpected_numbers,
            structures_by_page=structures_by_page,
            max_visible_number=max_visible_number,
        )

    sequence_gaps = _compute_sequence_gaps(extracted_numbers)

    issues: list[StepValidationIssue] = []

    issues.extend(
        _build_duplicate_step_issues(
            steps=steps,
            duplicate_numbers=duplicate_numbers,
        )
    )

    issues.extend(manual_structure_issues)

    issues.extend(
        _build_visible_detection_quality_issue(
            visible_numbers=visible_number_set,
            extracted_numbers=extracted_number_set,
            coverage=coverage,
            used_manual_structure=used_manual_structure,
        )
    )

    issues.extend(
        _build_missing_visible_step_issues(
            visible_by_page=visible_by_page,
            missing_visible_numbers=missing_visible_numbers,
            structures_by_page=structures_by_page,
            structure_is_suspicious=structure_is_suspicious,
        )
    )

    issues.extend(
        _build_unexpected_step_issues(
            steps=steps,
            unexpected_numbers=unexpected_numbers,
            structures_by_page=structures_by_page,
        )
    )

    issues.extend(
        _build_sequence_gap_issues(
            steps=steps,
            sequence_gaps=sequence_gaps,
        )
    )

    issues.extend(
        _build_step_on_non_assembly_page_issues(
            steps=steps,
            structures_by_page=structures_by_page,
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