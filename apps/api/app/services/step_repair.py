from uuid import UUID

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.step import Step
from app.schemas.extraction_validation import ExtractionValidationResult
from app.services.extraction_validation import validate_extracted_steps_for_job


AUTO_REVIEWABLE_ISSUE_TYPES = {
    "duplicate_step_number",
    "unexpected_step_number",
    "step_on_non_assembly_page",
    "missing_visible_step",
    "sequence_gap",
}


def _issue_to_dict(issue) -> dict:
    return {
        "issue_type": issue.issue_type,
        "severity": issue.severity,
        "page_number": issue.page_number,
        "step_number": issue.step_number,
        "message": issue.message,
        "suggested_action": issue.suggested_action,
        "related_step_ids": issue.related_step_ids or [],
        "metadata": issue.metadata or {},
    }


def _get_steps_by_id(
    db: Session,
    step_ids: list[str],
) -> dict[str, Step]:
    if not step_ids:
        return {}

    rows = (
        db.query(Step)
        .filter(Step.id.in_([UUID(step_id) for step_id in step_ids]))
        .all()
    )

    return {
        str(step.id): step
        for step in rows
    }


def _get_steps_for_job(
    db: Session,
    job_id: UUID,
) -> list[Step]:
    return (
        db.query(Step)
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .all()
    )


def _build_duplicate_step_repair(issue) -> dict:
    return {
        "repair_type": "merge_or_reject_duplicate_step",
        "priority": "high",
        "step_number": issue.step_number,
        "page_number": issue.page_number,
        "related_step_ids": issue.related_step_ids or [],
        "source_issue_types": [issue.issue_type],
        "instruction": (
            f"Step {issue.step_number} appears multiple times. "
            "Compare the duplicate entries. If they describe the same visible manual step, "
            "merge them into one step. If one is incorrect, reject or remove the incorrect duplicate."
        ),
        "requires_llm": True,
        "safe_auto_action": "mark_related_steps_needs_attention",
    }


def _build_missing_visible_step_repair(issue) -> dict:
    candidate_pages = (issue.metadata or {}).get("candidate_pages", [])

    return {
        "repair_type": "create_missing_visible_step",
        "priority": "high",
        "step_number": issue.step_number,
        "page_number": issue.page_number,
        "candidate_pages": candidate_pages,
        "related_step_ids": issue.related_step_ids or [],
        "source_issue_types": [issue.issue_type],
        "instruction": (
            f"Manual structure detected visible Step {issue.step_number}, "
            "but extraction did not create that step. Re-inspect the candidate page images "
            "and create the missing step only if the step number is visibly present."
        ),
        "requires_llm": True,
        "safe_auto_action": "none",
    }


def _build_unexpected_step_repair(issue) -> dict:
    page_types = (issue.metadata or {}).get("page_types", [])
    candidate_pages = (issue.metadata or {}).get("candidate_pages", [])

    return {
        "repair_type": "remove_or_reclassify_unexpected_step",
        "priority": "medium",
        "step_number": issue.step_number,
        "page_number": issue.page_number,
        "candidate_pages": candidate_pages,
        "page_types": page_types,
        "related_step_ids": issue.related_step_ids or [],
        "source_issue_types": [issue.issue_type],
        "instruction": (
            f"Extracted Step {issue.step_number} was not found in the visible manual-step structure. "
            "Verify whether this is a real assembly step. If it comes from an informational, "
            "variant, safety, or back-matter page, do not keep it as a numbered assembly step."
        ),
        "requires_llm": True,
        "safe_auto_action": "mark_related_steps_needs_attention",
    }


def _build_non_assembly_page_repair(issue) -> dict:
    metadata = issue.metadata or {}
    page_type = metadata.get("page_type")

    return {
        "repair_type": "review_step_from_non_assembly_page",
        "priority": "medium",
        "step_number": issue.step_number,
        "page_number": issue.page_number,
        "page_type": page_type,
        "related_step_ids": issue.related_step_ids or [],
        "source_issue_types": [issue.issue_type],
        "instruction": (
            f"Extracted Step {issue.step_number} comes from page {issue.page_number}, "
            f"which is classified as {page_type}. Verify whether this should remain an "
            "assembly step. If the page is informational, variant-only, cover, inventory, "
            "or back matter, remove it from the assembly sequence."
        ),
        "requires_llm": True,
        "safe_auto_action": "mark_related_steps_needs_attention",
    }


def _build_sequence_gap_repair(issue) -> dict:
    candidate_pages = (issue.metadata or {}).get("candidate_pages", [])

    return {
        "repair_type": "review_sequence_gap",
        "priority": "low",
        "step_number": issue.step_number,
        "page_number": issue.page_number,
        "candidate_pages": candidate_pages,
        "related_step_ids": issue.related_step_ids or [],
        "source_issue_types": [issue.issue_type],
        "instruction": (
            f"The extracted sequence appears to skip Step {issue.step_number}. "
            "Check nearby pages and existing extracted steps. Do not invent this step unless "
            "the manual visibly contains it."
        ),
        "requires_llm": False,
        "safe_auto_action": "none",
    }


def _build_generic_repair(issue) -> dict:
    return {
        "repair_type": "manual_review",
        "priority": "low",
        "step_number": issue.step_number,
        "page_number": issue.page_number,
        "related_step_ids": issue.related_step_ids or [],
        "source_issue_types": [issue.issue_type],
        "instruction": issue.suggested_action or issue.message,
        "requires_llm": False,
        "safe_auto_action": "none",
    }


def _build_repair_item(issue) -> dict:
    if issue.issue_type == "duplicate_step_number":
        return _build_duplicate_step_repair(issue)

    if issue.issue_type == "missing_visible_step":
        return _build_missing_visible_step_repair(issue)

    if issue.issue_type == "unexpected_step_number":
        return _build_unexpected_step_repair(issue)

    if issue.issue_type == "step_on_non_assembly_page":
        return _build_non_assembly_page_repair(issue)

    if issue.issue_type == "sequence_gap":
        return _build_sequence_gap_repair(issue)

    return _build_generic_repair(issue)


def _merge_unique_list_values(*values: list) -> list:
    merged = []

    for value_list in values:
        for value in value_list or []:
            if value not in merged:
                merged.append(value)

    return merged


def _priority_rank(priority: str) -> int:
    priority_order = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }

    return priority_order.get(priority, 99)


def _higher_priority(priority_a: str, priority_b: str) -> str:
    if _priority_rank(priority_a) <= _priority_rank(priority_b):
        return priority_a

    return priority_b


def _can_merge_unexpected_and_non_assembly(
    first: dict,
    second: dict,
) -> bool:
    same_step = first.get("step_number") == second.get("step_number")
    same_page = first.get("page_number") == second.get("page_number")

    if not same_step or not same_page:
        return False

    repair_types = {
        first.get("repair_type"),
        second.get("repair_type"),
    }

    return repair_types == {
        "remove_or_reclassify_unexpected_step",
        "review_step_from_non_assembly_page",
    }


def _merge_unexpected_and_non_assembly(
    first: dict,
    second: dict,
) -> dict:
    step_number = first.get("step_number")
    page_number = first.get("page_number")

    page_type = first.get("page_type") or second.get("page_type")

    page_types = _merge_unique_list_values(
        first.get("page_types", []),
        second.get("page_types", []),
        [page_type] if page_type else [],
    )

    candidate_pages = _merge_unique_list_values(
        first.get("candidate_pages", []),
        second.get("candidate_pages", []),
        [page_number] if page_number else [],
    )

    related_step_ids = _merge_unique_list_values(
        first.get("related_step_ids", []),
        second.get("related_step_ids", []),
    )

    source_issue_types = _merge_unique_list_values(
        first.get("source_issue_types", []),
        second.get("source_issue_types", []),
    )

    return {
        "repair_type": "remove_step_from_non_assembly_page",
        "priority": _higher_priority(
            first.get("priority", "low"),
            second.get("priority", "low"),
        ),
        "step_number": step_number,
        "page_number": page_number,
        "candidate_pages": candidate_pages,
        "page_types": page_types,
        "related_step_ids": related_step_ids,
        "source_issue_types": source_issue_types,
        "instruction": (
            f"Extracted Step {step_number} comes from page {page_number}, which is classified "
            f"as {page_type or 'non-assembly'}, and it was not found in the visible manual-step "
            "structure. Remove it from the numbered assembly sequence unless visible manual "
            "evidence clearly proves it is a real assembly step."
        ),
        "requires_llm": True,
        "safe_auto_action": "mark_related_steps_needs_attention",
    }


def _deduplicate_repair_plan(
    repair_items: list[dict],
) -> list[dict]:
    """
    Merge repair items that describe the same underlying problem.

    Example:
    - unexpected_step_number for Step 14
    - step_on_non_assembly_page for Step 14

    These are really one issue:
    - remove_step_from_non_assembly_page for Step 14
    """
    remaining = repair_items[:]
    merged_items: list[dict] = []

    while remaining:
        current = remaining.pop(0)
        merged = False

        for index, other in enumerate(remaining):
            if _can_merge_unexpected_and_non_assembly(current, other):
                merged_items.append(
                    _merge_unexpected_and_non_assembly(
                        current,
                        other,
                    )
                )

                remaining.pop(index)
                merged = True
                break

        if not merged:
            merged_items.append(current)

    # Also remove exact duplicates if the same item was produced twice.
    seen_keys = set()
    deduped: list[dict] = []

    for item in merged_items:
        key = (
            item.get("repair_type"),
            item.get("step_number"),
            item.get("page_number"),
            tuple(item.get("related_step_ids", [])),
        )

        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped.append(item)

    return deduped


def build_step_repair_plan(
    validation: ExtractionValidationResult,
) -> list[dict]:
    repair_items: list[dict] = []

    for issue in validation.issues:
        if issue.issue_type not in AUTO_REVIEWABLE_ISSUE_TYPES:
            continue

        repair_items.append(_build_repair_item(issue))

    repair_items = _deduplicate_repair_plan(repair_items)

    repair_items.sort(
        key=lambda item: (
            _priority_rank(item.get("priority", "low")),
            item.get("page_number") or 9999,
            item.get("step_number") or 9999,
        )
    )

    return repair_items


def _build_llm_repair_packet(
    job: Job,
    validation: ExtractionValidationResult,
    repair_plan: list[dict],
    steps: list[Step],
) -> dict:
    extracted_steps = []

    for step in steps:
        extracted_steps.append(
            {
                "step_id": str(step.id),
                "step_number": step.step_number,
                "order_index": step.order_index,
                "source_page_number": step.source_page_number,
                "action": step.action,
                "parts": step.parts or [],
                "tools": step.tools or [],
                "confidence": step.confidence,
                "review_status": step.review_status,
            }
        )

    return {
        "job_id": str(job.id),
        "document_id": str(job.document_id),
        "goal": (
            "Repair extracted furniture assembly steps using validation issues. "
            "Do not invent steps. Only create or modify steps when supported by visible manual evidence."
        ),
        "validation_summary": {
            "issue_count": validation.issue_count,
            "error_count": validation.error_count,
            "warning_count": validation.warning_count,
            "visible_step_numbers_by_page": validation.structure.visible_step_numbers_by_page,
            "extracted_step_numbers": validation.structure.extracted_step_numbers,
            "duplicate_step_numbers": validation.structure.duplicate_step_numbers,
            "missing_visible_step_numbers": validation.structure.missing_visible_step_numbers,
            "unexpected_step_numbers": validation.structure.unexpected_step_numbers,
        },
        "repair_plan": repair_plan,
        "current_extracted_steps": extracted_steps,
        "rules": [
            "Do not create a numbered assembly step from cover pages.",
            "Do not create a numbered assembly step from parts inventory pages.",
            "Do not create a numbered assembly step from informational or variant-only pages.",
            "If one visible manual step was split into two extracted steps, merge them.",
            "If a visible manual step is missing, create it only when the step number is visibly present.",
            "Do not invent sequence numbers just to fill gaps.",
            "Preserve step IDs when editing existing steps.",
            "Return proposed changes as structured JSON.",
        ],
    }


def _apply_safe_review_marks(
    db: Session,
    repair_plan: list[dict],
) -> list[str]:
    """
    Safe v1 behavior:
    - Do not delete steps.
    - Do not create steps.
    - Do not reorder steps.
    - Only mark related bad steps as needs_attention.

    This gives reviewers a clean queue without risking destructive changes.
    """
    updated_step_ids: list[str] = []
    related_step_ids: set[str] = set()

    for item in repair_plan:
        if item.get("safe_auto_action") != "mark_related_steps_needs_attention":
            continue

        for step_id in item.get("related_step_ids", []):
            related_step_ids.add(step_id)

    steps_by_id = _get_steps_by_id(
        db=db,
        step_ids=list(related_step_ids),
    )

    for step_id, step in steps_by_id.items():
        step.review_status = "needs_attention"

        note = (
            "Marked needs_attention by validation-driven repair planning. "
            "Review this step before Gate 1 approval."
        )

        if step.reviewer_notes:
            if note not in step.reviewer_notes:
                step.reviewer_notes = f"{step.reviewer_notes}\n{note}"
        else:
            step.reviewer_notes = note

        updated_step_ids.append(step_id)

    if updated_step_ids:
        db.commit()

    return updated_step_ids


def build_or_apply_step_repair(
    db: Session,
    job_id: UUID,
    apply_safe_fixes: bool = False,
) -> dict:
    job = db.get(Job, job_id)

    if job is None:
        raise ValueError("Job not found")

    validation = validate_extracted_steps_for_job(
        db=db,
        job_id=job_id,
    )

    steps = _get_steps_for_job(
        db=db,
        job_id=job_id,
    )

    repair_plan = build_step_repair_plan(validation)

    updated_step_ids: list[str] = []

    if apply_safe_fixes:
        updated_step_ids = _apply_safe_review_marks(
            db=db,
            repair_plan=repair_plan,
        )

    llm_repair_packet = _build_llm_repair_packet(
        job=job,
        validation=validation,
        repair_plan=repair_plan,
        steps=steps,
    )

    requires_llm_repair = any(
        item.get("requires_llm") is True
        for item in repair_plan
    )

    return {
        "job_id": str(job.id),
        "document_id": str(job.document_id),
        "apply_safe_fixes": apply_safe_fixes,
        "updated_step_ids": updated_step_ids,
        "requires_llm_repair": requires_llm_repair,
        "repair_item_count": len(repair_plan),
        "repair_plan": repair_plan,
        "llm_repair_packet": llm_repair_packet,
        "validation": validation,
    }