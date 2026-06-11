from uuid import UUID

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.step import Step
from app.schemas.step_repair import StepRepairProposal, StepRepairProposedChange
from app.services.extraction_validation import validate_extracted_steps_for_job


UNSAFE_APPLY_PHRASES = {
    "likely",
    "probably",
    "may involve",
    "might involve",
    "appears to involve",
    "requires human review",
    "human review required",
    "please inspect",
    "manual inspection",
    "cannot be confirmed",
    "could not be confirmed",
    "exact action not available",
    "exact instruction text not available",
    "requires manual inspection",
    "needs review",
    "needs manual review",
}


class StepRepairApplyError(Exception):
    def __init__(
        self,
        message: str,
        applied_changes: list[dict] | None = None,
        rejected_changes: list[dict] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.applied_changes = applied_changes or []
        self.rejected_changes = rejected_changes or []


def _contains_unsafe_language(value: str | None) -> bool:
    if not value:
        return False

    lowered = value.lower()

    return any(phrase in lowered for phrase in UNSAFE_APPLY_PHRASES)


def _normalize_uuid(value, field_name: str) -> UUID:
    try:
        return UUID(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Invalid {field_name}: {value}") from exc


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


def _get_step_or_none(
    db: Session,
    step_id: str | None,
) -> Step | None:
    if not step_id:
        return None

    try:
        parsed_id = UUID(step_id)
    except ValueError:
        return None

    return db.get(Step, parsed_id)


def _get_existing_task_id_for_job(
    db: Session,
    job_id: UUID,
):
    first_step = (
        db.query(Step)
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .first()
    )

    if first_step is None:
        return None

    return getattr(first_step, "task_id", None)


def _change_to_dict(change: StepRepairProposedChange) -> dict:
    return change.model_dump(mode="json")


def _reject_change(
    change: StepRepairProposedChange,
    reason: str,
) -> dict:
    return {
        "operation": change.operation,
        "step_id": change.step_id,
        "step_ids": change.step_ids,
        "step_number": change.step_number,
        "source_page_number": change.source_page_number,
        "order_index": change.order_index,
        "rejected": True,
        "reason": reason,
        "change": _change_to_dict(change),
    }


def _accept_change(
    change: StepRepairProposedChange,
    message: str,
    affected_step_ids: list[str],
) -> dict:
    return {
        "operation": change.operation,
        "step_id": change.step_id,
        "step_ids": change.step_ids,
        "step_number": change.step_number,
        "source_page_number": change.source_page_number,
        "order_index": change.order_index,
        "applied": True,
        "message": message,
        "affected_step_ids": affected_step_ids,
        "change": _change_to_dict(change),
    }


def _validate_create_step_change(
    change: StepRepairProposedChange,
) -> list[str]:
    errors: list[str] = []

    if change.operation != "create_step":
        return errors

    if change.step_number is None:
        errors.append("create_step is missing step_number")

    if change.source_page_number is None:
        errors.append("create_step is missing source_page_number")

    if change.order_index is None:
        errors.append("create_step is missing order_index")

    if not change.action or not change.action.strip():
        errors.append("create_step is missing action")

    if change.confidence is None:
        errors.append("create_step is missing confidence")
    elif change.confidence < 0.70:
        errors.append(f"create_step confidence below 0.70: {change.confidence}")

    if _contains_unsafe_language(change.action):
        errors.append("create_step action contains unsafe/uncertain language")

    if _contains_unsafe_language(change.reason):
        errors.append("create_step reason contains unsafe/uncertain language")

    if _contains_unsafe_language(change.evidence_notes):
        errors.append("create_step evidence_notes contains unsafe/uncertain language")

    if not change.parts and not change.tools:
        errors.append("create_step has no parts or tools")

    metadata = change.metadata or {}

    if metadata.get("safety_guard_converted") is True:
        errors.append("create_step was previously converted by safety guard")

    if metadata.get("requires_manual_inspection") is True:
        errors.append("create_step requires manual inspection")

    return errors


def _apply_create_step(
    db: Session,
    job_id: UUID,
    change: StepRepairProposedChange,
) -> dict:
    errors = _validate_create_step_change(change)

    if errors:
        return _reject_change(
            change=change,
            reason="; ".join(errors),
        )

    job = db.get(Job, job_id)

    if job is None:
        return _reject_change(
            change=change,
            reason="create_step target job not found",
        )

    task_id = _get_existing_task_id_for_job(
        db=db,
        job_id=job_id,
    )

    new_step_kwargs = {
        "tenant_id": job.tenant_id,
        "job_id": job_id,
        "document_id": job.document_id,
        "step_number": change.step_number,
        "order_index": change.order_index,
        "source_page_number": change.source_page_number,
        "action": change.action,
        "parts": change.parts or [],
        "tools": change.tools or [],
        "confidence": change.confidence,
        "review_status": "edited",
        "reviewer_notes": (
            "Created by validation-driven repair apply route. "
            "Human review recommended before Gate 1 approval."
        ),
    }

    if task_id is not None:
        new_step_kwargs["task_id"] = task_id

    new_step = Step(**new_step_kwargs)

    db.add(new_step)
    db.flush()

    return _accept_change(
        change=change,
        message="Created missing step from grounded repair proposal.",
        affected_step_ids=[str(new_step.id)],
    )


def _apply_delete_step(
    db: Session,
    job_id: UUID,
    change: StepRepairProposedChange,
) -> dict:
    step = _get_step_or_none(
        db=db,
        step_id=change.step_id,
    )

    if step is None:
        return _reject_change(
            change=change,
            reason="delete_step target step_id not found",
        )

    if step.job_id != job_id:
        return _reject_change(
            change=change,
            reason="delete_step target step does not belong to this job",
        )

    deleted_step_id = str(step.id)

    db.delete(step)
    db.flush()

    return _accept_change(
        change=change,
        message="Deleted extracted step.",
        affected_step_ids=[deleted_step_id],
    )


def _apply_edit_step(
    db: Session,
    job_id: UUID,
    change: StepRepairProposedChange,
) -> dict:
    step = _get_step_or_none(
        db=db,
        step_id=change.step_id,
    )

    if step is None:
        return _reject_change(
            change=change,
            reason="edit_step target step_id not found",
        )

    if step.job_id != job_id:
        return _reject_change(
            change=change,
            reason="edit_step target step does not belong to this job",
        )

    if change.action is not None:
        if not change.action.strip():
            return _reject_change(
                change=change,
                reason="edit_step action is empty",
            )

        if _contains_unsafe_language(change.action):
            return _reject_change(
                change=change,
                reason="edit_step action contains unsafe/uncertain language",
            )

        step.action = change.action

    if change.parts:
        step.parts = change.parts

    if change.tools:
        step.tools = change.tools

    if change.step_number is not None:
        step.step_number = change.step_number

    if change.source_page_number is not None:
        step.source_page_number = change.source_page_number

    if change.confidence is not None:
        step.confidence = change.confidence

    step.review_status = "edited"

    note = "Edited by validation-driven repair apply route."

    if step.reviewer_notes:
        if note not in step.reviewer_notes:
            step.reviewer_notes = f"{step.reviewer_notes}\n{note}"
    else:
        step.reviewer_notes = note

    db.flush()

    return _accept_change(
        change=change,
        message="Edited existing step.",
        affected_step_ids=[str(step.id)],
    )


def _apply_merge_steps(
    db: Session,
    job_id: UUID,
    change: StepRepairProposedChange,
) -> dict:
    if not change.step_ids or len(change.step_ids) < 2:
        return _reject_change(
            change=change,
            reason="merge_steps requires at least two step_ids",
        )

    survivor_id = change.step_id or change.step_ids[0]

    survivor = _get_step_or_none(
        db=db,
        step_id=survivor_id,
    )

    if survivor is None:
        return _reject_change(
            change=change,
            reason="merge_steps survivor step_id not found",
        )

    if survivor.job_id != job_id:
        return _reject_change(
            change=change,
            reason="merge_steps survivor does not belong to this job",
        )

    merged_step_ids: list[str] = []

    if change.action is not None:
        if not change.action.strip():
            return _reject_change(
                change=change,
                reason="merge_steps merged action is empty",
            )

        if _contains_unsafe_language(change.action):
            return _reject_change(
                change=change,
                reason="merge_steps merged action contains unsafe/uncertain language",
            )

        survivor.action = change.action

    if change.parts:
        survivor.parts = change.parts

    if change.tools:
        survivor.tools = change.tools

    if change.step_number is not None:
        survivor.step_number = change.step_number

    if change.source_page_number is not None:
        survivor.source_page_number = change.source_page_number

    if change.order_index is not None:
        survivor.order_index = change.order_index

    if change.confidence is not None:
        survivor.confidence = change.confidence

    survivor.review_status = "edited"

    note = (
        "Merged by validation-driven repair apply route. "
        f"Merged from step IDs: {change.step_ids}"
    )

    if survivor.reviewer_notes:
        if note not in survivor.reviewer_notes:
            survivor.reviewer_notes = f"{survivor.reviewer_notes}\n{note}"
    else:
        survivor.reviewer_notes = note

    for step_id in change.step_ids:
        if step_id == survivor_id:
            continue

        step = _get_step_or_none(
            db=db,
            step_id=step_id,
        )

        if step is None:
            continue

        if step.job_id != job_id:
            continue

        merged_step_ids.append(str(step.id))
        db.delete(step)

    db.flush()

    return _accept_change(
        change=change,
        message="Merged duplicate steps into survivor step and deleted duplicate rows.",
        affected_step_ids=[str(survivor.id)] + merged_step_ids,
    )


def _apply_renumber_step(
    db: Session,
    job_id: UUID,
    change: StepRepairProposedChange,
) -> dict:
    step = _get_step_or_none(
        db=db,
        step_id=change.step_id,
    )

    if step is None:
        return _reject_change(
            change=change,
            reason="renumber_step target step_id not found",
        )

    if step.job_id != job_id:
        return _reject_change(
            change=change,
            reason="renumber_step target step does not belong to this job",
        )

    if change.step_number is not None:
        step.step_number = change.step_number

    if change.order_index is not None:
        step.order_index = change.order_index

    step.review_status = "edited"

    db.flush()

    return _accept_change(
        change=change,
        message="Updated step number/order index.",
        affected_step_ids=[str(step.id)],
    )


def _recompute_order_indexes(
    db: Session,
    job_id: UUID,
) -> None:
    """
    Backend owns final ordering.

    We sort primarily by step_number and secondarily by current order_index.
    This avoids relying too much on Claude-generated renumber_step operations.
    """
    steps = _get_steps_for_job(
        db=db,
        job_id=job_id,
    )

    steps = sorted(
        steps,
        key=lambda step: (
            step.step_number if step.step_number is not None else 9999,
            step.order_index if step.order_index is not None else 9999,
        ),
    )

    for index, step in enumerate(steps, start=1):
        step.order_index = index

    db.flush()


def _apply_single_change(
    db: Session,
    job_id: UUID,
    change: StepRepairProposedChange,
) -> dict:
    if change.operation == "create_step":
        return _apply_create_step(
            db=db,
            job_id=job_id,
            change=change,
        )

    if change.operation == "delete_step":
        return _apply_delete_step(
            db=db,
            job_id=job_id,
            change=change,
        )

    if change.operation == "edit_step":
        return _apply_edit_step(
            db=db,
            job_id=job_id,
            change=change,
        )

    if change.operation == "merge_steps":
        return _apply_merge_steps(
            db=db,
            job_id=job_id,
            change=change,
        )

    if change.operation == "renumber_step":
        # We allow it, but final order_index is recomputed after all changes.
        return _apply_renumber_step(
            db=db,
            job_id=job_id,
            change=change,
        )

    if change.operation in {"no_change", "needs_manual_review"}:
        return _reject_change(
            change=change,
            reason=f"{change.operation} is not an applyable database operation",
        )

    return _reject_change(
        change=change,
        reason=f"Unsupported operation: {change.operation}",
    )


def apply_step_repair_proposal(
    db: Session,
    job_id: UUID,
    proposal: StepRepairProposal,
) -> dict:
    route_job_id = _normalize_uuid(job_id, "route job_id")
    proposal_job_id = _normalize_uuid(proposal.job_id, "proposal job_id")

    job = db.get(Job, route_job_id)

    if job is None:
        raise ValueError("Job not found")

    if proposal_job_id != route_job_id:
        raise ValueError(
            "Proposal job_id does not match route job_id. "
            f"proposal_job_id={proposal_job_id}, route_job_id={route_job_id}"
        )

    proposal_document_id = _normalize_uuid(proposal.document_id, "proposal document_id")
    job_document_id = _normalize_uuid(job.document_id, "job document_id")

    if proposal_document_id != job_document_id:
        raise ValueError(
            "Proposal document_id does not match job document_id. "
            f"proposal_document_id={proposal_document_id}, job_document_id={job_document_id}"
        )

    before_validation = validate_extracted_steps_for_job(
        db=db,
        job_id=route_job_id,
    )

    applied_changes: list[dict] = []
    rejected_changes: list[dict] = []

    try:
        for change in proposal.proposed_changes:
            result = _apply_single_change(
                db=db,
                job_id=route_job_id,
                change=change,
            )

            if result.get("applied") is True:
                applied_changes.append(result)
            else:
                rejected_changes.append(result)

        if applied_changes:
            _recompute_order_indexes(
                db=db,
                job_id=route_job_id,
            )

        db.commit()

    except Exception:
        db.rollback()
        raise

    after_validation = validate_extracted_steps_for_job(
        db=db,
        job_id=route_job_id,
    )

    return {
        "job_id": str(job.id),
        "document_id": str(job.document_id),
        "applied_change_count": len(applied_changes),
        "rejected_change_count": len(rejected_changes),
        "applied_changes": applied_changes,
        "rejected_changes": rejected_changes,
        "before_validation": before_validation,
        "after_validation": after_validation,
    }
