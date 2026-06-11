import uuid
from typing import Any

from app.services.step_repair import build_or_apply_step_repair
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy.orm import Session, selectinload
from app.services.extraction_validation import validate_extracted_steps_for_job
from app.core.database import get_db
from app.models.enums import JobState
from app.models.job import Job
from app.models.review_event import ReviewEvent
from app.models.step import Step
from app.models.task import Task
from app.schemas.step_repair import StepRepairProposal
from app.services.step_repair_apply import apply_step_repair_proposal
from app.schemas.step import (
    Gate1ApproveRequest,
    StepReorderRequest,
    StepResponse,
    StepUpdateRequest,
)
from app.services.gate1_validation import (
    Gate1ApprovalError,
    approve_gate_1_or_raise,
    get_gate_1_blocking_issues,
)
from app.schemas.task import TaskResponse
from app.services.job_state import transition_job_state

from app.services.claude_step_repair import propose_step_repair_with_claude



router = APIRouter(tags=["gate-1-steps"])


def _serialize_step_before(step: Step) -> dict:
    return {
        "action": step.action,
        "parts": step.parts,
        "tools": step.tools,
        "quantity": step.quantity,
        "orientation": step.orientation,
        "warning": step.warning,
        "expected_result": step.expected_result,
        "review_status": step.review_status,
        "reviewer_notes": step.reviewer_notes,
    }


def _parse_step_repair_proposal_payload(payload: dict[str, Any]) -> StepRepairProposal:
    if "llm_repair_packet" in payload:
        raise ValueError(
            "Invalid repair apply payload: received the repair planning response. "
            "Call POST /jobs/{job_id}/steps/repair/propose and send that response body "
            "to /jobs/{job_id}/steps/repair/apply."
        )

    if "response_schema" in payload:
        raise ValueError(
            "Invalid repair apply payload: received the response_schema template, "
            "not an actual StepRepairProposal."
        )

    proposal_payload = payload.get("proposal", payload)

    if not isinstance(proposal_payload, dict):
        raise ValueError("Invalid repair apply payload: proposal body must be an object")

    if proposal_payload.get("job_id") == "string":
        raise ValueError(
            "Invalid proposal job_id: received the literal schema placeholder 'string'. "
            "Use the actual job_id from POST /jobs/{job_id}/steps/repair/propose."
        )

    if proposal_payload.get("document_id") == "string":
        raise ValueError(
            "Invalid proposal document_id: received the literal schema placeholder 'string'. "
            "Use the actual document_id from POST /jobs/{job_id}/steps/repair/propose."
        )

    try:
        return StepRepairProposal.model_validate(proposal_payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid repair proposal payload: {exc}") from exc


@router.get("/jobs/{job_id}/tasks", response_model=list[TaskResponse])
def list_job_tasks(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return (
        db.query(Task)
        .filter(Task.job_id == job_id)
        .order_by(Task.order_index.asc())
        .all()
    )

@router.post("/jobs/{job_id}/steps/repair/propose")
def propose_job_step_repair(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    return propose_step_repair_with_claude(
        db=db,
        job_id=job_id,
    )

@router.get("/jobs/{job_id}/steps", response_model=list[StepResponse])
def list_job_steps(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    steps = (
        db.query(Step)
        .options(selectinload(Step.evidence_links))
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .all()
    )

    return steps

@router.post("/jobs/{job_id}/steps/repair")
def repair_job_steps(
    job_id: uuid.UUID,
    apply_safe_fixes: bool = False,
    db: Session = Depends(get_db),
):
    return build_or_apply_step_repair(
        db=db,
        job_id=job_id,
        apply_safe_fixes=apply_safe_fixes,
    )


@router.post("/jobs/{job_id}/steps/repair/apply")
def apply_job_step_repair(
    job_id: uuid.UUID,
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
):
    try:
        proposal = _parse_step_repair_proposal_payload(payload)

        return apply_step_repair_proposal(
            db=db,
            job_id=job_id,
            proposal=proposal,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/steps/{step_id}", response_model=StepResponse)
def update_step(
    step_id: uuid.UUID,
    payload: StepUpdateRequest,
    db: Session = Depends(get_db),
):
    step = (
        db.query(Step)
        .options(selectinload(Step.evidence_links))
        .filter(Step.id == step_id)
        .first()
    )

    if step is None:
        raise HTTPException(status_code=404, detail="Step not found")

    before = _serialize_step_before(step)

    update_data = payload.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(step, field, value)

    step.review_status = "edited"

    after = _serialize_step_before(step)

    review_event = ReviewEvent(
        tenant_id=step.tenant_id,
        job_id=step.job_id,
        gate_number=1,
        target_type="step",
        target_id=step.id,
        action="edit_step",
        fields_edited=list(update_data.keys()),
        before_json=before,
        after_json=after,
    )

    db.add(review_event)
    db.commit()
    db.refresh(step)

    return step


@router.post("/steps/{step_id}/approve", response_model=StepResponse)
def approve_step(
    step_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    step = (
        db.query(Step)
        .options(selectinload(Step.evidence_links))
        .filter(Step.id == step_id)
        .first()
    )

    if step is None:
        raise HTTPException(status_code=404, detail="Step not found")

    step.review_status = "approved"

    review_event = ReviewEvent(
        tenant_id=step.tenant_id,
        job_id=step.job_id,
        gate_number=1,
        target_type="step",
        target_id=step.id,
        action="approve_step",
    )

    db.add(review_event)
    db.commit()
    db.refresh(step)

    return step


@router.post("/steps/{step_id}/reject", response_model=StepResponse)
def reject_step(
    step_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    step = (
        db.query(Step)
        .options(selectinload(Step.evidence_links))
        .filter(Step.id == step_id)
        .first()
    )

    if step is None:
        raise HTTPException(status_code=404, detail="Step not found")

    step.review_status = "rejected"

    review_event = ReviewEvent(
        tenant_id=step.tenant_id,
        job_id=step.job_id,
        gate_number=1,
        target_type="step",
        target_id=step.id,
        action="reject_step",
    )

    db.add(review_event)
    db.commit()
    db.refresh(step)

    return step


@router.post("/jobs/{job_id}/steps/reorder", response_model=list[StepResponse])
def reorder_steps(
    job_id: uuid.UUID,
    payload: StepReorderRequest,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    step_ids = [item.step_id for item in payload.items]

    steps = db.query(Step).filter(Step.job_id == job_id, Step.id.in_(step_ids)).all()
    steps_by_id = {step.id: step for step in steps}

    for item in payload.items:
        step = steps_by_id.get(item.step_id)

        if not step:
            continue

        step.order_index = item.order_index
        step.step_number = item.step_number

    review_event = ReviewEvent(
        tenant_id=job.tenant_id,
        job_id=job.id,
        gate_number=1,
        target_type="job",
        target_id=job.id,
        action="reorder_steps",
        after_json=payload.model_dump(mode="json"),
    )

    db.add(review_event)
    db.commit()

    return (
        db.query(Step)
        .options(selectinload(Step.evidence_links))
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .all()
    )


@router.post("/jobs/{job_id}/gate-1/approve")
def approve_gate_1(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    try:
        validation = approve_gate_1_or_raise(
            db=db,
            job_id=job_id,
        )

        return {
            "approved": True,
            "message": "Gate 1 approved. Job moved to storyboarding.",
            "blocking_issue_count": 0,
            "blocking_issues": [],
            "validation": validation,
        }

    except Gate1ApprovalError as error:
        blocking_issues = get_gate_1_blocking_issues(error.validation)

        return {
            "approved": False,
            "message": error.message,
            "blocking_issue_count": len(blocking_issues),
            "blocking_issues": blocking_issues,
            "validation": error.validation,
        }


@router.get("/jobs/{job_id}/extraction-validation")
def get_extraction_validation(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return validate_extracted_steps_for_job(db=db, job_id=job_id)
