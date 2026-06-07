import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from app.core.database import get_db
from app.models.enums import JobState
from app.models.job import Job
from app.models.review_event import ReviewEvent
from app.models.step import Step
from app.models.task import Task
from app.schemas.step import (
    Gate1ApproveRequest,
    StepReorderRequest,
    StepResponse,
    StepUpdateRequest,
)
from app.schemas.task import TaskResponse
from app.services.job_state import transition_job_state

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
    payload: Gate1ApproveRequest,
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state != JobState.AWAITING_REVIEW_1:
        raise HTTPException(
            status_code=400,
            detail=f"Job must be awaiting_review_1, current state is {job.state.value}",
        )

    steps = db.query(Step).filter(Step.job_id == job_id).all()

    if not steps:
        raise HTTPException(status_code=400, detail="No steps found for this job")

    unapproved = [
        step.id
        for step in steps
        if step.review_status not in {"approved", "edited"}
    ]

    if unapproved:
        raise HTTPException(
            status_code=400,
            detail="All steps must be approved or edited before Gate 1 approval.",
        )

    review_event = ReviewEvent(
        tenant_id=job.tenant_id,
        job_id=job.id,
        gate_number=1,
        target_type="job",
        target_id=job.id,
        action="approve_gate_1",
        time_spent_seconds=payload.time_spent_seconds,
        click_count=payload.click_count,
        notes=payload.notes,
    )

    db.add(review_event)
    db.commit()

    transition_job_state(
        db,
        job_id=job.id,
        to_state=JobState.STORYBOARDING,
        actor_type="reviewer",
        message="Gate 1 approved. Steps are ready for storyboard generation.",
    )

    return {
        "job_id": str(job.id),
        "state": JobState.STORYBOARDING.value,
        "message": "Gate 1 approved. Job moved to storyboarding.",
    }