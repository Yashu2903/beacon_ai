import uuid

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog
from app.models.enums import JobState
from app.models.job import Job


ALLOWED_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.CREATED: {
        JobState.INGESTING,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.INGESTING: {
        JobState.EXTRACTING_EVIDENCE,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.EXTRACTING_EVIDENCE: {
        JobState.EXTRACTING_STEPS,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.EXTRACTING_STEPS: {
        JobState.AWAITING_REVIEW_1,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.AWAITING_REVIEW_1: {
        JobState.STORYBOARDING,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.STORYBOARDING: {
        JobState.AWAITING_REVIEW_2,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.AWAITING_REVIEW_2: {
        JobState.DETERMINISTIC_RENDERING,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.DETERMINISTIC_RENDERING: {
        JobState.OPTIONAL_GENERATING,
        JobState.COMPOSING,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.OPTIONAL_GENERATING: {
        JobState.COMPOSING,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.COMPOSING: {
        JobState.AWAITING_REVIEW_3,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.AWAITING_REVIEW_3: {
        JobState.EXPORTING,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.EXPORTING: {
        JobState.COMPLETED,
        JobState.NEEDS_ATTENTION,
        JobState.FAILED,
    },
    JobState.NEEDS_ATTENTION: set(JobState),
    JobState.FAILED: set(),
    JobState.COMPLETED: set(),
}


def transition_job_state(
    db: Session,
    *,
    job_id: uuid.UUID,
    to_state: JobState,
    actor_type: str = "system",
    message: str | None = None,
) -> Job:
    job = db.get(Job, job_id)

    if job is None:
        raise ValueError(f"Job not found: {job_id}")

    allowed_next_states = ALLOWED_TRANSITIONS.get(job.state, set())

    if to_state not in allowed_next_states:
        raise ValueError(
            f"Invalid state transition from {job.state.value} to {to_state.value}"
        )

    from_state = job.state
    job.state = to_state

    audit_log = AuditLog(
        tenant_id=job.tenant_id,
        job_id=job.id,
        actor_type=actor_type,
        action="job_state_transition",
        message=message or f"{from_state.value} -> {to_state.value}",
    )

    db.add(audit_log)
    db.commit()
    db.refresh(job)

    return job