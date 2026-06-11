from uuid import UUID

from sqlalchemy.orm import Session

from app.models.enums import JobState
from app.models.job import Job
from app.models.step import Step
from app.schemas.extraction_validation import ExtractionValidationResult
from app.services.extraction_validation import validate_extracted_steps_for_job
from app.services.job_state import transition_job_state


BLOCKING_ISSUE_TYPES = {
    "duplicate_step_number",
    "missing_visible_step",
    "empty_step_action",
}


CONDITIONALLY_BLOCKING_ISSUE_TYPES = {
    "step_on_non_assembly_page",
}


BLOCKING_NON_ASSEMBLY_PAGE_TYPES = {
    "cover",
    "parts_inventory",
    "back_matter",
}


class Gate1ApprovalError(Exception):
    def __init__(self, message: str, validation: ExtractionValidationResult):
        super().__init__(message)
        self.message = message
        self.validation = validation


def _is_blocking_issue(issue) -> bool:
    """
    Decide whether a validation issue should block Gate 1.

    Errors like duplicate step numbers and missing visible steps always block.
    Some warnings only block when they are severe enough, for example a step
    extracted from cover, inventory, or back-matter page.
    """
    if issue.issue_type in BLOCKING_ISSUE_TYPES:
        return True

    if issue.severity == "error":
        return True

    if issue.issue_type in CONDITIONALLY_BLOCKING_ISSUE_TYPES:
        page_type = (issue.metadata or {}).get("page_type")

        if page_type in BLOCKING_NON_ASSEMBLY_PAGE_TYPES:
            return True

    return False


def get_gate_1_blocking_issues(
    validation: ExtractionValidationResult,
) -> list:
    return [
        issue
        for issue in validation.issues
        if _is_blocking_issue(issue)
    ]


def approve_gate_1_or_raise(
    db: Session,
    job_id: UUID,
) -> ExtractionValidationResult:
    """
    Approve Gate 1 only when validation has no blocking issues.

    This should be called by POST /jobs/{job_id}/gate-1/approve.
    """
    job = db.get(Job, job_id)

    if job is None:
        raise ValueError("Job not found")

    validation = validate_extracted_steps_for_job(
        db=db,
        job_id=job_id,
    )

    blocking_issues = get_gate_1_blocking_issues(validation)

    if blocking_issues:
        raise Gate1ApprovalError(
            message=(
                "Gate 1 approval blocked. Resolve blocking extraction issues "
                "before moving to storyboarding."
            ),
            validation=validation,
        )

    steps = (
        db.query(Step)
        .filter(Step.job_id == job_id)
        .order_by(Step.order_index.asc())
        .all()
    )

    if not steps:
        raise Gate1ApprovalError(
            message="Gate 1 approval blocked. No extracted steps found.",
            validation=validation,
        )

    unapproved_steps = [
        step
        for step in steps
        if step.review_status not in {"approved", "edited"}
    ]

    if unapproved_steps:
        raise Gate1ApprovalError(
            message=(
                "Gate 1 approval blocked. All steps must be approved or edited "
                "before moving to storyboarding."
            ),
            validation=validation,
        )

    transition_job_state(
        db=db,
        job_id=job.id,
        to_state=JobState.STORYBOARDING,
        actor_type="system",
        message="Gate 1 approved after extraction validation.",
    )

    return validation