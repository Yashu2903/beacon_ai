import time
import uuid

from app.core.database import SessionLocal
from app.models.enums import JobState
from app.services.job_state import transition_job_state
from worker.celery_app import celery_app


@celery_app.task(name="worker.tasks.ingest_manual")
def ingest_manual(job_id: str):
    parsed_job_id = uuid.UUID(job_id)

    db = SessionLocal()

    try:
        transition_job_state(
            db,
            job_id=parsed_job_id,
            to_state=JobState.INGESTING,
            actor_type="worker",
            message="Skeleton ingest started.",
        )

        # Phase 1 placeholder.
        # In Phase 2, this becomes:
        # - PDF parsing
        # - page image rendering
        # - diagram extraction
        # - OCR fallback
        # - source evidence creation
        time.sleep(2)

        transition_job_state(
            db,
            job_id=parsed_job_id,
            to_state=JobState.EXTRACTING_EVIDENCE,
            actor_type="worker",
            message="Skeleton ingest complete. Ready for Phase 2 evidence extraction.",
        )

    except Exception as exc:
        try:
            transition_job_state(
                db,
                job_id=parsed_job_id,
                to_state=JobState.FAILED,
                actor_type="worker",
                message=str(exc),
            )
        except Exception:
            pass

        raise

    finally:
        db.close()