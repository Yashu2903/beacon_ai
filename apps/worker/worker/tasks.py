import uuid

from app.core.database import SessionLocal
from app.models.enums import JobState
from app.models.job import Job
from app.services.job_state import transition_job_state
from app.services.pdf_ingest import ingest_pdf_document
from worker.celery_app import celery_app


MIN_TEXT_SPANS_FOR_EXTRACTION = 3


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
            message="PDF ingest started.",
        )

        job = db.get(Job, parsed_job_id)

        if job is None:
            raise ValueError(f"Job not found: {parsed_job_id}")

        transition_job_state(
            db,
            job_id=parsed_job_id,
            to_state=JobState.EXTRACTING_EVIDENCE,
            actor_type="worker",
            message="Starting source evidence extraction.",
        )

        result = ingest_pdf_document(db, job.document_id)

        job = db.get(Job, parsed_job_id)

        if job is None:
            raise ValueError(f"Job not found after ingest: {parsed_job_id}")

        document = job.document

        if result.text_span_count < MIN_TEXT_SPANS_FOR_EXTRACTION:
            document.suitability_status = "needs_attention_low_text"
            job.failure_reason = (
                "Very little extractable text was found. "
                "This may be a scanned PDF or unsupported layout. "
                "OCR fallback is needed before step extraction."
            )
            db.commit()

            transition_job_state(
                db,
                job_id=parsed_job_id,
                to_state=JobState.NEEDS_ATTENTION,
                actor_type="worker",
                message=(
                    "Source evidence extraction found too little text. "
                    f"text_spans={result.text_span_count}, "
                    f"diagram_regions={result.diagram_region_count}, "
                    f"scanned_flag={result.scanned_flag}"
                ),
            )
            return

        document.suitability_status = "evidence_ready"
        db.commit()

        transition_job_state(
            db,
            job_id=parsed_job_id,
            to_state=JobState.EXTRACTING_STEPS,
            actor_type="worker",
            message=(
                "Source evidence extraction complete. "
                f"pages={result.page_count}, "
                f"text_spans={result.text_span_count}, "
                f"embedded_images={result.embedded_image_count}, "
                f"diagram_regions={result.diagram_region_count}, "
                f"scanned_flag={result.scanned_flag}"
            ),
        )

    except Exception as exc:
        db.rollback()

        job = db.get(Job, parsed_job_id)

        if job:
            job.failure_reason = str(exc)
            db.commit()

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
