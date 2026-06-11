import uuid

from app.core.database import SessionLocal
from app.models.enums import JobState
from app.models.job import Job
from app.services.extraction_validation import validate_extracted_steps_for_job
from app.services.job_state import transition_job_state
from app.services.manual_structure import detect_manual_structure_for_document
from app.services.pdf_ingest import ingest_pdf_document
from app.services.step_extraction import extract_steps_for_job_with_provider
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

        ingest_result = ingest_pdf_document(db, job.document_id)

        job = db.get(Job, parsed_job_id)

        if job is None:
            raise ValueError(f"Job not found after ingest: {parsed_job_id}")

        document = job.document

        has_too_little_text = (
            ingest_result.text_span_count < MIN_TEXT_SPANS_FOR_EXTRACTION
        )
        has_visual_evidence = (
            ingest_result.diagram_region_count > 0 or ingest_result.page_count > 0
        )

        if has_too_little_text and not has_visual_evidence:
            document.suitability_status = "needs_attention_low_evidence"
            job.failure_reason = (
                "Very little usable evidence was found. "
                "The PDF may be scanned, corrupted, or unsupported. "
                "Step extraction cannot continue without text or visual evidence."
            )
            db.commit()

            transition_job_state(
                db,
                job_id=parsed_job_id,
                to_state=JobState.NEEDS_ATTENTION,
                actor_type="worker",
                message=(
                    "Source evidence extraction found too little usable evidence. "
                    f"text_spans={ingest_result.text_span_count}, "
                    f"diagram_regions={ingest_result.diagram_region_count}, "
                    f"pages={ingest_result.page_count}, "
                    f"scanned_flag={ingest_result.scanned_flag}"
                ),
            )
            return

        document.suitability_status = "evidence_ready"
        db.commit()

        manual_structure = detect_manual_structure_for_document(
            db=db,
            document_id=job.document_id,
        )

        transition_job_state(
            db,
            job_id=parsed_job_id,
            to_state=JobState.EXTRACTING_STEPS,
            actor_type="worker",
            message=(
                "Source evidence extraction complete. "
                f"pages={ingest_result.page_count}, "
                f"text_spans={ingest_result.text_span_count}, "
                f"embedded_images={ingest_result.embedded_image_count}, "
                f"diagram_regions={ingest_result.diagram_region_count}, "
                f"manual_structure_pages={len(manual_structure)}, "
                f"scanned_flag={ingest_result.scanned_flag}. "
                "Starting provider-based step extraction."
            ),
        )

        step_count = extract_steps_for_job_with_provider(db, parsed_job_id)

        if step_count == 0:
            job = db.get(Job, parsed_job_id)
            if job:
                job.failure_reason = (
                    "No assembly steps were extracted from the manual evidence. "
                    "Manual may need better OCR, better diagram parsing, or LLM review."
                )
                db.commit()

            transition_job_state(
                db,
                job_id=parsed_job_id,
                to_state=JobState.NEEDS_ATTENTION,
                actor_type="worker",
                message="Step extraction produced zero steps.",
            )
            return

        validation = validate_extracted_steps_for_job(
            db=db,
            job_id=parsed_job_id,
        )

        transition_job_state(
            db,
            job_id=parsed_job_id,
            to_state=JobState.AWAITING_REVIEW_1,
            actor_type="worker",
            message=(
                "Step extraction complete. "
                f"steps={step_count}. "
                f"validation_errors={validation.error_count}. "
                "Awaiting Gate 1 review."
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
