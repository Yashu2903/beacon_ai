from collections import defaultdict
from uuid import UUID

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.source_evidence import SourceEvidence
from app.models.step import Step
from app.models.step_source_evidence import StepSourceEvidence
from app.models.task import Task
from app.schemas.llm_manual import LLMManualExtractionResult


def _get_valid_evidence_map(
    db: Session,
    document_id: UUID,
    evidence_ids: set[str],
) -> dict[str, SourceEvidence]:
    if not evidence_ids:
        return {}

    parsed_ids: list[UUID] = []

    for evidence_id in evidence_ids:
        try:
            parsed_ids.append(UUID(evidence_id))
        except ValueError:
            continue

    if not parsed_ids:
        return {}

    evidence_rows = (
        db.query(SourceEvidence)
        .filter(
            SourceEvidence.document_id == document_id,
            SourceEvidence.id.in_(parsed_ids),
        )
        .all()
    )

    return {str(row.id): row for row in evidence_rows}


def _normalize_task_title(title: str | None) -> str:
    if not title or not title.strip():
        return "Furniture Assembly"

    return title.strip()


def persist_llm_extraction_result(
    db: Session,
    job_id: UUID,
    extraction_result: LLMManualExtractionResult,
    extraction_method: str,
) -> int:
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise ValueError(f"Job not found: {job_id}")

    document_id = job.document_id
    tenant_id = job.tenant_id

    # Avoid duplicate steps if extraction is re-run for same job.
    existing_steps = db.query(Step).filter(Step.job_id == job_id).all()

    for step in existing_steps:
        db.query(StepSourceEvidence).filter(
            StepSourceEvidence.step_id == step.id
        ).delete()

    db.query(Step).filter(Step.job_id == job_id).delete()
    db.query(Task).filter(Task.job_id == job_id).delete()
    db.flush()

    all_evidence_ids: set[str] = set()

    for page in extraction_result.pages:
        for extracted_step in page.extracted_steps:
            all_evidence_ids.update(extracted_step.source_evidence_ids)
            all_evidence_ids.update(extracted_step.visual_evidence_ids)
            all_evidence_ids.update(extracted_step.text_evidence_ids)

    valid_evidence_map = _get_valid_evidence_map(
        db=db,
        document_id=document_id,
        evidence_ids=all_evidence_ids,
    )

    task_by_title: dict[str, Task] = {}
    saved_steps = 0
    global_step_order = 1

    for page in sorted(extraction_result.pages, key=lambda item: item.page_number):
        for extracted_step in page.extracted_steps:
            task_title = _normalize_task_title(extracted_step.task_title)

            if task_title not in task_by_title:
                task = Task(
                    tenant_id=tenant_id,
                    job_id=job_id,
                    document_id=document_id,
                    title=task_title,
                    order_index=len(task_by_title) + 1,
                    review_status="pending",
                )
                db.add(task)
                db.flush()

                task_by_title[task_title] = task

            task = task_by_title[task_title]

            referenced_ids = set()
            referenced_ids.update(extracted_step.source_evidence_ids)
            referenced_ids.update(extracted_step.visual_evidence_ids)
            referenced_ids.update(extracted_step.text_evidence_ids)

            valid_ids = [
                evidence_id
                for evidence_id in referenced_ids
                if evidence_id in valid_evidence_map
            ]

            invalid_ids = [
                evidence_id
                for evidence_id in referenced_ids
                if evidence_id not in valid_evidence_map
            ]

            needs_attention_reason = extracted_step.needs_attention_reason

            review_status = "pending"

            if invalid_ids:
                review_status = "needs_attention"
                invalid_note = (
                    "Invalid source evidence IDs: " + ", ".join(invalid_ids)
                )
                needs_attention_reason = (
                    f"{needs_attention_reason}; {invalid_note}"
                    if needs_attention_reason
                    else invalid_note
                )

            if not valid_ids:
                review_status = "needs_attention"
                missing_note = "No valid source evidence linked to this step."
                needs_attention_reason = (
                    f"{needs_attention_reason}; {missing_note}"
                    if needs_attention_reason
                    else missing_note
                )

            if extracted_step.confidence < 0.70:
                review_status = "needs_attention"
                low_conf_note = (
                    f"Low LLM confidence: {extracted_step.confidence}"
                )
                needs_attention_reason = (
                    f"{needs_attention_reason}; {low_conf_note}"
                    if needs_attention_reason
                    else low_conf_note
                )

            quantity_text = None

            if extracted_step.quantities:
                quantity_text = ", ".join(extracted_step.quantities)

            step_number = extracted_step.step_number or global_step_order

            step = Step(
                tenant_id=tenant_id,
                job_id=job_id,
                document_id=document_id,
                task_id=task.id,
                step_number=step_number,
                order_index=global_step_order,
                action=extracted_step.action,
                parts=extracted_step.parts,
                tools=extracted_step.tools,
                quantity=quantity_text,
                orientation=extracted_step.orientation,
                warning=extracted_step.warning,
                expected_result=extracted_step.expected_result,
                source_page_number=page.page_number,
                confidence=extracted_step.confidence,
                extraction_method=extraction_method,
                review_status=review_status,
                reviewer_notes=needs_attention_reason,
            )

            db.add(step)
            db.flush()

            for evidence_id in valid_ids:
                source_evidence = valid_evidence_map[evidence_id]

                if evidence_id in extracted_step.visual_evidence_ids:
                    link_type = "visual"
                elif evidence_id in extracted_step.text_evidence_ids:
                    link_type = "text"
                else:
                    link_type = "source"

                link = StepSourceEvidence(
                    step_id=step.id,
                    source_evidence_id=source_evidence.id,
                    link_type=link_type,
                )

                db.add(link)

            saved_steps += 1
            global_step_order += 1

    db.commit()

    return saved_steps