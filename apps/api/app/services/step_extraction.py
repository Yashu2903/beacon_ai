from __future__ import annotations

import re
import uuid
from uuid import UUID
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.orm import Session

from app.models.job import Job
from app.models.manual_page_structure import ManualPageStructure
from app.models.source_evidence import SourceEvidence
from app.models.step import Step
from app.models.step_source_evidence import StepSourceEvidence
from app.models.task import Task
from app.core.config import settings
from app.services.claude_manual_extractor import ClaudeManualExtractor
from app.services.llm_step_persistence import persist_llm_extraction_result
from app.services.manual_structure import detect_manual_structure_for_document


STEP_PATTERN = re.compile(
    r"^\s*(step\s*)?(\d{1,2})[\.\):\-]?\s+(.+)$",
    re.IGNORECASE,
)

WARNING_KEYWORDS = [
    "warning",
    "caution",
    "do not",
    "careful",
    "important",
]


@dataclass
class StepExtractionResult:
    step_count: int
    task_count: int


def _looks_like_step(text: str) -> re.Match | None:
    cleaned = " ".join(text.split())

    if len(cleaned) < 8:
        return None

    return STEP_PATTERN.match(cleaned)


def _extract_warning(text: str) -> str | None:
    lower = text.lower()

    if any(keyword in lower for keyword in WARNING_KEYWORDS):
        return text

    return None


def _extract_parts(text: str) -> list[str]:
    # Simple first-pass furniture part/code extraction.
    # Examples it may catch: A, B, P1, M6, screw, bolt, dowel, panel.
    candidates: list[str] = []

    code_matches = re.findall(r"\b[A-Z]\b|\b[A-Z]\d+\b|\bM\d+\b", text)

    for item in code_matches:
        if item not in candidates:
            candidates.append(item)

    part_words = [
        "panel",
        "screw",
        "bolt",
        "dowel",
        "cam",
        "nut",
        "washer",
        "bracket",
        "leg",
        "shelf",
        "frame",
        "rail",
    ]

    lower = text.lower()

    for word in part_words:
        if word in lower and word not in candidates:
            candidates.append(word)

    return candidates


def _extract_tools(text: str) -> list[str]:
    tool_words = [
        "allen key",
        "hex key",
        "screwdriver",
        "drill",
        "hammer",
        "wrench",
    ]

    lower = text.lower()
    return [tool for tool in tool_words if tool in lower]


def _extract_quantity(text: str) -> str | None:
    match = re.search(
        r"\b(\d+)\s*(x|pcs|pieces|piece|screws|bolts|dowels)?\b",
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    return match.group(0)


def _make_task_title(action: str, step_number: int) -> str:
    words = action.split()

    if len(words) <= 8:
        return action

    return f"Step {step_number}: {' '.join(words[:8])}"


def clear_existing_extraction(db: Session, job_id: uuid.UUID) -> None:
    existing_steps = db.query(Step).filter(Step.job_id == job_id).all()

    for step in existing_steps:
        db.query(StepSourceEvidence).filter(
            StepSourceEvidence.step_id == step.id
        ).delete()

    db.query(Step).filter(Step.job_id == job_id).delete()
    db.query(Task).filter(Task.job_id == job_id).delete()
    db.commit()


def extract_steps_for_job(db: Session, job_id: uuid.UUID) -> StepExtractionResult:
    job = db.get(Job, job_id)

    if job is None:
        raise ValueError(f"Job not found: {job_id}")

    clear_existing_extraction(db, job_id)

    text_evidence = (
        db.query(SourceEvidence)
        .filter(SourceEvidence.document_id == job.document_id)
        .filter(SourceEvidence.evidence_type == "page_text_span")
        .order_by(
            SourceEvidence.page_number.asc(),
            SourceEvidence.y.asc(),
            SourceEvidence.x.asc(),
        )
        .all()
    )

    extracted_steps: list[tuple[SourceEvidence, re.Match]] = []

    for evidence in text_evidence:
        if not evidence.extracted_text:
            continue

        match = _looks_like_step(evidence.extracted_text)

        if match:
            extracted_steps.append((evidence, match))

    task_count = 0
    step_count = 0

    for order_index, (evidence, match) in enumerate(extracted_steps):
        step_number = int(match.group(2))
        action = match.group(3).strip()

        task = Task(
            tenant_id=job.tenant_id,
            job_id=job.id,
            document_id=job.document_id,
            title=_make_task_title(action, step_number),
            order_index=order_index,
            review_status="pending",
        )

        db.add(task)
        db.flush()
        task_count += 1

        step = Step(
            tenant_id=job.tenant_id,
            job_id=job.id,
            document_id=job.document_id,
            task_id=task.id,
            step_number=step_number,
            order_index=order_index,
            action=action,
            parts=_extract_parts(action),
            tools=_extract_tools(action),
            quantity=_extract_quantity(action),
            orientation=None,
            warning=_extract_warning(action),
            expected_result=None,
            source_page_number=evidence.page_number,
            confidence=Decimal("0.6500"),
            extraction_method="heuristic_v1",
            review_status="pending",
        )

        db.add(step)
        db.flush()

        link = StepSourceEvidence(
            tenant_id=job.tenant_id,
            step_id=step.id,
            source_evidence_id=evidence.id,
            link_type="primary_text_span",
        )

        db.add(link)
        step_count += 1

    db.commit()

    return StepExtractionResult(
        step_count=step_count,
        task_count=task_count,
    )


def extract_steps_for_job_with_provider(db: Session, job_id: UUID) -> int:
    job = db.query(Job).filter(Job.id == job_id).first()

    if not job:
        raise ValueError(f"Job not found: {job_id}")

    provider = settings.llm_extractor_provider.strip().lower()

    if provider in {"claude", "anthropic"} and settings.anthropic_api_key:
        existing_structure_count = (
            db.query(ManualPageStructure)
            .filter(ManualPageStructure.document_id == job.document_id)
            .count()
        )

        if existing_structure_count == 0:
            detect_manual_structure_for_document(
                db=db,
                document_id=job.document_id,
            )

        extractor = ClaudeManualExtractor()

        extraction_result = extractor.extract_document(
            db=db,
            document_id=job.document_id,
            job_id=job.id,
        )

        extraction_method = (
            "claude_fallback_v1"
            if extraction_result.fallback_used
            else "claude_primary_v1"
        )

        return persist_llm_extraction_result(
            db=db,
            job_id=job.id,
            extraction_result=extraction_result,
            extraction_method=extraction_method,
        )

    return extract_steps_for_job(db=db, job_id=job_id)
