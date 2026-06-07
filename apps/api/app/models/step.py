import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Step(Base):
    __tablename__ = "steps"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id"),
        index=True,
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("tasks.id"),
        nullable=True,
        index=True,
    )

    step_number: Mapped[int] = mapped_column(Integer, index=True)
    order_index: Mapped[int] = mapped_column(Integer, index=True)

    action: Mapped[str] = mapped_column(Text)
    parts: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    tools: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    quantity: Mapped[str | None] = mapped_column(String(128), nullable=True)
    orientation: Mapped[str | None] = mapped_column(Text, nullable=True)
    warning: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_result: Mapped[str | None] = mapped_column(Text, nullable=True)

    source_page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)

    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)

    extraction_method: Mapped[str] = mapped_column(String(64), default="heuristic")
    review_status: Mapped[str] = mapped_column(String(64), default="pending")

    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    task = relationship("Task", back_populates="steps")
    evidence_links = relationship("StepSourceEvidence", back_populates="step")