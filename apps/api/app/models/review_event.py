import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ReviewEvent(Base):
    __tablename__ = "review_events"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id"), index=True)

    gate_number: Mapped[int] = mapped_column(Integer, index=True)

    reviewer_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    target_type: Mapped[str] = mapped_column(String(64))
    target_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True, index=True)

    action: Mapped[str] = mapped_column(String(128))

    time_spent_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    click_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    fields_edited: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    before_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    after_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)