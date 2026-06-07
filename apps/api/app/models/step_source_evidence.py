import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class StepSourceEvidence(Base):
    __tablename__ = "step_source_evidence"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("steps.id"), index=True)
    source_evidence_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("source_evidence.id"),
        index=True,
    )

    link_type: Mapped[str] = mapped_column(String(64), default="supports_step")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    step = relationship("Step", back_populates="evidence_links")