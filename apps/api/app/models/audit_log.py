import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("jobs.id"),
        nullable=True,
        index=True,
    )

    actor_type: Mapped[str] = mapped_column(String(64), default="system")
    action: Mapped[str] = mapped_column(String(128))
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

