import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"), index=True)

    original_filename: Mapped[str] = mapped_column(String(512))
    storage_key: Mapped[str] = mapped_column(String(1024), unique=True)

    content_type: Mapped[str] = mapped_column(String(128), default="application/pdf")
    size_bytes: Mapped[int] = mapped_column(Integer)

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scanned_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    detected_language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    suitability_status: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    jobs = relationship("Job", back_populates="document")
