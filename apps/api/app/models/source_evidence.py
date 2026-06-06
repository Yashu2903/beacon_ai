import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SourceEvidence(Base):
    __tablename__ = "source_evidence"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id"),
        index=True,
    )

    document_page_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_pages.id"),
        nullable=True,
        index=True,
    )

    evidence_type: Mapped[str] = mapped_column(String(64), index=True)

    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)

    text_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    x: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    y: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    width: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    height: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)

    coordinate_system: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default="pdf_points_top_left",
    )

    storage_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)

    validation_flag: Mapped[str | None] = mapped_column(String(64), nullable=True)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="source_evidence")
    document_page = relationship("DocumentPage")