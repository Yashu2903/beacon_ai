import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DocumentPage(Base):
    __tablename__ = "document_pages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True, default="demo")

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id"),
        index=True,
    )

    page_number: Mapped[int] = mapped_column(Integer, index=True)

    width: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    height: Mapped[Decimal] = mapped_column(Numeric(10, 2))

    page_image_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    text_extraction_status: Mapped[str] = mapped_column(
        String(64),
        default="pending",
    )

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="pages")
