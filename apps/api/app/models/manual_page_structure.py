import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SQLEnum, ForeignKey, Integer, JSON, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ManualPageType(str, Enum):
    cover = "cover"
    parts_inventory = "parts_inventory"
    assembly_step = "assembly_step"
    mixed_inventory_and_step = "mixed_inventory_and_step"
    informational = "informational"
    back_matter = "back_matter"
    unknown = "unknown"


class ManualPageStructure(Base):
    __tablename__ = "manual_page_structures"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    page_type: Mapped[ManualPageType] = mapped_column(
        SQLEnum(ManualPageType, name="manual_page_type"),
        nullable=False,
        default=ManualPageType.unknown,
    )

    visible_step_numbers: Mapped[list[int]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
    )

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    metadata_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )