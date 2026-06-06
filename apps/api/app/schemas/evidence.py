import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, computed_field


class DocumentPageResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    width: Decimal
    height: Decimal
    page_image_key: str | None
    text_extraction_status: str
    created_at: datetime

    @computed_field
    @property
    def page_image_url(self) -> str | None:
        if not self.page_image_key:
            return None
        return f"/storage/{self.page_image_key}"

    class Config:
        from_attributes = True


class SourceEvidenceResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    document_page_id: uuid.UUID | None
    evidence_type: str
    page_number: int | None
    text_start: int | None
    text_end: int | None
    extracted_text: str | None
    x: Decimal | None
    y: Decimal | None
    width: Decimal | None
    height: Decimal | None
    coordinate_system: str | None
    storage_key: str | None
    confidence: Decimal | None
    validation_flag: str | None
    metadata_json: dict | None
    created_at: datetime

    @computed_field
    @property
    def storage_url(self) -> str | None:
        if not self.storage_key:
            return None
        return f"/storage/{self.storage_key}"

    class Config:
        from_attributes = True