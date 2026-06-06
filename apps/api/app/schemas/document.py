import uuid
from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    storage_key: str
    content_type: str
    size_bytes: int
    page_count: int | None
    scanned_flag: bool
    detected_language: str | None
    suitability_status: str | None
    created_at: datetime

    class Config:
        from_attributes = True
