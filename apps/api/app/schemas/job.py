import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.enums import JobState


class JobResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    state: JobState
    failure_reason: str | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True