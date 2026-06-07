import uuid
from datetime import datetime

from pydantic import BaseModel


class ReviewEventResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    gate_number: int
    target_type: str
    target_id: uuid.UUID | None
    action: str
    time_spent_seconds: int | None
    click_count: int | None
    fields_edited: list | None
    notes: str | None
    created_at: datetime

    class Config:
        from_attributes = True