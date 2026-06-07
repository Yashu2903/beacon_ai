import uuid
from datetime import datetime

from pydantic import BaseModel


class TaskResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    order_index: int
    review_status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True