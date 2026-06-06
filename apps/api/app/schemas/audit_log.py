import uuid
from datetime import datetime

from pydantic import BaseModel


class AuditLogResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID | None
    actor_type: str
    action: str
    message: str | None
    created_at: datetime

    class Config:
        from_attributes = True

