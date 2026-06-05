import uuid

from pydantic import BaseModel

from app.models.enums import JobState


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    state: JobState