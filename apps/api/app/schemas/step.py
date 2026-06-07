import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class StepEvidenceLinkResponse(BaseModel):
    id: uuid.UUID
    source_evidence_id: uuid.UUID
    link_type: str

    class Config:
        from_attributes = True


class StepResponse(BaseModel):
    id: uuid.UUID
    job_id: uuid.UUID
    document_id: uuid.UUID
    task_id: uuid.UUID | None

    step_number: int
    order_index: int

    action: str
    parts: list | None
    tools: list | None
    quantity: str | None
    orientation: str | None
    warning: str | None
    expected_result: str | None

    source_page_number: int | None
    confidence: Decimal | None
    extraction_method: str
    review_status: str
    reviewer_notes: str | None

    created_at: datetime
    updated_at: datetime

    evidence_links: list[StepEvidenceLinkResponse] = Field(default_factory=list)

    class Config:
        from_attributes = True


class StepUpdateRequest(BaseModel):
    action: str | None = None
    parts: list | None = None
    tools: list | None = None
    quantity: str | None = None
    orientation: str | None = None
    warning: str | None = None
    expected_result: str | None = None
    reviewer_notes: str | None = None


class StepReorderItem(BaseModel):
    step_id: uuid.UUID
    order_index: int
    step_number: int


class StepReorderRequest(BaseModel):
    items: list[StepReorderItem]


class Gate1ApproveRequest(BaseModel):
    time_spent_seconds: int | None = None
    click_count: int | None = None
    notes: str | None = None