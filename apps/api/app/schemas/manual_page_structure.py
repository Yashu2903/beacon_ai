import uuid

from pydantic import BaseModel, ConfigDict


class ManualPageStructureResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    page_type: str
    visible_step_numbers: list[int]
    confidence: float
    metadata_json: dict