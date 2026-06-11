from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class StepRepairProposedChange(BaseModel):
    operation: Literal[
        "merge_steps",
        "delete_step",
        "create_step",
        "edit_step",
        "renumber_step",
        "no_change",
        "needs_manual_review",
    ]

    reason: str

    step_id: str | None = None
    step_ids: list[str] = Field(default_factory=list)

    step_number: int | None = None
    source_page_number: int | None = None
    order_index: int | None = None

    action: str | None = None
    parts: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

    confidence: float | None = None
    evidence_notes: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class StepRepairProposal(BaseModel):
    job_id: str
    document_id: str

    summary: str
    can_apply_automatically: bool = False
    requires_human_review: bool = True

    proposed_changes: list[StepRepairProposedChange] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)
    raw_model_response: dict[str, Any] | None = None

    @field_validator("job_id", "document_id", mode="before")
    @classmethod
    def _coerce_uuid_fields_to_string(cls, value: Any) -> str:
        if isinstance(value, UUID):
            return str(value)

        if isinstance(value, str):
            return value.strip()

        return str(value).strip()
