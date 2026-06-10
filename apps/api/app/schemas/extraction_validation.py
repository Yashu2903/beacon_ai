from pydantic import BaseModel, Field


class StepValidationIssue(BaseModel):
    issue_type: str
    severity: str = "warning"  # info, warning, error
    page_number: int | None = None
    step_number: int | None = None
    message: str
    suggested_action: str | None = None
    related_step_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class DocumentStepStructure(BaseModel):
    visible_step_numbers_by_page: dict[int, list[int]] = Field(default_factory=dict)
    extracted_step_numbers: list[int] = Field(default_factory=list)
    duplicate_step_numbers: list[int] = Field(default_factory=list)
    missing_visible_step_numbers: list[int] = Field(default_factory=list)
    unexpected_step_numbers: list[int] = Field(default_factory=list)


class ExtractionValidationResult(BaseModel):
    job_id: str
    document_id: str
    issue_count: int
    error_count: int
    warning_count: int
    structure: DocumentStepStructure
    issues: list[StepValidationIssue] = Field(default_factory=list)