from pydantic import BaseModel, Field


class LLMExtractedStep(BaseModel):
    step_number: int | None = None
    task_title: str | None = None

    action: str = Field(..., min_length=1)

    parts: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    quantities: list[str] = Field(default_factory=list)

    orientation: str | None = None
    warning: str | None = None
    expected_result: str | None = None

    source_evidence_ids: list[str] = Field(default_factory=list)
    visual_evidence_ids: list[str] = Field(default_factory=list)
    text_evidence_ids: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_attention_reason: str | None = None


class LLMPageExtractionResponse(BaseModel):
    page_number: int
    contains_assembly_steps: bool = False
    extracted_steps: list[LLMExtractedStep] = Field(default_factory=list)
    page_summary: str | None = None
    unresolved_questions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LLMManualExtractionResult(BaseModel):
    pages: list[LLMPageExtractionResponse] = Field(default_factory=list)
    total_steps: int = 0
    model_used: str
    fallback_used: bool = False
    warnings: list[str] = Field(default_factory=list)

    input_tokens: int | None = None
    output_tokens: int | None = None