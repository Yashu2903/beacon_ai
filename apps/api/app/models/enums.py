import enum

class JobState(str, enum.Enum):
    CREATED = "created"
    INGESTING = "ingesting"
    EXTRACTING_EVIDENCE = "extracting_evidence"
    EXTRACTING_STEPS = "extracting_steps"
    AWAITING_REVIEW_1 = "awaiting_review_1"
    STORYBOARDING = "storyboarding"
    AWAITING_REVIEW_2 = "awaiting_review_2"
    DETERMINISTIC_RENDERING = "deterministic_rendering"
    OPTIONAL_GENERATING = "optional_generating"
    COMPOSING = "composing"
    AWAITING_REVIEW_3 = "awaiting_review_3"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    NEEDS_ATTENTION = "needs_attention"
    FAILED = "failed"


