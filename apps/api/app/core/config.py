from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Beacon AI API"

    database_url: str = (
        "postgresql+psycopg://beacon:beacon_password@localhost:5433/beacon_db"
    )

    redis_url: str = "redis://localhost:6379/0"

    local_storage_dir: str = "storage/local"

    demo_tenant_id: str = "demo"
    demo_user_email: str = "demo@beacon.local"

    tesseract_cmd: str | None = None

    anthropic_api_key: str | None = None

    llm_extractor_provider: str = "anthropic"

    claude_primary_model: str = "claude-sonnet-4-6"
    claude_fallback_model: str = "claude-opus-4-7"

    claude_max_tokens: int = 4096
    claude_temperature: float = 0.0
    claude_enable_fallback: bool = True

    llm_min_step_confidence: float = 0.70
    max_diagram_images_per_page: int = 8
    include_full_page_image_for_llm: bool = True

settings = Settings()

