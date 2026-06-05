from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Beacon AI API"

    database_url: str = (
        "postgresql+psycopg://beacon:beacon_password@localhost:5433/beacon_db"
    )

    redis_url: str = "redis://localhost:6379/0"

    local_storage_dir: str = "storage/local"

    demo_tenant_id: str = "demo"
    demo_user_email: str = "demo@beacon.local"

    class Config:
        env_file = ".env"


settings = Settings()

