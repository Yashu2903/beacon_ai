from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import audit_logs, evidence, jobs, upload
from app.core.config import settings


app = FastAPI(title="Beacon AI API")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Local development only.
# In production, use signed URLs from object storage.
app.mount(
    "/storage",
    StaticFiles(directory=settings.local_storage_dir),
    name="storage",
)


app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(evidence.router)
app.include_router(audit_logs.router)
