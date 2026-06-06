from fastapi import FastAPI

from app.api.routes import evidence, jobs, upload
from app.core.database import Base, engine
from app.models import AuditLog, Document, DocumentPage, Job, SourceEvidence, User


app = FastAPI(title="Beacon AI API")


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(evidence.router)