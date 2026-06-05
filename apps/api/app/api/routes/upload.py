from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.document import Document
from app.models.enums import JobState
from app.models.job import Job
from app.models.user import User
from app.schemas.upload import UploadResponse
from app.services.queue import celery_app
from app.services.storage import storage_service

router = APIRouter(prefix="/uploads", tags=["uploads"])


def get_or_create_demo_user(db: Session) -> User:
    user = db.query(User).filter(User.email == settings.demo_user_email).first()

    if user:
        return user

    user = User(
        tenant_id=settings.demo_tenant_id,
        email=settings.demo_user_email,
        display_name="Demo User",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user


@router.post("/manual", response_model=UploadResponse)
async def upload_manual(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF furniture manuals are supported in V0.",
        )

    user = get_or_create_demo_user(db)

    storage_key = storage_service.build_storage_key(file.filename or "manual.pdf")
    size_bytes = await storage_service.save_upload(file, storage_key)

    document = Document(
        tenant_id=settings.demo_tenant_id,
        user_id=user.id,
        original_filename=file.filename or "manual.pdf",
        storage_key=storage_key,
        content_type=file.content_type,
        size_bytes=size_bytes,
        suitability_status="pending",
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    job = Job(
        tenant_id=settings.demo_tenant_id,
        user_id=user.id,
        document_id=document.id,
        state=JobState.CREATED,
    )

    db.add(job)
    db.commit()
    db.refresh(job)

    celery_app.send_task(
        "worker.tasks.ingest_manual",
        args=[str(job.id)],
    )

    return UploadResponse(
        document_id=document.id,
        job_id=job.id,
        state=job.state,
    )