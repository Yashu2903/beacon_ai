import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.audit_log import AuditLog
from app.models.job import Job
from app.schemas.audit_log import AuditLogResponse

router = APIRouter(prefix="/jobs", tags=["audit-logs"])


@router.get("/{job_id}/audit-logs", response_model=list[AuditLogResponse])
def list_job_audit_logs(
    job_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    job = db.get(Job, job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    logs = (
        db.query(AuditLog)
        .filter(AuditLog.job_id == job_id)
        .order_by(AuditLog.created_at.asc())
        .limit(limit)
        .all()
    )

    return logs