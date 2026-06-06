import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document
from app.models.document_page import DocumentPage
from app.models.source_evidence import SourceEvidence
from app.schemas.document import DocumentResponse
from app.schemas.evidence import DocumentPageResponse, SourceEvidenceResponse

router = APIRouter(prefix="/documents", tags=["evidence"])


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    return document


@router.get("/{document_id}/pages", response_model=list[DocumentPageResponse])
def list_document_pages(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = (
        db.query(DocumentPage)
        .filter(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number.asc())
        .all()
    )

    return pages


@router.get("/{document_id}/evidence", response_model=list[SourceEvidenceResponse])
def list_source_evidence(
    document_id: uuid.UUID,
    page_number: int | None = Query(default=None),
    evidence_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    query = db.query(SourceEvidence).filter(
        SourceEvidence.document_id == document_id
    )

    if page_number is not None:
        query = query.filter(SourceEvidence.page_number == page_number)

    if evidence_type is not None:
        query = query.filter(SourceEvidence.evidence_type == evidence_type)

    evidence = (
        query.order_by(
            SourceEvidence.page_number.asc(),
            SourceEvidence.created_at.asc(),
        )
        .limit(limit)
        .all()
    )

    return evidence