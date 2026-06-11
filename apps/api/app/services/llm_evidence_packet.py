from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document_page import DocumentPage
from app.models.source_evidence import SourceEvidence
from app.services.manual_structure_prompt_context import build_manual_structure_prompt_context

@dataclass
class EvidenceItem:
    id: str
    evidence_type: str
    page_number: int
    text: str | None = None
    storage_key: str | None = None
    local_path: str | None = None
    confidence: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PageEvidencePacket:
    document_id: str
    page_number: int
    full_page_image: EvidenceItem | None
    text_items: list[EvidenceItem]
    diagram_items: list[EvidenceItem]
    warnings: list[dict]

    def to_prompt_json(self) -> dict:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "full_page_image_evidence_id": (
                self.full_page_image.id if self.full_page_image else None
            ),
            "full_page_image_storage_key": (
                self.full_page_image.storage_key if self.full_page_image else None
            ),
            "text_evidence": [
                {
                    "id": item.id,
                    "text": item.text,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                }
                for item in self.text_items
            ],
            "diagram_evidence": [
                {
                    "id": item.id,
                    "evidence_type": item.evidence_type,
                    "storage_key": item.storage_key,
                    "confidence": item.confidence,
                    "metadata": item.metadata,
                }
                for item in self.diagram_items
            ],
            "warnings": self.warnings,
        }


def _storage_key_to_local_path(storage_key: str | None) -> str | None:
    if not storage_key:
        return None

    return str(Path(settings.local_storage_dir) / storage_key)


def _evidence_to_item(evidence: SourceEvidence) -> EvidenceItem:
    metadata = evidence.metadata_json or {}

    return EvidenceItem(
        id=str(evidence.id),
        evidence_type=evidence.evidence_type,
        page_number=evidence.page_number,
        text=evidence.extracted_text,
        storage_key=evidence.storage_key,
        local_path=_storage_key_to_local_path(evidence.storage_key),
        confidence=float(evidence.confidence) if evidence.confidence is not None else None,
        metadata=metadata,
    )


def _load_page_warnings(document_id: UUID, page_number: int) -> list[dict]:
    warnings_path = (
        Path(settings.local_storage_dir)
        / "documents"
        / str(document_id)
        / "diagrams"
        / "warnings.json"
    )

    if not warnings_path.exists():
        return []

    try:
        import json

        all_warnings = json.loads(warnings_path.read_text(encoding="utf-8"))
        return [
            warning
            for warning in all_warnings
            if warning.get("page_number") == page_number
        ]
    except Exception:
        return [
            {
                "warning_type": "warnings_file_parse_error",
                "message": f"Could not parse {warnings_path}",
            }
        ]


def _sort_diagram_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    def score(item: EvidenceItem) -> tuple:
        metadata = item.metadata or {}

        has_step = 1 if metadata.get("step_number") is not None else 0
        is_step_region = 1 if metadata.get("region_type") == "step" else 0
        confidence = item.confidence or 0.0

        bbox = metadata.get("bbox_px") or {}
        area = bbox.get("width", 0) * bbox.get("height", 0)

        return (
            has_step,
            is_step_region,
            confidence,
            area,
        )

    return sorted(items, key=score, reverse=True)


def build_page_evidence_packets(
    db: Session,
    document_id: UUID,
) -> list[PageEvidencePacket]:
    pages = (
        db.query(DocumentPage)
        .filter(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number.asc())
        .all()
    )

    packets: list[PageEvidencePacket] = []

    for page in pages:
        evidence_rows = (
            db.query(SourceEvidence)
            .filter(
                SourceEvidence.document_id == document_id,
                SourceEvidence.page_number == page.page_number,
            )
            .all()
        )

        text_items: list[EvidenceItem] = []
        diagram_items: list[EvidenceItem] = []

        for evidence in evidence_rows:
            item = _evidence_to_item(evidence)

            if evidence.evidence_type == "page_text_span":
                text_items.append(item)

            elif evidence.evidence_type == "diagram_region":
                diagram_items.append(item)

        diagram_items = _sort_diagram_items(diagram_items)[
            : settings.max_diagram_images_per_page
        ]

        full_page_image = EvidenceItem(
            id=str(page.id),
            evidence_type="full_page_image",
            page_number=page.page_number,
            storage_key=page.page_image_key,
            local_path=_storage_key_to_local_path(page.page_image_key),
            confidence=1.0,
            metadata={
                "source": "document_pages",
                "page_width": page.width,
                "page_height": page.height,
            },
        )

        packets.append(
            PageEvidencePacket(
                document_id=str(document_id),
                page_number=page.page_number,
                full_page_image=full_page_image,
                text_items=text_items,
                diagram_items=diagram_items,
                warnings=_load_page_warnings(document_id, page.page_number),
            )
        )

    return packets

def build_llm_evidence_packet(
    db: Session,
    document_id: UUID,
) -> dict:
    """
    Builds the full evidence packet for the LLM extractor.

    This combines:
    - page-level OCR / diagram / image evidence
    - manual structure context from ManualPageStructure

    The LLM should use manual_structure_context as the planning map before
    extracting steps.
    """
    page_packets = build_page_evidence_packets(
        db=db,
        document_id=document_id,
    )

    manual_structure_context = build_manual_structure_prompt_context(
        db=db,
        document_id=document_id,
    )

    return {
        "document_id": str(document_id),
        "manual_structure_context": manual_structure_context,
        "pages": [
            packet.to_prompt_json()
            for packet in page_packets
        ],
    }
