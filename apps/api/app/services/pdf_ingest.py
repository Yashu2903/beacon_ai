from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

import fitz
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_page import DocumentPage
from app.models.source_evidence import SourceEvidence
from app.services.diagram_extraction import (
    DiagramExtractionResult,
    DiagramRegion,
    OpenCVExtractor,
    PageExtractionInput,
)
from app.services.storage import storage_service


@dataclass
class IngestResult:
    page_count: int
    text_span_count: int
    embedded_image_count: int
    diagram_region_count: int
    scanned_flag: bool


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(value, 2)))


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _render_page_to_png(page: fitz.Page, zoom: float = 2.0) -> tuple[bytes, int, int]:
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pixmap.tobytes("png")
    return png_bytes, pixmap.width, pixmap.height


def _create_text_span_evidence(
    *,
    db: Session,
    document: Document,
    document_page: DocumentPage,
    page_number: int,
    block_index: int,
    text: str,
    bbox: tuple[float, float, float, float],
    running_text_offset: int,
) -> int:
    cleaned_text = _normalize_text(text)

    if not cleaned_text:
        return 0

    x0, y0, x1, y1 = bbox

    evidence = SourceEvidence(
        tenant_id=document.tenant_id,
        document_id=document.id,
        document_page_id=document_page.id,
        evidence_type="page_text_span",
        page_number=page_number,
        text_start=running_text_offset,
        text_end=running_text_offset + len(cleaned_text),
        extracted_text=cleaned_text,
        x=_decimal(x0),
        y=_decimal(y0),
        width=_decimal(x1 - x0),
        height=_decimal(y1 - y0),
        coordinate_system="pdf_points_top_left",
        confidence=Decimal("0.9500"),
        validation_flag=None,
        metadata_json={
            "block_index": block_index,
            "source": "pymupdf_get_text_blocks",
        },
    )

    db.add(evidence)

    return len(cleaned_text)


def _extract_text_blocks(
    *,
    db: Session,
    document: Document,
    document_page: DocumentPage,
    page: fitz.Page,
    page_number: int,
) -> int:
    blocks = page.get_text("blocks")

    span_count = 0
    running_text_offset = 0

    for block_index, block in enumerate(blocks):
        x0, y0, x1, y1, text, *_ = block

        added_chars = _create_text_span_evidence(
            db=db,
            document=document,
            document_page=document_page,
            page_number=page_number,
            block_index=block_index,
            text=text,
            bbox=(x0, y0, x1, y1),
            running_text_offset=running_text_offset,
        )

        if added_chars > 0:
            span_count += 1
            running_text_offset += added_chars + 1

    document_page.text_extraction_status = (
        "extracted" if span_count > 0 else "no_text_found"
    )

    return span_count


def _extract_embedded_images(
    *,
    db: Session,
    document: Document,
    document_page: DocumentPage,
    pdf_document: fitz.Document,
    page: fitz.Page,
    page_number: int,
) -> int:
    embedded_image_count = 0

    for image_index, image_info in enumerate(page.get_images(full=True)):
        xref = image_info[0]

        try:
            extracted = pdf_document.extract_image(xref)
        except Exception:
            continue

        image_bytes = extracted.get("image")
        image_ext = extracted.get("ext", "png")

        if not image_bytes:
            continue

        image_id = uuid.uuid4()

        storage_key = storage_service.build_generated_key(
            "documents",
            str(document.id),
            "images",
            f"page_{page_number}_embedded_{image_index}_{image_id}.{image_ext}",
        )

        storage_service.write_bytes(storage_key, image_bytes)

        rects = page.get_image_rects(xref)

        if rects:
            rect = rects[0]
            x = _decimal(rect.x0)
            y = _decimal(rect.y0)
            width = _decimal(rect.width)
            height = _decimal(rect.height)
        else:
            x = y = width = height = None

        evidence = SourceEvidence(
            tenant_id=document.tenant_id,
            document_id=document.id,
            document_page_id=document_page.id,
            evidence_type="embedded_image",
            page_number=page_number,
            x=x,
            y=y,
            width=width,
            height=height,
            coordinate_system="pdf_points_top_left",
            storage_key=storage_key,
            confidence=Decimal("0.8000"),
            validation_flag=None,
            metadata_json={
                "image_index": image_index,
                "xref": xref,
                "image_ext": image_ext,
                "source": "pymupdf_extract_image",
            },
        )

        db.add(evidence)
        embedded_image_count += 1

    return embedded_image_count


def _create_diagram_region_evidence(
    *,
    db: Session,
    document: Document,
    document_page: DocumentPage,
    region: DiagramRegion,
) -> int:
    if region.storage_key is None:
        return 0

    metadata = dict(region.metadata)
    metadata.update(
        {
            "region_id": region.region_id,
            "region_type": region.region_type,
            "parent_region_id": region.parent_region_id,
            "step_number": region.step_number,
            "part_number": region.part_number,
            "quantity": region.quantity,
            "nearby_text_tokens": region.nearby_text_tokens,
            "warning_flag": region.warning_flag,
            "bbox_px": region.bbox_px,
            "pixel_bbox": region.bbox_px,
            "padding": region.metadata.get("padding"),
            "extractor_version": metadata.get(
                "extractor_version",
                OpenCVExtractor.extractor_version,
            ),
        }
    )

    evidence = SourceEvidence(
        tenant_id=document.tenant_id,
        document_id=document.id,
        document_page_id=document_page.id,
        evidence_type="diagram_region",
        page_number=region.page_number,
        x=_decimal(float(region.bbox_pdf["x"])),
        y=_decimal(float(region.bbox_pdf["y"])),
        width=_decimal(float(region.bbox_pdf["width"])),
        height=_decimal(float(region.bbox_pdf["height"])),
        coordinate_system="pdf_points_top_left",
        storage_key=region.storage_key,
        confidence=Decimal(str(round(region.confidence, 4))),
        validation_flag="needs_review",
        metadata_json=metadata,
    )

    db.add(evidence)
    return 1


def _persist_diagram_regions(
    *,
    db: Session,
    document: Document,
    pages_by_number: dict[int, DocumentPage],
    page_results: dict[int, DiagramExtractionResult],
) -> int:
    diagram_count = 0

    for page_number in sorted(page_results):
        document_page = pages_by_number[page_number]
        for region in page_results[page_number].regions:
            diagram_count += _create_diagram_region_evidence(
                db=db,
                document=document,
                document_page=document_page,
                region=region,
            )

    return diagram_count


def ingest_pdf_document(db: Session, document_id: uuid.UUID) -> IngestResult:
    document = db.get(Document, document_id)

    if document is None:
        raise ValueError(f"Document not found: {document_id}")

    pdf_path = storage_service.get_local_path(document.storage_key)

    text_span_count = 0
    embedded_image_count = 0
    diagram_region_count = 0
    pages_without_text = 0
    page_inputs: list[PageExtractionInput] = []
    pages_by_number: dict[int, DocumentPage] = {}
    diagram_extractor = OpenCVExtractor()

    with fitz.open(pdf_path) as pdf_document:
        document.page_count = pdf_document.page_count

        for page_index in range(pdf_document.page_count):
            page = pdf_document.load_page(page_index)
            page_number = page_index + 1
            rect = page.rect

            page_image_key = storage_service.build_generated_key(
                "documents",
                str(document.id),
                "pages",
                f"page_{page_number}.png",
            )

            page_png, page_width_px, page_height_px = _render_page_to_png(page)
            storage_service.write_bytes(page_image_key, page_png)

            document_page = DocumentPage(
                tenant_id=document.tenant_id,
                document_id=document.id,
                page_number=page_number,
                width=_decimal(rect.width),
                height=_decimal(rect.height),
                page_image_key=page_image_key,
                text_extraction_status="pending",
            )

            db.add(document_page)
            db.flush()
            pages_by_number[page_number] = document_page
            page_inputs.append(
                PageExtractionInput(
                    page_number=page_number,
                    page_png_bytes=page_png,
                    pdf_page_width=float(rect.width),
                    pdf_page_height=float(rect.height),
                )
            )

            page_span_count = _extract_text_blocks(
                db=db,
                document=document,
                document_page=document_page,
                page=page,
                page_number=page_number,
            )

            if page_span_count == 0:
                pages_without_text += 1

            text_span_count += page_span_count

            embedded_image_count += _extract_embedded_images(
                db=db,
                document=document,
                document_page=document_page,
                pdf_document=pdf_document,
                page=page,
                page_number=page_number,
            )

        diagram_storage_prefix = storage_service.build_generated_key(
            "documents",
            str(document.id),
            "diagrams",
        )
        diagram_results = diagram_extractor.extract_document_regions(
            page_inputs=page_inputs,
            storage_prefix=diagram_storage_prefix,
        )
        diagram_extractor.write_warnings_json(
            page_results=diagram_results,
            storage_prefix=diagram_storage_prefix,
        )
        diagram_region_count = _persist_diagram_regions(
            db=db,
            document=document,
            pages_by_number=pages_by_number,
            page_results=diagram_results,
        )

        document.scanned_flag = pages_without_text > 0
        document.detected_language = "en"
        document.suitability_status = "needs_review"

    db.commit()

    return IngestResult(
        page_count=document.page_count or 0,
        text_span_count=text_span_count,
        embedded_image_count=embedded_image_count,
        diagram_region_count=diagram_region_count,
        scanned_flag=document.scanned_flag,
    )
