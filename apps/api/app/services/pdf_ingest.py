from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

import cv2
import fitz
import numpy as np
from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_page import DocumentPage
from app.models.source_evidence import SourceEvidence
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


def _is_probable_diagram_region(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    page_width_px: int,
    page_height_px: int,
) -> bool:
    area = w * h
    page_area = page_width_px * page_height_px

    if area < page_area * 0.015:
        return False

    if area > page_area * 0.80:
        return False

    if w < 120 or h < 80:
        return False

    aspect_ratio = w / max(h, 1)

    if aspect_ratio < 0.25 or aspect_ratio > 6.0:
        return False

    return True


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.15,
) -> list[tuple[int, int, int, int]]:
    def iou(box_a, box_b) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh

        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    merged: list[tuple[int, int, int, int]] = []

    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        should_add = True

        for existing in merged:
            if iou(box, existing) > iou_threshold:
                should_add = False
                break

        if should_add:
            merged.append(box)

    return merged


def _detect_and_crop_diagram_regions(
    *,
    db: Session,
    document: Document,
    document_page: DocumentPage,
    page_number: int,
    page_png_bytes: bytes,
    pdf_page_width: float,
    pdf_page_height: float,
) -> int:
    image_array = np.frombuffer(page_png_bytes, dtype=np.uint8)
    page_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if page_image is None:
        return 0

    page_height_px, page_width_px = page_image.shape[:2]

    gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

    # Furniture manuals are usually black line drawings on white background.
    # We invert threshold so dark lines/text become white foreground.
    _, threshold = cv2.threshold(
        gray,
        245,
        255,
        cv2.THRESH_BINARY_INV,
    )

    # Morphological closing connects nearby lines into larger diagram regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidate_boxes: list[tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if _is_probable_diagram_region(
            x=x,
            y=y,
            w=w,
            h=h,
            page_width_px=page_width_px,
            page_height_px=page_height_px,
        ):
            candidate_boxes.append((x, y, w, h))

    boxes = _merge_overlapping_boxes(candidate_boxes)

    diagram_count = 0

    scale_x = pdf_page_width / page_width_px
    scale_y = pdf_page_height / page_height_px

    for diagram_index, (x, y, w, h) in enumerate(boxes):
        padding = 20

        crop_x1 = max(x - padding, 0)
        crop_y1 = max(y - padding, 0)
        crop_x2 = min(x + w + padding, page_width_px)
        crop_y2 = min(y + h + padding, page_height_px)

        crop = page_image[crop_y1:crop_y2, crop_x1:crop_x2]

        success, encoded = cv2.imencode(".png", crop)

        if not success:
            continue

        diagram_id = uuid.uuid4()

        storage_key = storage_service.build_generated_key(
            "documents",
            str(document.id),
            "diagrams",
            f"page_{page_number}_diagram_{diagram_index}_{diagram_id}.png",
        )

        storage_service.write_bytes(storage_key, encoded.tobytes())

        pdf_x = crop_x1 * scale_x
        pdf_y = crop_y1 * scale_y
        pdf_w = (crop_x2 - crop_x1) * scale_x
        pdf_h = (crop_y2 - crop_y1) * scale_y

        evidence = SourceEvidence(
            tenant_id=document.tenant_id,
            document_id=document.id,
            document_page_id=document_page.id,
            evidence_type="diagram_region",
            page_number=page_number,
            x=_decimal(pdf_x),
            y=_decimal(pdf_y),
            width=_decimal(pdf_w),
            height=_decimal(pdf_h),
            coordinate_system="pdf_points_top_left",
            storage_key=storage_key,
            confidence=Decimal("0.7000"),
            validation_flag="needs_review",
            metadata_json={
                "source": "opencv_contour_crop_from_rendered_page",
                "pixel_bbox": {
                    "x": crop_x1,
                    "y": crop_y1,
                    "width": crop_x2 - crop_x1,
                    "height": crop_y2 - crop_y1,
                },
                "rendered_page_width_px": page_width_px,
                "rendered_page_height_px": page_height_px,
            },
        )

        db.add(evidence)
        diagram_count += 1

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

            diagram_region_count += _detect_and_crop_diagram_regions(
                db=db,
                document=document,
                document_page=document_page,
                page_number=page_number,
                page_png_bytes=page_png,
                pdf_page_width=float(rect.width),
                pdf_page_height=float(rect.height),
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
