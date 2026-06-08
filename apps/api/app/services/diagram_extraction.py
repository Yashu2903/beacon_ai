from __future__ import annotations

import json
import logging
import math
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.core.config import settings
from app.services.storage import storage_service

logger = logging.getLogger(__name__)


BBox = dict[str, int | float]
COMMON_WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def resolve_tesseract_cmd() -> str | None:
    candidates: list[str] = []

    if settings.tesseract_cmd:
        candidates.append(settings.tesseract_cmd)

    candidates.extend(COMMON_WINDOWS_TESSERACT_PATHS)

    path_cmd = shutil.which("tesseract")
    if path_cmd:
        candidates.append(path_cmd)

    for candidate in candidates:
        try:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.is_file():
                return str(candidate_path)
        except (OSError, ValueError):
            continue

    return None


@dataclass
class DiagramRegion:
    region_id: str
    page_number: int
    bbox_px: BBox
    bbox_pdf: BBox
    region_type: str = "unknown"
    confidence: float = 0.7
    storage_key: str | None = None
    parent_region_id: str | None = None
    step_number: int | None = None
    part_number: str | None = None
    quantity: int | None = None
    nearby_text_tokens: list[str] = field(default_factory=list)
    warning_flag: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class OCRToken:
    text: str
    confidence: float
    bbox_px: BBox
    page_number: int


@dataclass
class ExtractionWarning:
    warning_type: str
    page_number: int | None
    step_number: int | None
    message: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DiagramExtractionResult:
    regions: list[DiagramRegion] = field(default_factory=list)
    warnings: list[ExtractionWarning] = field(default_factory=list)
    ocr_tokens: list[OCRToken] = field(default_factory=list)


@dataclass
class PageExtractionInput:
    page_number: int
    page_png_bytes: bytes
    pdf_page_width: float
    pdf_page_height: float


class DiagramExtractor(Protocol):
    def extract_page_regions(
        self,
        *,
        page_number: int,
        page_png_bytes: bytes,
        pdf_page_width: float,
        pdf_page_height: float,
        storage_prefix: str | None = None,
    ) -> DiagramExtractionResult:
        ...


class OpenCVExtractor:
    extractor_version = "opencv_extractor_v2_5"

    def __init__(
        self,
        *,
        threshold_value: int = 245,
        morph_kernel_size: int = 25,
        morph_iterations: int = 2,
        min_area_fraction: float = 0.015,
        max_area_fraction: float = 0.80,
        min_width_px: int = 120,
        min_height_px: int = 80,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 6.0,
        merge_iou_threshold: float = 0.15,
        max_merged_area_fraction: float = 0.65,
        desired_padding_px: int = 30,
        containment_threshold: float = 0.80,
        ocr_confidence_threshold: float = 70.0,
        step_min_height_px: int = 25,
        exempt_pages: list[int] | None = None,
    ):
        self.threshold_value = threshold_value
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.min_width_px = min_width_px
        self.min_height_px = min_height_px
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.merge_iou_threshold = merge_iou_threshold
        self.max_merged_area_fraction = max_merged_area_fraction
        self.desired_padding_px = desired_padding_px
        self.containment_threshold = containment_threshold
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.step_min_height_px = step_min_height_px
        self.exempt_pages = set(exempt_pages or [])

    def extract_page_regions(
        self,
        *,
        page_number: int,
        page_png_bytes: bytes,
        pdf_page_width: float,
        pdf_page_height: float,
        storage_prefix: str | None = None,
    ) -> DiagramExtractionResult:
        page_image = self._decode_page(page_png_bytes)
        if page_image is None:
            return DiagramExtractionResult(
                warnings=[
                    ExtractionWarning(
                        warning_type="page_decode_failed",
                        page_number=page_number,
                        step_number=None,
                        message="Rendered page PNG could not be decoded.",
                    )
                ]
            )

        page_height_px, page_width_px = page_image.shape[:2]
        boxes = self._detect_candidate_boxes(page_image)
        boxes = self._merge_overlapping_boxes(
            boxes,
            page_width_px=page_width_px,
            page_height_px=page_height_px,
        )

        ocr_tokens, ocr_warnings = self._extract_ocr_tokens(
            page_image=page_image,
            page_number=page_number,
        )

        regions = [
            self._region_from_box(
                box=box,
                page_number=page_number,
                page_width_px=page_width_px,
                page_height_px=page_height_px,
                pdf_page_width=pdf_page_width,
                pdf_page_height=pdf_page_height,
            )
            for box in boxes
        ]

        result = DiagramExtractionResult(
            regions=regions,
            warnings=ocr_warnings,
            ocr_tokens=ocr_tokens,
        )

        if storage_prefix is not None:
            return self.finalize_document_results(
                page_inputs=[
                    PageExtractionInput(
                        page_number=page_number,
                        page_png_bytes=page_png_bytes,
                        pdf_page_width=pdf_page_width,
                        pdf_page_height=pdf_page_height,
                    )
                ],
                page_results={page_number: result},
                storage_prefix=storage_prefix,
            )[page_number]

        return result

    def extract_document_regions(
        self,
        *,
        page_inputs: list[PageExtractionInput],
        storage_prefix: str,
    ) -> dict[int, DiagramExtractionResult]:
        page_results: dict[int, DiagramExtractionResult] = {}

        for page_input in page_inputs:
            page_results[page_input.page_number] = self.extract_page_regions(
                page_number=page_input.page_number,
                page_png_bytes=page_input.page_png_bytes,
                pdf_page_width=page_input.pdf_page_width,
                pdf_page_height=page_input.pdf_page_height,
            )

        return self.finalize_document_results(
            page_inputs=page_inputs,
            page_results=page_results,
            storage_prefix=storage_prefix,
        )

    def finalize_document_results(
        self,
        *,
        page_inputs: list[PageExtractionInput],
        page_results: dict[int, DiagramExtractionResult],
        storage_prefix: str,
    ) -> dict[int, DiagramExtractionResult]:
        page_images = {
            page_input.page_number: self._decode_page(page_input.page_png_bytes)
            for page_input in page_inputs
        }
        page_dims = {
            page_input.page_number: (
                page_images[page_input.page_number].shape[1],
                page_images[page_input.page_number].shape[0],
                page_input.pdf_page_width,
                page_input.pdf_page_height,
            )
            for page_input in page_inputs
            if page_images[page_input.page_number] is not None
        }

        validated_steps, rejected_steps = self._validate_step_tokens(page_results)
        self._assign_step_numbers(page_results, validated_steps)
        self._associate_part_tokens(page_results)
        self._apply_adjacency_padding(page_results, page_dims)
        self._assign_parent_child_regions(page_results)
        self._save_region_crops(page_results, page_images, page_dims, storage_prefix)
        self._append_completeness_warnings(
            page_inputs=page_inputs,
            page_results=page_results,
            validated_steps=validated_steps,
            rejected_steps=rejected_steps,
        )

        return page_results

    def write_warnings_json(
        self,
        *,
        page_results: dict[int, DiagramExtractionResult],
        storage_prefix: str,
    ) -> str:
        warnings = [
            self._warning_to_dict(warning)
            for page_number in sorted(page_results)
            for warning in page_results[page_number].warnings
        ]
        storage_key = f"{storage_prefix}/warnings.json"
        storage_service.write_bytes(
            storage_key,
            json.dumps(warnings, indent=2, sort_keys=True).encode("utf-8"),
        )
        for warning in warnings:
            logger.warning("diagram extraction warning: %s", warning)
        return storage_key

    def _decode_page(self, page_png_bytes: bytes):
        image_array = np.frombuffer(page_png_bytes, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    def _detect_candidate_boxes(self, page_image) -> list[tuple[int, int, int, int]]:
        page_height_px, page_width_px = page_image.shape[:2]
        gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(
            gray,
            self.threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        closed = cv2.morphologyEx(
            threshold,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.morph_iterations,
        )
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidate_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self._is_probable_diagram_region(
                x=x,
                y=y,
                w=w,
                h=h,
                page_width_px=page_width_px,
                page_height_px=page_height_px,
            ):
                candidate_boxes.append((x, y, w, h))
        return candidate_boxes

    def _is_probable_diagram_region(
        self,
        *,
        x: int,
        y: int,
        w: int,
        h: int,
        page_width_px: int,
        page_height_px: int,
    ) -> bool:
        del x, y
        area = w * h
        page_area = page_width_px * page_height_px
        aspect_ratio = w / max(h, 1)

        return (
            area >= page_area * self.min_area_fraction
            and area <= page_area * self.max_area_fraction
            and w >= self.min_width_px
            and h >= self.min_height_px
            and aspect_ratio >= self.min_aspect_ratio
            and aspect_ratio <= self.max_aspect_ratio
        )

    def _merge_overlapping_boxes(
        self,
        boxes: list[tuple[int, int, int, int]],
        *,
        page_width_px: int,
        page_height_px: int,
    ) -> list[tuple[int, int, int, int]]:
        merged: list[tuple[int, int, int, int]] = []
        page_area = page_width_px * page_height_px

        for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
            did_merge = False
            next_merged: list[tuple[int, int, int, int]] = []

            for existing in merged:
                if self._iou(box, existing) <= self.merge_iou_threshold:
                    next_merged.append(existing)
                    continue

                union_box = self._union_box(box, existing)
                merged_area_fraction = (union_box[2] * union_box[3]) / page_area
                if merged_area_fraction > self.max_merged_area_fraction:
                    next_merged.append(existing)
                    continue

                next_merged.append(union_box)
                did_merge = True

            if not did_merge:
                next_merged.append(box)

            merged = next_merged

        return merged

    def _region_from_box(
        self,
        *,
        box: tuple[int, int, int, int],
        page_number: int,
        page_width_px: int,
        page_height_px: int,
        pdf_page_width: float,
        pdf_page_height: float,
    ) -> DiagramRegion:
        x, y, w, h = box
        scale_x = pdf_page_width / page_width_px
        scale_y = pdf_page_height / page_height_px
        area_fraction = (w * h) / (page_width_px * page_height_px)

        return DiagramRegion(
            region_id=str(uuid.uuid4()),
            page_number=page_number,
            bbox_px={"x": x, "y": y, "width": w, "height": h},
            bbox_pdf={
                "x": x * scale_x,
                "y": y * scale_y,
                "width": w * scale_x,
                "height": h * scale_y,
            },
            region_type="unknown",
            confidence=0.7,
            metadata={
                "source": "opencv_contour_crop_from_rendered_page",
                "rendered_page_width_px": page_width_px,
                "rendered_page_height_px": page_height_px,
                "area_fraction": area_fraction,
                "extractor_version": self.extractor_version,
            },
        )

    def _extract_ocr_tokens(self, *, page_image, page_number: int):
        try:
            import pytesseract
            from pytesseract import Output
        except Exception as exc:
            return [], [self._ocr_unavailable_warning(page_number, exc)]

        resolved_cmd = resolve_tesseract_cmd()
        if resolved_cmd:
            pytesseract.pytesseract.tesseract_cmd = resolved_cmd

        try:
            data = pytesseract.image_to_data(page_image, output_type=Output.DICT)
        except pytesseract.TesseractNotFoundError as exc:
            return [], [self._ocr_unavailable_warning(page_number, exc)]
        except Exception as exc:
            return [], [self._ocr_unavailable_warning(page_number, exc)]

        tokens = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = (raw_text or "").strip()
            if not text:
                continue
            try:
                confidence = float(data["conf"][index])
            except (TypeError, ValueError):
                confidence = -1.0
            tokens.append(
                OCRToken(
                    text=text,
                    confidence=confidence,
                    page_number=page_number,
                    bbox_px={
                        "x": int(data["left"][index]),
                        "y": int(data["top"][index]),
                        "width": int(data["width"][index]),
                        "height": int(data["height"][index]),
                    },
                )
            )
        return tokens, []

    def _ocr_unavailable_warning(self, page_number: int, exc: Exception):
        return ExtractionWarning(
            warning_type="ocr_unavailable",
            page_number=page_number,
            step_number=None,
            message="OCR was unavailable; diagram extraction continued without OCR metadata.",
            metadata={
                "error": str(exc),
                "resolved_tesseract_cmd": resolve_tesseract_cmd(),
            },
        )

    def _validate_step_tokens(
        self,
        page_results: dict[int, DiagramExtractionResult],
    ) -> tuple[dict[int, list[OCRToken]], list[OCRToken]]:
        candidates = []
        for page_number in sorted(page_results):
            for token in page_results[page_number].ocr_tokens:
                if self._is_step_candidate(token):
                    candidates.append(token)

        candidates.sort(key=lambda token: (token.page_number, token.bbox_px["y"], token.bbox_px["x"]))
        expected_next: int | None = None
        validated: dict[int, list[OCRToken]] = {}
        rejected: list[OCRToken] = []

        for token in candidates:
            step_number = int(token.text)
            if expected_next is None:
                expected_next = step_number
                validated.setdefault(token.page_number, []).append(token)
                continue

            if step_number == expected_next or step_number == expected_next + 1:
                if step_number == expected_next + 1:
                    expected_next = step_number
                validated.setdefault(token.page_number, []).append(token)
            else:
                rejected.append(token)

        return validated, rejected

    def _is_step_candidate(self, token: OCRToken) -> bool:
        text = token.text.strip()
        height = int(token.bbox_px["height"])
        return (
            text.isdigit()
            and 1 <= len(text) <= 2
            and token.confidence >= self.ocr_confidence_threshold
            and height >= self.step_min_height_px
        )

    def _assign_step_numbers(
        self,
        page_results: dict[int, DiagramExtractionResult],
        validated_steps: dict[int, list[OCRToken]],
    ) -> None:
        for page_number, result in page_results.items():
            tokens = sorted(validated_steps.get(page_number, []), key=lambda t: t.bbox_px["y"])
            if not tokens:
                continue

            page_regions = result.regions
            for region in page_regions:
                center_y = self._center(region.bbox_px)[1]
                assigned = None
                for index, token in enumerate(tokens):
                    start_y = token.bbox_px["y"]
                    end_y = tokens[index + 1].bbox_px["y"] if index + 1 < len(tokens) else math.inf
                    if center_y >= start_y and center_y < end_y:
                        assigned = int(token.text)
                        break
                if assigned is None and center_y < tokens[0].bbox_px["y"]:
                    assigned = int(tokens[0].text)
                region.step_number = assigned

    def _associate_part_tokens(self, page_results: dict[int, DiagramExtractionResult]) -> None:
        quantity_pattern = re.compile(r"^(\d{1,2})[xX]$")
        part_pattern = re.compile(r"^\d{5,8}$")

        for result in page_results.values():
            for token in result.ocr_tokens:
                quantity_match = quantity_pattern.match(token.text)
                is_part = bool(part_pattern.match(token.text))
                if not quantity_match and not is_part:
                    continue

                region = self._nearest_region(token, result.regions)
                if region is None:
                    continue

                old_bbox = dict(region.bbox_px)
                region.bbox_px = self._bbox_dict(self._union_box(self._tuple_box(region.bbox_px), self._tuple_box(token.bbox_px)))
                region.nearby_text_tokens.append(token.text)
                associations = region.metadata.setdefault("ocr_associations", [])
                association = {
                    "token": token.text,
                    "confidence": token.confidence,
                    "bbox_px": token.bbox_px,
                    "region_bbox_before_union": old_bbox,
                }
                if quantity_match:
                    region.quantity = int(quantity_match.group(1))
                    association["association_type"] = "quantity"
                if is_part:
                    region.part_number = token.text
                    association["association_type"] = "part_number"
                associations.append(association)

    def _apply_adjacency_padding(
        self,
        page_results: dict[int, DiagramExtractionResult],
        page_dims: dict[int, tuple[int, int, float, float]],
    ) -> None:
        for page_number, result in page_results.items():
            if page_number not in page_dims:
                continue
            page_width_px, page_height_px, pdf_width, pdf_height = page_dims[page_number]
            scale_x = pdf_width / page_width_px
            scale_y = pdf_height / page_height_px

            for region in result.regions:
                x, y, w, h = self._tuple_box(region.bbox_px)
                right_limit = left_limit = top_limit = bottom_limit = self.desired_padding_px

                for neighbor in result.regions:
                    if neighbor.region_id == region.region_id:
                        continue
                    nx, ny, nw, nh = self._tuple_box(neighbor.bbox_px)
                    if self._ranges_overlap(y, y + h, ny, ny + nh):
                        if nx >= x + w:
                            right_limit = min(right_limit, max(0, (nx - (x + w)) // 2))
                        if x >= nx + nw:
                            left_limit = min(left_limit, max(0, (x - (nx + nw)) // 2))
                    if self._ranges_overlap(x, x + w, nx, nx + nw):
                        if ny >= y + h:
                            bottom_limit = min(bottom_limit, max(0, (ny - (y + h)) // 2))
                        if y >= ny + nh:
                            top_limit = min(top_limit, max(0, (y - (ny + nh)) // 2))

                padded_x1 = max(0, x - left_limit)
                padded_y1 = max(0, y - top_limit)
                padded_x2 = min(page_width_px, x + w + right_limit)
                padded_y2 = min(page_height_px, y + h + bottom_limit)

                padding = {
                    "left": x - padded_x1,
                    "top": y - padded_y1,
                    "right": padded_x2 - (x + w),
                    "bottom": padded_y2 - (y + h),
                }
                region.metadata["padding"] = padding
                region.metadata["unpadded_bbox_px"] = {"x": x, "y": y, "width": w, "height": h}
                region.bbox_px = {
                    "x": padded_x1,
                    "y": padded_y1,
                    "width": padded_x2 - padded_x1,
                    "height": padded_y2 - padded_y1,
                }
                region.bbox_pdf = {
                    "x": padded_x1 * scale_x,
                    "y": padded_y1 * scale_y,
                    "width": (padded_x2 - padded_x1) * scale_x,
                    "height": (padded_y2 - padded_y1) * scale_y,
                }

    def _assign_parent_child_regions(self, page_results: dict[int, DiagramExtractionResult]) -> None:
        for result in page_results.values():
            regions = result.regions
            for child in regions:
                child_area = self._area(child.bbox_px)
                for parent in regions:
                    if child.region_id == parent.region_id:
                        continue
                    parent_area = self._area(parent.bbox_px)
                    if parent_area <= child_area:
                        continue
                    if self._contained_fraction(child.bbox_px, parent.bbox_px) >= self.containment_threshold:
                        child.parent_region_id = parent.region_id
                        child.region_type = "sub_diagram"
                        break

            by_step: dict[int, list[DiagramRegion]] = {}
            for region in regions:
                if region.step_number is not None:
                    by_step.setdefault(region.step_number, []).append(region)

            for step_regions in by_step.values():
                primary = max(step_regions, key=lambda r: self._area(r.bbox_px), default=None)
                if primary is None:
                    continue
                primary_area = self._area(primary.bbox_px)
                for region in step_regions:
                    if region.region_id == primary.region_id:
                        continue
                    if self._area(region.bbox_px) < primary_area * 0.5:
                        region.parent_region_id = primary.region_id
                        region.region_type = "sub_diagram"

            for region in regions:
                if region.region_type == "unknown":
                    region.region_type = "step" if region.step_number is not None else "unknown"

    def _save_region_crops(
        self,
        page_results: dict[int, DiagramExtractionResult],
        page_images: dict[int, object],
        page_dims: dict[int, tuple[int, int, float, float]],
        storage_prefix: str,
    ) -> None:
        for page_number, result in page_results.items():
            page_image = page_images.get(page_number)
            if page_image is None or page_number not in page_dims:
                continue
            page_width_px, page_height_px, _, _ = page_dims[page_number]

            for diagram_index, region in enumerate(result.regions):
                x, y, w, h = self._tuple_box(region.bbox_px)
                crop_x1 = max(0, x)
                crop_y1 = max(0, y)
                crop_x2 = min(page_width_px, x + w)
                crop_y2 = min(page_height_px, y + h)
                crop = page_image[crop_y1:crop_y2, crop_x1:crop_x2]
                success, encoded = cv2.imencode(".png", crop)
                if not success:
                    result.warnings.append(
                        ExtractionWarning(
                            warning_type="crop_encode_failed",
                            page_number=page_number,
                            step_number=region.step_number,
                            message="Diagram crop could not be encoded.",
                            metadata={"region_id": region.region_id},
                        )
                    )
                    continue

                storage_key = (
                    f"{storage_prefix}/"
                    f"page_{page_number}_diagram_{diagram_index}_{region.region_id}.png"
                )
                storage_service.write_bytes(storage_key, encoded.tobytes())
                region.storage_key = storage_key

    def _append_completeness_warnings(
        self,
        *,
        page_inputs: list[PageExtractionInput],
        page_results: dict[int, DiagramExtractionResult],
        validated_steps: dict[int, list[OCRToken]],
        rejected_steps: list[OCRToken],
    ) -> None:
        for token in rejected_steps:
            page_results[token.page_number].warnings.append(
                ExtractionWarning(
                    warning_type="rejected_step_candidate",
                    page_number=token.page_number,
                    step_number=int(token.text) if token.text.isdigit() else None,
                    message="OCR step candidate rejected by monotonic sequence validation.",
                    metadata={"token": self._ocr_token_to_dict(token)},
                )
            )

        all_valid_step_numbers = sorted(
            {int(token.text) for tokens in validated_steps.values() for token in tokens}
        )
        all_regions = [
            region
            for result in page_results.values()
            for region in result.regions
        ]

        for step_number in all_valid_step_numbers:
            step_regions = [region for region in all_regions if region.step_number == step_number]
            if not step_regions:
                self._add_global_warning(
                    page_results,
                    ExtractionWarning(
                        warning_type="missing_step_region",
                        page_number=None,
                        step_number=step_number,
                        message="Validated step number has no associated diagram region.",
                    ),
                )
            if step_regions and not any(region.parent_region_id is None for region in step_regions):
                self._add_global_warning(
                    page_results,
                    ExtractionWarning(
                        warning_type="missing_primary_step_region",
                        page_number=None,
                        step_number=step_number,
                        message="Validated step number has only sub-diagram regions.",
                    ),
                )

        for page_input in page_inputs:
            if page_input.page_number in self.exempt_pages:
                continue
            result = page_results[page_input.page_number]
            if not result.regions:
                result.warnings.append(
                    ExtractionWarning(
                        warning_type="missing_page_regions",
                        page_number=page_input.page_number,
                        step_number=None,
                        message="No diagram regions detected for non-exempt page.",
                    )
                )

            page_area = 1
            page_image = self._decode_page(page_input.page_png_bytes)
            if page_image is not None:
                page_area = page_image.shape[0] * page_image.shape[1]
            for region in result.regions:
                area_fraction = self._area(region.bbox_px) / page_area
                if area_fraction > self.max_merged_area_fraction:
                    region.warning_flag = True
                    result.warnings.append(
                        ExtractionWarning(
                            warning_type="giant_region",
                            page_number=page_input.page_number,
                            step_number=region.step_number,
                            message="Detected diagram region exceeds maximum merged area fraction.",
                            metadata={
                                "region_id": region.region_id,
                                "area_fraction": area_fraction,
                                "max_merged_area_fraction": self.max_merged_area_fraction,
                            },
                        )
                    )

    def _add_global_warning(
        self,
        page_results: dict[int, DiagramExtractionResult],
        warning: ExtractionWarning,
    ) -> None:
        if not page_results:
            return
        first_page = sorted(page_results)[0]
        page_results[first_page].warnings.append(warning)

    def _nearest_region(
        self,
        token: OCRToken,
        regions: list[DiagramRegion],
    ) -> DiagramRegion | None:
        if not regions:
            return None
        token_center = self._center(token.bbox_px)
        return min(
            regions,
            key=lambda region: self._distance(token_center, self._center(region.bbox_px)),
        )

    def _iou(self, box_a, box_b) -> float:
        inter_area = self._intersection_area(self._bbox_dict(box_a), self._bbox_dict(box_b))
        union_area = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def _contained_fraction(self, child: BBox, parent: BBox) -> float:
        child_area = self._area(child)
        if child_area == 0:
            return 0.0
        return self._intersection_area(child, parent) / child_area

    def _intersection_area(self, box_a: BBox, box_b: BBox) -> int:
        ax, ay, aw, ah = self._tuple_box(box_a)
        bx, by, bw, bh = self._tuple_box(box_b)
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax + aw, bx + bw)
        inter_y2 = min(ay + ah, by + bh)
        return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    def _union_box(self, box_a, box_b) -> tuple[int, int, int, int]:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        x1 = min(ax, bx)
        y1 = min(ay, by)
        x2 = max(ax + aw, bx + bw)
        y2 = max(ay + ah, by + bh)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)

    def _bbox_dict(self, box) -> BBox:
        x, y, w, h = box
        return {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}

    def _tuple_box(self, box: BBox) -> tuple[int, int, int, int]:
        return (
            int(box["x"]),
            int(box["y"]),
            int(box["width"]),
            int(box["height"]),
        )

    def _center(self, box: BBox) -> tuple[float, float]:
        x, y, w, h = self._tuple_box(box)
        return (x + w / 2, y + h / 2)

    def _distance(self, point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def _area(self, box: BBox) -> int:
        return int(box["width"]) * int(box["height"])

    def _ranges_overlap(self, a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return max(a_start, b_start) < min(a_end, b_end)

    def _ocr_token_to_dict(self, token: OCRToken) -> dict:
        return {
            "text": token.text,
            "confidence": token.confidence,
            "bbox_px": token.bbox_px,
            "page_number": token.page_number,
        }

    def _warning_to_dict(self, warning: ExtractionWarning) -> dict:
        return {
            "warning_type": warning.warning_type,
            "page_number": warning.page_number,
            "step_number": warning.step_number,
            "message": warning.message,
            "metadata": warning.metadata,
        }


def write_report_json(path: str | Path, report: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
