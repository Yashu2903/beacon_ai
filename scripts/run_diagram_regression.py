from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import fitz


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "apps" / "api"
if str(API_PATH) not in sys.path:
    sys.path.insert(0, str(API_PATH))

from app.services.diagram_extraction import (
    OpenCVExtractor,
    PageExtractionInput,
    write_report_json,
)


ZOOM = 2.0
EXPECTED_SPEC_PATH = REPO_ROOT / "tests" / "fixtures" / "malm_expected.json"


def render_page_to_png(page: fitz.Page) -> tuple[bytes, int, int]:
    matrix = fitz.Matrix(ZOOM, ZOOM)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    return pixmap.tobytes("png"), pixmap.width, pixmap.height


def load_expected_spec() -> dict:
    return json.loads(EXPECTED_SPEC_PATH.read_text(encoding="utf-8"))


def build_page_inputs(pdf_path: str) -> list[PageExtractionInput]:
    page_inputs = []
    with fitz.open(pdf_path) as pdf_document:
        for page_index in range(pdf_document.page_count):
            page = pdf_document.load_page(page_index)
            page_png, _, _ = render_page_to_png(page)
            rect = page.rect
            page_inputs.append(
                PageExtractionInput(
                    page_number=page_index + 1,
                    page_png_bytes=page_png,
                    pdf_page_width=float(rect.width),
                    pdf_page_height=float(rect.height),
                )
            )
    return page_inputs


def page_area_from_input(page_input: PageExtractionInput) -> int:
    import cv2
    import numpy as np

    image_array = np.frombuffer(page_input.page_png_bytes, dtype=np.uint8)
    page_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if page_image is None:
        return 1
    return page_image.shape[0] * page_image.shape[1]


def run_regression(manual_path: str) -> tuple[bool, dict]:
    expected = load_expected_spec()
    page_inputs = build_page_inputs(manual_path)
    temp_root = Path(tempfile.mkdtemp(prefix="beacon_diagram_regression_"))
    storage_prefix = "regression/malm/diagrams"

    extractor = OpenCVExtractor(exempt_pages=expected["exempt_pages"])
    page_results = extractor.extract_document_regions(
        page_inputs=page_inputs,
        storage_prefix=storage_prefix,
    )

    max_region_area_fraction = float(expected["max_region_area_fraction"])
    min_region_counts = {
        int(page): count
        for page, count in expected["min_region_counts_by_page"].items()
    }
    expected_steps = {
        int(page): set(steps)
        for page, steps in expected["expected_step_numbers_by_page"].items()
    }
    page_area = {
        page_input.page_number: page_area_from_input(page_input)
        for page_input in page_inputs
    }

    pages = []
    failures = []
    for page_number in sorted(page_results):
        result = page_results[page_number]
        area_fractions = [
            (region.bbox_px["width"] * region.bbox_px["height"]) / page_area[page_number]
            for region in result.regions
        ]
        found_steps = sorted(
            {
                region.step_number
                for region in result.regions
                if region.step_number is not None
            }
        )
        expected_page_steps = sorted(expected_steps.get(page_number, set()))
        missing_steps = sorted(set(expected_page_steps) - set(found_steps))
        oversized_regions = [
            region.region_id
            for region, area_fraction in zip(result.regions, area_fractions)
            if area_fraction > max_region_area_fraction
        ]
        min_regions = min_region_counts.get(page_number)

        page_report = {
            "page_number": page_number,
            "region_count": len(result.regions),
            "average_region_area_fraction": (
                sum(area_fractions) / len(area_fractions) if area_fractions else 0.0
            ),
            "expected_step_numbers": expected_page_steps,
            "found_step_numbers": found_steps,
            "expected_step_numbers_found": not missing_steps,
            "missing_step_numbers": missing_steps,
            "oversized_region_ids": oversized_regions,
            "warnings": [
                {
                    "warning_type": warning.warning_type,
                    "page_number": warning.page_number,
                    "step_number": warning.step_number,
                    "message": warning.message,
                    "metadata": warning.metadata,
                }
                for warning in result.warnings
            ],
        }
        pages.append(page_report)

        if min_regions is not None and len(result.regions) < min_regions:
            failures.append(
                f"page {page_number}: expected at least {min_regions} regions, found {len(result.regions)}"
            )
        if missing_steps:
            failures.append(f"page {page_number}: missing expected steps {missing_steps}")
        if oversized_regions:
            failures.append(f"page {page_number}: oversized regions {oversized_regions}")

    for inventory_page in expected["parts_inventory_pages"]:
        result = page_results.get(inventory_page)
        if result is None:
            continue
        has_part_or_quantity = any(
            region.part_number or region.quantity for region in result.regions
        )
        if not has_part_or_quantity:
            failures.append(
                f"page {inventory_page}: part_inventory_incomplete, no quantity or part-number OCR association"
            )

    report = {
        "manual_id": expected["manual_id"],
        "manual_path": manual_path,
        "passed": not failures,
        "failures": failures,
        "pages": pages,
        "report_dir": str(temp_root),
    }

    report_path = temp_root / "diagram_regression_report.json"
    write_report_json(report_path, report)
    report["report_path"] = str(report_path)
    return not failures, report


def main() -> int:
    manual_path = os.environ.get("MALM_MANUAL_PATH")
    if not manual_path:
        print("SKIPPED: MALM_MANUAL_PATH is not set.")
        return 0

    if not Path(manual_path).exists():
        print(f"FAILED: MALM_MANUAL_PATH does not exist: {manual_path}")
        return 1

    passed, report = run_regression(manual_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
