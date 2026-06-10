import re
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytesseract
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document_page import DocumentPage
from app.models.manual_page_structure import ManualPageStructure, ManualPageType
from app.models.source_evidence import SourceEvidence


@dataclass
class OCRToken:
    text: str
    conf: float
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2


def _storage_root() -> Path:
    return Path(settings.local_storage_dir)


def _load_page_image(page: DocumentPage) -> Image.Image | None:
    if not page.page_image_key:
        return None

    image_path = _storage_root() / page.page_image_key

    if not image_path.exists():
        return None

    return Image.open(image_path).convert("RGB")


def _ocr_page(image: Image.Image) -> list[OCRToken]:
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--psm 6",
    )

    tokens: list[OCRToken] = []

    for i, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()

        if not text:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < 30:
            continue

        tokens.append(
            OCRToken(
                text=text,
                conf=conf,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            )
        )

    return tokens


def _is_integer_token(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{1,2}", text.strip()))


def _is_quantity_token(text: str) -> bool:
    normalized = (
        text.strip()
        .lower()
        .replace("×", "x")
        .replace(" ", "")
    )

    return bool(re.fullmatch(r"[0-9]{1,2}x", normalized))


def _is_part_number_token(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{5,8}", text.strip()))


def _is_footer_token(token: OCRToken, image_height: int) -> bool:
    return token.center_y > image_height * 0.92


def _detect_inventory_band_bottom(
    tokens: list[OCRToken],
    image_height: int,
) -> int | None:
    quantity_tokens = [
        token
        for token in tokens
        if _is_quantity_token(token.text) and token.center_y < image_height * 0.75
    ]

    if len(quantity_tokens) < 4:
        return None

    lowest_quantity_y = max(token.bottom for token in quantity_tokens)

    return min(
        int(lowest_quantity_y + image_height * 0.08),
        int(image_height * 0.75),
    )


def _page_has_inventory_signals(tokens: list[OCRToken]) -> bool:
    quantity_count = 0
    part_number_count = 0

    for token in tokens:
        text = token.text.strip()

        if _is_quantity_token(text):
            quantity_count += 1

        if _is_part_number_token(text):
            part_number_count += 1

    if quantity_count >= 5:
        return True

    if quantity_count >= 3 and part_number_count >= 2:
        return True

    return False


def _page_has_assembly_visual_signals(
    db: Session,
    document_id: UUID,
    page_number: int,
) -> bool:
    count = (
        db.query(SourceEvidence)
        .filter(SourceEvidence.document_id == document_id)
        .filter(SourceEvidence.page_number == page_number)
        .filter(SourceEvidence.evidence_type == "diagram_region")
        .count()
    )

    return count > 0


def _score_step_number_candidate(
    token: OCRToken,
    image_width: int,
    image_height: int,
    inventory_band_bottom: int | None,
) -> float:
    text = token.text.strip()

    if not _is_integer_token(text):
        return 0.0

    if _is_footer_token(token, image_height):
        return 0.0

    if token.height < 18:
        return 0.0

    if inventory_band_bottom is not None and token.center_y < inventory_band_bottom:
        return 0.0

    number = int(text)

    if number < 1 or number > 50:
        return 0.0

    score = 0.0

    if token.left < image_width * 0.16:
        score += 0.45
    elif token.left < image_width * 0.25:
        score += 0.25

    if token.height >= 40:
        score += 0.35
    elif token.height >= 28:
        score += 0.20

    if token.conf >= 80:
        score += 0.20
    elif token.conf >= 55:
        score += 0.10

    if token.width < 18 and token.height < 24:
        score -= 0.35

    return score


def _detect_step_numbers_from_ocr(
    tokens: list[OCRToken],
    image_width: int,
    image_height: int,
) -> list[int]:
    inventory_band_bottom = _detect_inventory_band_bottom(
        tokens=tokens,
        image_height=image_height,
    )

    candidates: list[tuple[int, float, OCRToken]] = []

    for token in tokens:
        score = _score_step_number_candidate(
            token=token,
            image_width=image_width,
            image_height=image_height,
            inventory_band_bottom=inventory_band_bottom,
        )

        if score < 0.55:
            continue

        candidates.append((int(token.text.strip()), score, token))

    return sorted({number for number, _score, _token in candidates})


def _classify_page(
    *,
    page_number: int,
    total_pages: int,
    visible_step_numbers: list[int],
    has_inventory_signals: bool,
    has_assembly_visual_signals: bool,
    tokens: list[OCRToken],
) -> tuple[ManualPageType, float, dict]:
    token_texts = [token.text for token in tokens]
    joined = " ".join(token_texts).lower()

    metadata: dict = {
        "has_inventory_signals": has_inventory_signals,
        "has_assembly_visual_signals": has_assembly_visual_signals,
        "ocr_token_count": len(tokens),
    }

    if page_number == 1:
        return ManualPageType.cover, 0.85, metadata

    if page_number == total_pages or "inter ikea" in joined or ("systems" in joined and "202" in joined):
        return ManualPageType.back_matter, 0.85, metadata

    if has_inventory_signals and visible_step_numbers:
        return ManualPageType.mixed_inventory_and_step, 0.80, metadata

    if has_inventory_signals and not visible_step_numbers:
        return ManualPageType.parts_inventory, 0.80, metadata

    if visible_step_numbers:
        return ManualPageType.assembly_step, 0.80, metadata

    if has_assembly_visual_signals:
        return ManualPageType.informational, 0.60, metadata

    return ManualPageType.unknown, 0.40, metadata


def _is_repairable_page(structure: ManualPageStructure) -> bool:
    if structure.page_type in {
        ManualPageType.cover,
        ManualPageType.back_matter,
    }:
        return False

    if structure.page_type == ManualPageType.parts_inventory:
        return False

    metadata = structure.metadata_json or {}

    if metadata.get("has_assembly_visual_signals") is True:
        return True

    if structure.page_type in {
        ManualPageType.assembly_step,
        ManualPageType.mixed_inventory_and_step,
        ManualPageType.informational,
        ManualPageType.unknown,
    }:
        return True

    return False


def _add_repair_metadata(
    structure: ManualPageStructure,
    *,
    repair_type: str,
    notes: list[str],
    confidence: float = 0.65,
) -> None:
    metadata = structure.metadata_json or {}

    existing_types = metadata.get("repair_types", [])
    if repair_type not in existing_types:
        existing_types.append(repair_type)

    existing_notes = metadata.get("repair_notes", [])
    existing_notes.extend(notes)

    metadata["repair_applied"] = True
    metadata["repair_types"] = existing_types
    metadata["repair_notes"] = existing_notes

    structure.metadata_json = metadata
    structure.confidence = min(float(structure.confidence or 0.0), confidence)


def _soft_repair_same_page_pairs(
    structures: list[ManualPageStructure],
) -> None:
    """
    Repair simple cases where OCR detects only one step on a page that probably has two.

    Example:
    page 5: [4], next page: [6] -> page 5 becomes [4, 5]
    page 6: [6], next page: [8] -> page 6 becomes [6, 7]

    This pass is local and conservative.
    """
    page_to_structure = {
        structure.page_number: structure
        for structure in structures
    }

    max_page = max(page_to_structure.keys()) if page_to_structure else 0

    for structure in structures:
        if not _is_repairable_page(structure):
            continue

        current_numbers = sorted(set(structure.visible_step_numbers))

        if not current_numbers:
            continue

        current_max = max(current_numbers)

        next_numbers: list[int] = []

        for next_page in range(structure.page_number + 1, max_page + 1):
            next_structure = page_to_structure.get(next_page)

            if not next_structure or not _is_repairable_page(next_structure):
                continue

            if next_structure.visible_step_numbers:
                next_numbers = sorted(set(next_structure.visible_step_numbers))
                break

        if not next_numbers:
            continue

        next_min = min(next_numbers)

        if next_min - current_max == 2:
            inferred = current_max + 1
            structure.visible_step_numbers = sorted(set(current_numbers + [inferred]))

            _add_repair_metadata(
                structure,
                repair_type="same_page_pair_repair",
                notes=[
                    f"Inferred missing step {inferred} between current page step {current_max} "
                    f"and next detected step {next_min}."
                ],
                confidence=0.70,
            )


def _repair_step_sequence_across_pages(
    structures: list[ManualPageStructure],
) -> None:
    """
    Sequence-aware repair.

    This fills larger gaps across repairable pages.

    Example:
    page 7: [8]
    page 8: []
    page 9: [11]
    => page 8 becomes [9, 10]

    Rules:
    - Never repair cover/back_matter/parts_inventory pages.
    - Only fill gaps between two anchor pages.
    - Only use pages with assembly visual signals or repairable page type.
    - Mark all inferred numbers in metadata.
    """
    if not structures:
        return

    structures_sorted = sorted(structures, key=lambda s: s.page_number)

    anchor_indices = [
        idx
        for idx, structure in enumerate(structures_sorted)
        if structure.visible_step_numbers and _is_repairable_page(structure)
    ]

    if len(anchor_indices) < 2:
        return

    for left_anchor_pos, right_anchor_pos in zip(anchor_indices, anchor_indices[1:]):
        left = structures_sorted[left_anchor_pos]
        right = structures_sorted[right_anchor_pos]

        left_max = max(left.visible_step_numbers)
        right_min = min(right.visible_step_numbers)

        if right_min <= left_max:
            continue

        missing_numbers = list(range(left_max + 1, right_min))

        if not missing_numbers:
            continue

        candidate_pages = [
            structure
            for structure in structures_sorted[left_anchor_pos + 1:right_anchor_pos]
            if _is_repairable_page(structure)
        ]

        if not candidate_pages:
            continue

        # If there are no empty candidate pages, do not force a repair here.
        empty_candidate_pages = [
            structure
            for structure in candidate_pages
            if not structure.visible_step_numbers
        ]

        if not empty_candidate_pages:
            continue

        # Distribute missing numbers across empty candidate pages.
        # For one empty page and two missing numbers, assign both to that page.
        if len(empty_candidate_pages) == 1:
            target = empty_candidate_pages[0]
            target.visible_step_numbers = sorted(
                set(target.visible_step_numbers + missing_numbers)
            )

            if target.page_type in {
                ManualPageType.informational,
                ManualPageType.unknown,
            }:
                target.page_type = ManualPageType.assembly_step

            _add_repair_metadata(
                target,
                repair_type="cross_page_sequence_repair",
                notes=[
                    f"Inferred steps {missing_numbers} between page {left.page_number} "
                    f"step {left_max} and page {right.page_number} step {right_min}."
                ],
                confidence=0.65,
            )

            continue

        # If multiple empty pages exist, assign one step at a time in page order.
        for missing_number, target in zip(missing_numbers, empty_candidate_pages):
            target.visible_step_numbers = sorted(
                set(target.visible_step_numbers + [missing_number])
            )

            if target.page_type in {
                ManualPageType.informational,
                ManualPageType.unknown,
            }:
                target.page_type = ManualPageType.assembly_step

            _add_repair_metadata(
                target,
                repair_type="cross_page_sequence_repair",
                notes=[
                    f"Inferred step {missing_number} between page {left.page_number} "
                    f"step {left_max} and page {right.page_number} step {right_min}."
                ],
                confidence=0.65,
            )


def _repair_inventory_plus_bottom_step(
    structures: list[ManualPageStructure],
) -> None:
    """
    Handle IKEA pages that contain inventory at the top and a real first step below.

    Example:
    UTESPELARE page 3:
    - top: inventory grid
    - bottom: Step 1 diagram

    OCR often misses the bottom step number because inventory filtering is aggressive.
    If the page is parts_inventory, has assembly visuals, and the next detected
    assembly page starts at step 2, infer step 1 on this page.
    """
    structures_sorted = sorted(structures, key=lambda s: s.page_number)

    for idx, structure in enumerate(structures_sorted):
        if structure.page_type != ManualPageType.parts_inventory:
            continue

        metadata = structure.metadata_json or {}

        if metadata.get("has_assembly_visual_signals") is not True:
            continue

        next_numbers: list[int] = []

        for next_structure in structures_sorted[idx + 1:]:
            if not _is_repairable_page(next_structure):
                continue

            if next_structure.visible_step_numbers:
                next_numbers = next_structure.visible_step_numbers
                break

        if not next_numbers:
            continue

        if min(next_numbers) == 2:
            structure.page_type = ManualPageType.mixed_inventory_and_step
            structure.visible_step_numbers = sorted(set(structure.visible_step_numbers + [1]))

            _add_repair_metadata(
                structure,
                repair_type="inventory_plus_bottom_step_repair",
                notes=[
                    "Detected parts inventory page with assembly visual signals before step 2; "
                    "inferred bottom assembly step 1."
                ],
                confidence=0.65,
            )


def _repair_trailing_final_step(
    structures: list[ManualPageStructure],
) -> None:
    """
    Repair final missing step after the last detected step.

    Example:
    EKENÄSET:
    page 10: [11, 12]
    page 11: informational with assembly visuals
    page 12: back matter
    => page 11 likely contains [13]

    Conservative rule:
    - Only repair if there is exactly one repairable page between the last anchor and back matter.
    - That page must have assembly visual signals.
    """
    structures_sorted = sorted(structures, key=lambda s: s.page_number)

    anchors = [
        structure
        for structure in structures_sorted
        if structure.visible_step_numbers and _is_repairable_page(structure)
    ]

    if not anchors:
        return

    last_anchor = anchors[-1]
    last_number = max(last_anchor.visible_step_numbers)

    trailing_pages = [
        structure
        for structure in structures_sorted
        if structure.page_number > last_anchor.page_number
        and _is_repairable_page(structure)
        and not structure.visible_step_numbers
    ]

    if len(trailing_pages) != 1:
        return

    target = trailing_pages[0]
    metadata = target.metadata_json or {}

    if metadata.get("has_assembly_visual_signals") is not True:
        return

    inferred = last_number + 1

    if inferred > 50:
        return

    target.page_type = ManualPageType.assembly_step
    target.visible_step_numbers = [inferred]

    _add_repair_metadata(
        target,
        repair_type="trailing_final_step_repair",
        notes=[
            f"Inferred final step {inferred} after last detected step {last_number} "
            f"on page {last_anchor.page_number}."
        ],
        confidence=0.65,
    )


def _apply_structure_repairs(
    structures: list[ManualPageStructure],
) -> None:
    """
    Apply repair passes in safe order.

    1. Inventory+bottom-step repair handles mixed inventory/step pages.
    2. Same-page pair repair fills one missing number on pages with detected anchors.
    3. Cross-page repair fills empty pages between anchors.
    4. Trailing final-step repair handles final assembly page before back matter.
    """
    _repair_inventory_plus_bottom_step(structures)
    _soft_repair_same_page_pairs(structures)
    _repair_step_sequence_across_pages(structures)
    _repair_trailing_final_step(structures)


def detect_manual_structure_for_document(
    db: Session,
    document_id: UUID,
) -> list[ManualPageStructure]:
    pages = (
        db.query(DocumentPage)
        .filter(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number.asc())
        .all()
    )

    if not pages:
        return []

    (
        db.query(ManualPageStructure)
        .filter(ManualPageStructure.document_id == document_id)
        .delete(synchronize_session=False)
    )

    total_pages = len(pages)
    structures: list[ManualPageStructure] = []

    for page in pages:
        image = _load_page_image(page)

        if image is None:
            visible_step_numbers: list[int] = []
            tokens: list[OCRToken] = []
            image_width = 0
            image_height = 0
            inventory_band_bottom = None
        else:
            image_width, image_height = image.size
            tokens = _ocr_page(image)
            inventory_band_bottom = _detect_inventory_band_bottom(
                tokens=tokens,
                image_height=image_height,
            )
            visible_step_numbers = _detect_step_numbers_from_ocr(
                tokens=tokens,
                image_width=image_width,
                image_height=image_height,
            )

        has_inventory_signals = _page_has_inventory_signals(tokens)

        has_assembly_visual_signals = _page_has_assembly_visual_signals(
            db=db,
            document_id=document_id,
            page_number=page.page_number,
        )

        page_type, confidence, metadata = _classify_page(
            page_number=page.page_number,
            total_pages=total_pages,
            visible_step_numbers=visible_step_numbers,
            has_inventory_signals=has_inventory_signals,
            has_assembly_visual_signals=has_assembly_visual_signals,
            tokens=tokens,
        )

        metadata.update(
            {
                "detector": "heuristic_ocr_v3",
                "image_width": image_width,
                "image_height": image_height,
                "inventory_band_bottom": inventory_band_bottom,
                "ocr_tokens_preview": [
                    {
                        "text": token.text,
                        "conf": token.conf,
                        "left": token.left,
                        "top": token.top,
                        "width": token.width,
                        "height": token.height,
                    }
                    for token in tokens[:100]
                ],
            }
        )

        structure = ManualPageStructure(
            document_id=document_id,
            page_number=page.page_number,
            page_type=page_type,
            visible_step_numbers=visible_step_numbers,
            confidence=confidence,
            metadata_json=metadata,
        )

        db.add(structure)
        structures.append(structure)

    db.flush()

    _apply_structure_repairs(structures)

    db.commit()

    for structure in structures:
        db.refresh(structure)

    return structures


def get_manual_structure_for_document(
    db: Session,
    document_id: UUID,
) -> list[ManualPageStructure]:
    return (
        db.query(ManualPageStructure)
        .filter(ManualPageStructure.document_id == document_id)
        .order_by(ManualPageStructure.page_number.asc())
        .all()
    )