import base64
import mimetypes
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document_page import DocumentPage
from app.models.source_evidence import SourceEvidence


IMAGE_EVIDENCE_TYPES = {
    "page_image",
    "page_render",
    "rendered_page",
    "full_page_image",
    "diagram_region",
}


IMAGE_PATH_KEYS = {
    "storage_key",
    "image_path",
    "file_path",
    "local_path",
    "storage_path",
    "render_path",
    "page_image_path",
    "cropped_image_path",
}


def _project_root() -> Path:
    return Path(".").resolve()


def _storage_root() -> Path:
    return Path(settings.local_storage_dir).resolve()


def _candidate_paths(raw_path: str) -> list[Path]:
    path = Path(raw_path)

    candidates = []

    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((_project_root() / path).resolve())
        candidates.append((_storage_root() / path).resolve())
        candidates.append(Path(raw_path).resolve())

    return candidates


def _resolve_existing_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    for candidate in _candidate_paths(raw_path):
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def _extract_possible_image_paths_from_metadata(metadata: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    for key in IMAGE_PATH_KEYS:
        value = metadata.get(key)

        if isinstance(value, str) and value.strip():
            paths.append(value)

    return paths


def _extract_possible_image_paths_from_row(row: SourceEvidence) -> list[str]:
    paths: list[str] = []

    for attr in IMAGE_PATH_KEYS:
        value = getattr(row, attr, None)

        if isinstance(value, str) and value.strip():
            paths.append(value)

    metadata = row.metadata_json or {}
    paths.extend(_extract_possible_image_paths_from_metadata(metadata))

    return paths


def _encode_image(path: Path) -> dict | None:
    media_type, _ = mimetypes.guess_type(str(path))

    if media_type not in {"image/png", "image/jpeg", "image/webp"}:
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            media_type = "image/jpeg"
        elif path.suffix.lower() == ".webp":
            media_type = "image/webp"
        else:
            media_type = "image/png"

    data = base64.b64encode(path.read_bytes()).decode("utf-8")

    return {
        "path": str(path),
        "media_type": media_type,
        "base64": data,
    }


def get_page_image_evidence_for_pages(
    db: Session,
    document_id: UUID,
    page_numbers: list[int],
    max_images_per_page: int = 1,
) -> list[dict]:
    """
    Finds local image evidence for candidate pages.

    This is intentionally defensive because image paths may be stored differently
    across extraction versions:
    - direct SourceEvidence fields
    - metadata_json fields
    - page render evidence
    - diagram_region crop evidence

    If no image is found, returns an empty list. The safety guard will prevent
    guessed create_step proposals.
    """
    if not page_numbers:
        return []

    pages = (
        db.query(DocumentPage)
        .filter(DocumentPage.document_id == document_id)
        .filter(DocumentPage.page_number.in_(page_numbers))
        .all()
    )

    evidence_images: list[dict] = []
    count_by_page: dict[int, int] = {}

    for page in sorted(pages, key=lambda page: page.page_number):
        if count_by_page.get(page.page_number, 0) >= max_images_per_page:
            continue

        resolved = _resolve_existing_path(page.page_image_key)

        if resolved is None:
            continue

        encoded = _encode_image(resolved)

        if encoded is None:
            continue

        evidence_images.append(
            {
                "document_id": str(document_id),
                "page_number": page.page_number,
                "evidence_id": str(page.id),
                "evidence_type": "full_page_image",
                "path": encoded["path"],
                "media_type": encoded["media_type"],
                "base64": encoded["base64"],
            }
        )

        count_by_page[page.page_number] = count_by_page.get(page.page_number, 0) + 1

    rows = (
        db.query(SourceEvidence)
        .filter(SourceEvidence.document_id == document_id)
        .filter(SourceEvidence.page_number.in_(page_numbers))
        .all()
    )

    # Prefer full-page images over diagram crops.
    rows = sorted(
        rows,
        key=lambda row: (
            0 if row.evidence_type in {"page_image", "page_render", "rendered_page", "full_page_image"} else 1,
            row.page_number,
        ),
    )

    for row in rows:
        if row.evidence_type not in IMAGE_EVIDENCE_TYPES:
            continue

        if count_by_page.get(row.page_number, 0) >= max_images_per_page:
            continue

        possible_paths = _extract_possible_image_paths_from_row(row)

        for raw_path in possible_paths:
            resolved = _resolve_existing_path(raw_path)

            if resolved is None:
                continue

            encoded = _encode_image(resolved)

            if encoded is None:
                continue

            evidence_images.append(
                {
                    "document_id": str(document_id),
                    "page_number": row.page_number,
                    "evidence_id": str(row.id),
                    "evidence_type": row.evidence_type,
                    "path": encoded["path"],
                    "media_type": encoded["media_type"],
                    "base64": encoded["base64"],
                }
            )

            count_by_page[row.page_number] = count_by_page.get(row.page_number, 0) + 1
            break

    return evidence_images