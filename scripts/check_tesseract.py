from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "apps" / "api"
if str(API_PATH) not in sys.path:
    sys.path.insert(0, str(API_PATH))

from app.core.config import settings
from app.services.diagram_extraction import resolve_tesseract_cmd


def main() -> int:
    resolved_cmd = resolve_tesseract_cmd()

    print(f"configured TESSERACT_CMD: {settings.tesseract_cmd or '(not set)'}")
    print(f"resolved executable path: {resolved_cmd or '(not found)'}")

    try:
        import pytesseract
    except Exception as exc:
        print(f"pytesseract importable: no ({exc})")
        print("OCR fallback remains graceful when pytesseract is unavailable.")
        return 0

    print("pytesseract importable: yes")

    if not resolved_cmd:
        print("Tesseract executable was not found. OCR fallback remains graceful.")
        return 0

    pytesseract.pytesseract.tesseract_cmd = resolved_cmd

    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:
        print(f"Tesseract executable check failed: {exc}")
        print("OCR fallback remains graceful if Tesseract cannot be executed.")
        return 0

    print(f"Tesseract version: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
