from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_PATH = REPO_ROOT / "apps" / "api"
if str(API_PATH) not in sys.path:
    sys.path.insert(0, str(API_PATH))

from app.services import diagram_extraction


def test_resolve_tesseract_cmd_prefers_configured_path(monkeypatch, tmp_path):
    configured = tmp_path / "configured-tesseract.exe"
    configured.write_text("", encoding="utf-8")
    common = tmp_path / "common-tesseract.exe"
    common.write_text("", encoding="utf-8")
    path_cmd = tmp_path / "path-tesseract.exe"
    path_cmd.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        diagram_extraction.settings,
        "tesseract_cmd",
        str(configured),
    )
    monkeypatch.setattr(
        diagram_extraction,
        "COMMON_WINDOWS_TESSERACT_PATHS",
        (str(common),),
    )
    monkeypatch.setattr(diagram_extraction.shutil, "which", lambda _: str(path_cmd))

    assert diagram_extraction.resolve_tesseract_cmd() == str(configured)


def test_resolve_tesseract_cmd_falls_back_to_common_windows_path(monkeypatch, tmp_path):
    common = tmp_path / "common-tesseract.exe"
    common.write_text("", encoding="utf-8")
    path_cmd = tmp_path / "path-tesseract.exe"
    path_cmd.write_text("", encoding="utf-8")

    monkeypatch.setattr(diagram_extraction.settings, "tesseract_cmd", None)
    monkeypatch.setattr(
        diagram_extraction,
        "COMMON_WINDOWS_TESSERACT_PATHS",
        (str(common),),
    )
    monkeypatch.setattr(diagram_extraction.shutil, "which", lambda _: str(path_cmd))

    assert diagram_extraction.resolve_tesseract_cmd() == str(common)


def test_resolve_tesseract_cmd_falls_back_to_path_lookup(monkeypatch, tmp_path):
    path_cmd = tmp_path / "path-tesseract.exe"
    path_cmd.write_text("", encoding="utf-8")

    monkeypatch.setattr(diagram_extraction.settings, "tesseract_cmd", None)
    monkeypatch.setattr(diagram_extraction, "COMMON_WINDOWS_TESSERACT_PATHS", ())
    monkeypatch.setattr(diagram_extraction.shutil, "which", lambda _: str(path_cmd))

    assert diagram_extraction.resolve_tesseract_cmd() == str(path_cmd)


def test_resolve_tesseract_cmd_returns_none_when_no_candidate_exists(monkeypatch):
    monkeypatch.setattr(diagram_extraction.settings, "tesseract_cmd", None)
    monkeypatch.setattr(
        diagram_extraction,
        "COMMON_WINDOWS_TESSERACT_PATHS",
        (r"C:\missing\tesseract.exe",),
    )
    monkeypatch.setattr(diagram_extraction.shutil, "which", lambda _: None)

    assert diagram_extraction.resolve_tesseract_cmd() is None
