import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


class LocalStorageService:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def build_storage_key(self, filename: str) -> str:
        safe_filename = filename.replace("/", "_").replace("\\", "_")
        return f"manuals/{uuid.uuid4()}_{safe_filename}"

    def build_generated_key(self, *parts: str) -> str:
        clean_parts = [
            part.replace("/", "_").replace("\\", "_").strip()
            for part in parts
            if part
        ]
        return "/".join(clean_parts)

    async def save_upload(self, file: UploadFile, storage_key: str) -> int:
        destination = self.base_dir / storage_key
        destination.parent.mkdir(parents=True, exist_ok=True)

        size = 0

        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                output.write(chunk)

        return size

    def write_bytes(self, storage_key: str, data: bytes) -> int:
        destination = self.base_dir / storage_key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return len(data)

    def get_local_path(self, storage_key: str) -> str:
        return str(self.base_dir / storage_key)


storage_service = LocalStorageService(settings.local_storage_dir)