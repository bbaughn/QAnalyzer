from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


def _progress_dir() -> Path:
    d = settings.storage_root / "_job_meta"
    d.mkdir(parents=True, exist_ok=True)
    return d


def progress_path(job_id: str) -> Path:
    return _progress_dir() / f"{job_id}.json"


def read_progress(job_id: str) -> dict[str, Any] | None:
    p = progress_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_progress(job_id: str, payload: dict[str, Any]) -> None:
    p = progress_path(job_id)
    try:
        p.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return
