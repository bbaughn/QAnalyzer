from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.services.ingestion import cleanup_job_storage


def test_cleanup_job_storage_removes_only_job_dir():
    job_id = "cleanup-test-job"
    job_dir = settings.storage_root / job_id
    cache_dir = settings.storage_root / "_youtube_cache"

    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "audio.wav").write_text("dummy", encoding="utf-8")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_marker = cache_dir / "marker.txt"
    cache_marker.write_text("keep", encoding="utf-8")

    cleanup_job_storage(job_id)

    assert not job_dir.exists()
    assert cache_dir.exists()
    assert cache_marker.exists()

    cache_marker.unlink(missing_ok=True)
