from __future__ import annotations

import argparse
from datetime import datetime, timezone
import time
from typing import Any

from app.config import settings
from app.database import SessionLocal, init_db
from app.repository import JobRepository
from app.services.analysis import analyze_audio_file
from app.services.errors import MediaDecodeError, SourceError
from app.services.ingestion import cleanup_job_storage, ingest_source
from app.services.progress import write_progress


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_progress(
    job_id: str,
    payload: dict[str, Any],
    *,
    current_stage: str | None = None,
    status: str | None = None,
    finalize_current_stage: bool = False,
) -> dict[str, Any]:
    now_perf = time.perf_counter()
    now_iso = _iso_utc_now()

    if finalize_current_stage:
        active = payload.get("current_stage")
        if active:
            stages = payload.setdefault("stages", {})
            stage_data = stages.get(active)
            if stage_data and not stage_data.get("finished_at"):
                stage_data["finished_at"] = now_iso
                started_perf = stage_data.get("_started_perf")
                if isinstance(started_perf, (int, float)):
                    stage_data["duration_sec"] = round(max(0.0, now_perf - float(started_perf)), 3)
                stage_data.pop("_started_perf", None)

    if current_stage:
        active = payload.get("current_stage")
        if active != current_stage:
            payload = _update_progress(
                job_id,
                payload,
                finalize_current_stage=True,
            )
            stages = payload.setdefault("stages", {})
            stage = stages.get(current_stage)
            if stage and not stage.get("finished_at"):
                # Stage already active.
                payload["current_stage"] = current_stage
            else:
                stages[current_stage] = {
                    "started_at": now_iso,
                    "finished_at": None,
                    "duration_sec": None,
                    "_started_perf": now_perf,
                }
                payload["current_stage"] = current_stage

    if status:
        payload["status"] = status
        if status in {"succeeded", "failed"}:
            payload["finished_at"] = now_iso
            payload["current_stage"] = None

    write_progress(job_id, payload)
    return payload


def process_one_job() -> bool:
    db = SessionLocal()
    try:
        repo = JobRepository(db)
        job = repo.claim_next_queued_job()
        if not job:
            return False

        progress: dict[str, Any] = {
            "job_id": job.id,
            "status": "running",
            "started_at": _iso_utc_now(),
            "current_stage": None,
            "stages": {},
        }
        write_progress(job.id, progress)

        def stage_hook(stage: str) -> None:
            nonlocal progress
            progress = _update_progress(job.id, progress, current_stage=stage)

        try:
            audio_path, sha, source_meta = ingest_source(
                job.id,
                job.source_type,
                job.source,
                stage_hook=stage_hook,
            )
            progress = _update_progress(job.id, progress, current_stage="analyze")
            result = analyze_audio_file(str(audio_path), profile=job.analysis_profile)
            progress = _update_progress(job.id, progress, status="succeeded", finalize_current_stage=True)
            stage_timings = progress.get("stages", {})
            total_run_sec = None
            if isinstance(job.started_at, datetime):
                total_run_sec = round(max(0.0, (datetime.utcnow() - job.started_at).total_seconds()), 3)
            result["timing"] = {
                "stages": stage_timings,
                "total_run_sec": total_run_sec,
            }
            result["track"] = {
                "title": source_meta.get("title"),
                "artist": source_meta.get("artist"),
                "source_url": source_meta.get("source_url"),
            }
            repo.mark_succeeded(job.id, result, sha)
        except SourceError as e:
            progress = _update_progress(job.id, progress, status="failed", finalize_current_stage=True)
            repo.mark_failed(job.id, "source_error", str(e))
        except MediaDecodeError as e:
            progress = _update_progress(job.id, progress, status="failed", finalize_current_stage=True)
            repo.mark_failed(job.id, "media_decode_error", str(e))
        except Exception as e:  # noqa: BLE001
            progress = _update_progress(job.id, progress, status="failed", finalize_current_stage=True)
            repo.mark_failed(job.id, "analysis_error", str(e))
        finally:
            cleanup_job_storage(job.id)
        return True
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="EDM analysis worker")
    parser.add_argument("--once", action="store_true", help="Process one job and exit")
    args = parser.parse_args()

    init_db()

    if args.once:
        process_one_job()
        return

    while True:
        had_job = process_one_job()
        if not had_job:
            time.sleep(settings.worker_loop_sleep_seconds)


if __name__ == "__main__":
    main()
