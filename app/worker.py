from __future__ import annotations

import argparse
import multiprocessing
import signal
import sys
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


def _run_job_in_process(job_id: str, source_type: str, source: str, analysis_profile: str) -> None:
    """Run a single job. Called in a child process so OOM kills only this process."""
    # Re-init DB connection in child process
    db = SessionLocal()
    try:
        repo = JobRepository(db)

        progress: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "started_at": _iso_utc_now(),
            "current_stage": None,
            "stages": {},
        }
        write_progress(job_id, progress)

        def stage_hook(stage: str) -> None:
            nonlocal progress
            progress = _update_progress(job_id, progress, current_stage=stage)

        try:
            audio_path, sha, source_meta = ingest_source(
                job_id,
                source_type,
                source,
                stage_hook=stage_hook,
            )
            progress = _update_progress(job_id, progress, current_stage="analyze")
            result = analyze_audio_file(str(audio_path), profile=analysis_profile)
            progress = _update_progress(job_id, progress, status="succeeded", finalize_current_stage=True)
            stage_timings = progress.get("stages", {})
            result["timing"] = {
                "stages": stage_timings,
            }
            result["track"] = {
                "title": source_meta.get("title"),
                "artist": source_meta.get("artist"),
                "source_url": source_meta.get("source_url"),
            }
            repo.mark_succeeded(job_id, result, sha)
        except SourceError as e:
            print(f"[worker] Job {job_id} source error: {e}", flush=True)
            progress = _update_progress(job_id, progress, status="failed", finalize_current_stage=True)
            repo.mark_failed(job_id, "source_error", str(e))
        except MediaDecodeError as e:
            print(f"[worker] Job {job_id} media decode error: {e}", flush=True)
            progress = _update_progress(job_id, progress, status="failed", finalize_current_stage=True)
            repo.mark_failed(job_id, "media_decode_error", str(e))
        except Exception as e:  # noqa: BLE001
            print(f"[worker] Job {job_id} unexpected error: {e}", flush=True)
            progress = _update_progress(job_id, progress, status="failed", finalize_current_stage=True)
            repo.mark_failed(job_id, "analysis_error", str(e))
        finally:
            cleanup_job_storage(job_id)
    finally:
        db.close()


def process_one_job() -> bool:
    """Claim next queued job and run it in a child process.

    If the child is OOM-killed (exit code -9), the parent marks the job failed
    so the queue continues.
    """
    db = SessionLocal()
    try:
        repo = JobRepository(db)
        job = repo.claim_next_queued_job()
        if not job:
            return False
        job_id = job.id
        source_type = job.source_type
        source = job.source
        analysis_profile = job.analysis_profile
        print(f"[worker] Claimed job {job_id} ({source_type}: {source[:60]})", flush=True)
    finally:
        db.close()

    # Run in child process so OOM only kills the child
    proc = multiprocessing.Process(
        target=_run_job_in_process,
        args=(job_id, source_type, source, analysis_profile),
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        exit_reason = f"Child process exited with code {proc.exitcode}"
        if proc.exitcode == -signal.SIGKILL:
            exit_reason = "Process killed (likely OOM — out of memory)"
        print(f"[worker] Job {job_id} failed: {exit_reason}", flush=True)

        # Mark job as failed in parent process
        db = SessionLocal()
        try:
            repo = JobRepository(db)
            repo.mark_failed(job_id, "worker_crash", exit_reason)
        finally:
            db.close()

        cleanup_job_storage(job_id)
    else:
        print(f"[worker] Job {job_id} completed successfully", flush=True)

    return True


def main() -> None:
    multiprocessing.set_start_method("fork", force=True)

    parser = argparse.ArgumentParser(description="EDM analysis worker")
    parser.add_argument("--once", action="store_true", help="Process one job and exit")
    args = parser.parse_args()

    init_db()

    if args.once:
        process_one_job()
        return

    print("[worker] Entering main loop", flush=True)
    loop_count = 0
    while True:
        loop_count += 1
        if loop_count % 30 == 1:  # Log every ~60 seconds
            try:
                db = SessionLocal()
                from sqlalchemy import func
                from app.models import Job, JobStatus
                queued = db.query(func.count(Job.id)).filter(Job.status == JobStatus.queued).scalar()
                running = db.query(func.count(Job.id)).filter(Job.status == JobStatus.running).scalar()
                db.close()
                print(f"[worker] Heartbeat: loop={loop_count}, queued={queued}, running={running}", flush=True)
            except Exception as e:
                print(f"[worker] Heartbeat error: {e}", flush=True)
        try:
            had_job = process_one_job()
        except Exception as e:
            print(f"[worker] Unexpected error in process_one_job: {e}", flush=True)
            had_job = False
        if had_job:
            print("[worker] Finished processing a job", flush=True)
        time.sleep(settings.worker_loop_sleep_seconds)


if __name__ == "__main__":
    main()
