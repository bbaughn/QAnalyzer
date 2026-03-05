from __future__ import annotations

import argparse
import time

from app.config import settings
from app.database import SessionLocal, init_db
from app.repository import JobRepository
from app.services.analysis import analyze_audio_file
from app.services.errors import MediaDecodeError, SourceError
from app.services.ingestion import ingest_source


def process_one_job() -> bool:
    db = SessionLocal()
    try:
        repo = JobRepository(db)
        job = repo.claim_next_queued_job()
        if not job:
            return False

        try:
            audio_path, sha, source_meta = ingest_source(job.id, job.source_type, job.source)
            result = analyze_audio_file(str(audio_path), profile=job.analysis_profile)
            result["track"] = {
                "title": source_meta.get("title"),
                "artist": source_meta.get("artist"),
                "source_url": source_meta.get("source_url"),
            }
            repo.mark_succeeded(job.id, result, sha)
        except SourceError as e:
            repo.mark_failed(job.id, "source_error", str(e))
        except MediaDecodeError as e:
            repo.mark_failed(job.id, "media_decode_error", str(e))
        except Exception as e:  # noqa: BLE001
            repo.mark_failed(job.id, "analysis_error", str(e))
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
