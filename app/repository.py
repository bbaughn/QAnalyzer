from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Job, JobStatus


class JobRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_job(self, *, source_type: str, source: str, analysis_profile: str, options: dict) -> Job:
        from app.config import settings as app_settings
        job = Job(source_type=source_type, source=source, analysis_profile=analysis_profile, options=options, analyzer_version=app_settings.app_version)
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job

    def get_job(self, job_id: str) -> Job | None:
        return self.db.get(Job, job_id)

    def claim_next_queued_job(self) -> Job | None:
        stmt = select(Job).where(Job.status == JobStatus.queued).order_by(Job.created_at.asc()).limit(1)
        job = self.db.execute(stmt).scalar_one_or_none()
        if job is None:
            return None
        job.status = JobStatus.running
        job.started_at = datetime.utcnow()
        job.error_code = None
        job.error_message = None
        self.db.commit()
        self.db.refresh(job)
        return job

    def mark_succeeded(self, job_id: str, result_json: dict, audio_sha256: str | None) -> None:
        job = self.db.get(Job, job_id)
        if not job:
            return
        job.status = JobStatus.succeeded
        job.finished_at = datetime.utcnow()
        job.result_json = result_json
        job.audio_sha256 = audio_sha256
        self.db.commit()

    def mark_failed(self, job_id: str, code: str, message: str) -> None:
        job = self.db.get(Job, job_id)
        if not job:
            return
        job.status = JobStatus.failed
        job.error_code = code
        job.error_message = message
        job.finished_at = datetime.utcnow()
        self.db.commit()
