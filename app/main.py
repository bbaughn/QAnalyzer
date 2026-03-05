from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, init_db
from app.repository import JobRepository
from app.schemas import AnalyzeRequest, AnalyzeResultResponse, AnalyzeSubmissionResponse, JobStatusResponse
from app.services.analysis import (
    _build_sections,
    _coalesce_key_segments_same_label,
    _coalesce_tempo_segments,
    _confirm_key_segments,
)

app = FastAPI(title=settings.app_name, version=settings.app_version)
UI_FILE = Path(__file__).resolve().parent / "static" / "index.html"


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(UI_FILE)


@app.post("/v1/analyze", response_model=AnalyzeSubmissionResponse)
def submit_analysis(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalyzeSubmissionResponse:
    repo = JobRepository(db)
    job = repo.create_job(
        source_type=payload.source_type,
        source=payload.source,
        analysis_profile=payload.analysis_profile,
        options=payload.options,
    )
    return AnalyzeSubmissionResponse(job_id=job.id, status="queued")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    repo = JobRepository(db)
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    failure = None
    if job.status.value == "failed":
        failure = {"code": job.error_code or "unknown_error", "message": job.error_message or "Unknown failure"}

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        submitted_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        failure=failure,
    )


@app.get("/v1/results/{job_id}", response_model=AnalyzeResultResponse)
def get_result(job_id: str, db: Session = Depends(get_db)) -> AnalyzeResultResponse:
    repo = JobRepository(db)
    job = repo.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status.value != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job status is {job.status.value}")
    result = dict(job.result_json or {})
    raw_tempo = result.get("tempo_segments")
    confirmed_tempo = []
    if isinstance(raw_tempo, list):
        confirmed_tempo = _coalesce_tempo_segments(raw_tempo, settings.tempo_section_min_delta_bpm)
    raw_keys = result.get("key_segments_raw")
    confirmed_keys = []
    if isinstance(raw_keys, list) and raw_keys:
        confirmed_keys = _confirm_key_segments(_coalesce_key_segments_same_label(raw_keys))
    else:
        raw_or_confirmed = result.get("key_segments")
        if isinstance(raw_or_confirmed, list):
            confirmed_keys = _confirm_key_segments(raw_or_confirmed)
    key_value_segments = _coalesce_key_segments_same_label(raw_keys) if isinstance(raw_keys, list) else confirmed_keys

    legacy_sections = result.get("sections")
    if isinstance(legacy_sections, list) and legacy_sections:
        first = legacy_sections[0]
        if isinstance(first, dict) and "label" in first and "tempo_bpm" not in first:
            result["form_sections"] = legacy_sections
            result.pop("sections", None)

    if "sections" not in result:
        legacy_overall = result.get("overall_segments")
        if isinstance(legacy_overall, list):
            result["sections"] = legacy_overall
        else:
            duration = float(result.get("track_info", {}).get("duration_sec", 0.0))
            result["sections"] = _build_sections(
                tempo_segments=confirmed_tempo,
                key_segments=confirmed_keys,
                key_value_segments=key_value_segments,
                duration_sec=duration,
                fuzz_sec=settings.segment_boundary_fuzz_sec,
            )

    if "form_sections" not in result and isinstance(legacy_sections, list):
        first = legacy_sections[0] if legacy_sections else {}
        if isinstance(first, dict) and "label" in first:
            result["form_sections"] = legacy_sections

    if not isinstance(result.get("sections"), list):
        duration = float(result.get("track_info", {}).get("duration_sec", 0.0))
        result["sections"] = _build_sections(
            tempo_segments=confirmed_tempo,
            key_segments=confirmed_keys,
            key_value_segments=key_value_segments,
            duration_sec=duration,
            fuzz_sec=settings.segment_boundary_fuzz_sec,
        )

    result.pop("tempo_segments", None)
    result.pop("key_segments", None)
    result.pop("key_segments_raw", None)
    result.pop("overall_segments", None)
    return AnalyzeResultResponse(job_id=job.id, status="succeeded", result=result)
