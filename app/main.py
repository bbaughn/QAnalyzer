from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, init_db
from app.repository import JobRepository
from app.schemas import AnalyzeRequest, AnalyzeResultResponse, AnalyzeSubmissionResponse, JobStatusResponse
from app.services.progress import read_progress

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

    progress = read_progress(job.id) or {}
    stage_timings = progress.get("stages")
    if isinstance(stage_timings, dict):
        for stage_data in stage_timings.values():
            if isinstance(stage_data, dict):
                stage_data.pop("_started_perf", None)

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        submitted_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        current_stage=progress.get("current_stage"),
        stage_timings=stage_timings if isinstance(stage_timings, dict) else None,
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
    return AnalyzeResultResponse(job_id=job.id, status="succeeded", result=job.result_json or {})


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")


@app.get("/admin/debug")
def admin_debug(token: str = "") -> dict:
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    import shutil
    import subprocess
    info: dict = {}
    for cmd_name in ("node", "nodejs", "deno", "bun"):
        path = shutil.which(cmd_name)
        info[cmd_name] = {"path": path}
        if path:
            try:
                ver = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
                info[cmd_name]["version"] = ver.stdout.strip()
            except Exception as e:
                info[cmd_name]["error"] = str(e)
    try:
        ytdlp = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=5)
        info["yt-dlp"] = {"version": ytdlp.stdout.strip()}
    except Exception as e:
        info["yt-dlp"] = {"error": str(e)}
    cookies_file = os.environ.get("YTDLP_COOKIES_FILE", "")
    info["cookies"] = {"env": cookies_file, "exists": Path(cookies_file).exists() if cookies_file else False}
    return info


@app.post("/admin/reset-stuck-jobs")
def reset_stuck_jobs(token: str = "", db: Session = Depends(get_db)) -> dict:
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    from app.models import Job, JobStatus
    stuck = db.query(Job).filter(Job.status == JobStatus.running).all()
    count = 0
    for job in stuck:
        job.status = JobStatus.queued
        job.started_at = None
        job.error_code = None
        job.error_message = None
        count += 1
    db.commit()
    return {"reset": count}


@app.post("/admin/upload-cookies")
async def upload_cookies(file: UploadFile, token: str = "") -> dict:
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid admin token")
    cookies_path = settings.storage_root.parent / "cookies.txt"
    content = await file.read()
    cookies_path.write_bytes(content)
    return {"ok": True, "path": str(cookies_path), "size": len(content)}
