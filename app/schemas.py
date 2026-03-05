from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    source_type: Literal["youtube", "file"]
    source: str = Field(min_length=1)
    analysis_profile: str = "edm_v1"
    options: dict[str, Any] = Field(default_factory=dict)


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure: dict[str, str] | None = None


class AnalyzeSubmissionResponse(BaseModel):
    job_id: str
    status: Literal["queued"]


class AnalyzeResultResponse(BaseModel):
    job_id: str
    status: Literal["succeeded"]
    result: dict[str, Any]
