from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "EDM Analysis Service"
    app_version: str = "0.1.0"
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./data/edm_analysis.db")
    storage_root: Path = Path(os.getenv("STORAGE_ROOT", "./data/storage"))
    temp_root: Path = Path(os.getenv("TEMP_ROOT", "./data/tmp"))
    analysis_sr: int = int(os.getenv("ANALYSIS_SAMPLE_RATE", "44100"))
    harmonic_conf_threshold: float = float(os.getenv("HARMONIC_CONF_THRESHOLD", "0.32"))
    harmonic_consecutive_beats: int = int(os.getenv("HARMONIC_CONSECUTIVE_BEATS", "8"))
    tempo_section_min_delta_bpm: float = float(os.getenv("TEMPO_SECTION_MIN_DELTA_BPM", "2.0"))
    key_change_min_persist_windows: int = int(os.getenv("KEY_CHANGE_MIN_PERSIST_WINDOWS", "3"))
    key_change_conf_margin: float = float(os.getenv("KEY_CHANGE_CONF_MARGIN", "0.08"))
    key_change_min_confidence: float = float(os.getenv("KEY_CHANGE_MIN_CONFIDENCE", "0.45"))
    segment_boundary_fuzz_sec: float = float(os.getenv("SEGMENT_BOUNDARY_FUZZ_SEC", "0.75"))
    queue_poll_seconds: float = float(os.getenv("QUEUE_POLL_SECONDS", "3.0"))
    worker_loop_sleep_seconds: float = float(os.getenv("WORKER_LOOP_SLEEP_SECONDS", "2.0"))


settings = Settings()
settings.storage_root.mkdir(parents=True, exist_ok=True)
settings.temp_root.mkdir(parents=True, exist_ok=True)
