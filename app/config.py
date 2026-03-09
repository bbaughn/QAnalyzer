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
    harmonic_perc_threshold: float = float(os.getenv("HARMONIC_PERC_THRESHOLD", "0.40"))
    harmonic_atk_threshold: float = float(os.getenv("HARMONIC_ATK_THRESHOLD", "1.0"))
    tempo_section_min_delta_bpm: float = float(os.getenv("TEMPO_SECTION_MIN_DELTA_BPM", "2.0"))
    key_change_min_persist_windows: int = int(os.getenv("KEY_CHANGE_MIN_PERSIST_WINDOWS", "3"))
    key_change_conf_margin: float = float(os.getenv("KEY_CHANGE_CONF_MARGIN", "0.08"))
    key_change_min_confidence: float = float(os.getenv("KEY_CHANGE_MIN_CONFIDENCE", "0.45"))
    segment_boundary_fuzz_sec: float = float(os.getenv("SEGMENT_BOUNDARY_FUZZ_SEC", "0.75"))
    queue_poll_seconds: float = float(os.getenv("QUEUE_POLL_SECONDS", "3.0"))
    worker_loop_sleep_seconds: float = float(os.getenv("WORKER_LOOP_SLEEP_SECONDS", "2.0"))
    tempo_correction_tg_ratio: float = float(os.getenv("TEMPO_CORRECTION_TG_RATIO", "0.75"))
    perc_hpss_rescue_atk_p95: float = float(os.getenv("PERC_HPSS_RESCUE_ATK_P95", "2.0"))
    min_section_key_sec: float = float(os.getenv("MIN_SECTION_KEY_SEC", "30.0"))
    min_section_tempo_sec: float = float(os.getenv("MIN_SECTION_TEMPO_SEC", "30.0"))
    no_key_max_competitive_pcs: int = int(os.getenv("NO_KEY_MAX_COMPETITIVE_PCS", "4"))
    tempo_reliable_onset_cv_min: float = float(os.getenv("TEMPO_RELIABLE_ONSET_CV_MIN", "0.6"))
    no_key_min_winner_margin: float = float(os.getenv("NO_KEY_MIN_WINNER_MARGIN", "0.10"))


settings = Settings()
settings.storage_root.mkdir(parents=True, exist_ok=True)
settings.temp_root.mkdir(parents=True, exist_ok=True)
