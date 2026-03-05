from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

from app.config import settings
from app.services.errors import MediaDecodeError, SourceError


def _run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaDecodeError(f"Command failed ({' '.join(cmd)}): {proc.stderr.strip()}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _normalize_to_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            str(settings.analysis_sr),
            "-ac",
            "1",
            str(dst),
        ]
    )


def ingest_source(job_id: str, source_type: str, source: str) -> tuple[Path, str]:
    job_dir = settings.storage_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = job_dir / "audio.wav"

    if source_type == "file":
        src = Path(source)
        if not src.exists():
            raise SourceError(f"File source not found: {source}")
        copied = job_dir / src.name
        shutil.copy2(src, copied)
        _normalize_to_wav(copied, normalized_path)
        return normalized_path, _sha256_file(normalized_path)

    if source_type == "youtube":
        downloaded = job_dir / "source.%(ext)s"
        attempts = [
            [
                "yt-dlp",
                "--no-update",
                "--no-playlist",
                "-f",
                "bestaudio[ext=m4a]/bestaudio/best",
                "--extractor-args",
                "youtube:player_client=android,web",
                "-x",
                "--audio-format",
                "wav",
                "-o",
                str(downloaded),
                source,
            ],
            [
                "yt-dlp",
                "--no-update",
                "--no-playlist",
                "-f",
                "bestaudio/best",
                "-x",
                "--audio-format",
                "wav",
                "-o",
                str(downloaded),
                source,
            ],
        ]
        last_error = ""
        for cmd in attempts:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                last_error = ""
                break
            last_error = (proc.stderr or proc.stdout or "").strip()
        if last_error:
            raise SourceError(
                "Failed to fetch source URL. "
                "Try updating yt-dlp in your environment and retry. "
                f"Downloader error: {last_error}"
            )

        candidates = list(job_dir.glob("source.*"))
        if not candidates:
            raise SourceError("No media file produced by downloader")
        src_media = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
        _normalize_to_wav(src_media, normalized_path)
        return normalized_path, _sha256_file(normalized_path)

    raise SourceError(f"Unsupported source_type: {source_type}")
