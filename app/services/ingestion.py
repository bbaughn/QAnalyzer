from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from urllib.parse import parse_qs, urlparse
from pathlib import Path
from typing import Callable

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


def _youtube_cache_key(source: str) -> str:
    return hashlib.sha256(_canonical_youtube_source(source).encode("utf-8")).hexdigest()


def _canonical_youtube_source(source: str) -> str:
    s = source.strip()
    parsed = urlparse(s)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    query = parse_qs(parsed.query)

    if "youtube.com" in host:
        video_id = query.get("v", [None])[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    if host.endswith("youtu.be"):
        video_id = path.strip("/").split("/", 1)[0]
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return s


def _youtube_cache_path(source: str) -> Path:
    key = _youtube_cache_key(source)
    cache_dir = settings.storage_root / "_youtube_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.wav"


def _youtube_metadata_cache_path(source: str) -> Path:
    key = _youtube_cache_key(source)
    cache_dir = settings.storage_root / "_youtube_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _yt_dlp_cmd_prefix() -> list[str]:
    resolved = shutil.which("yt-dlp")
    if resolved:
        return [resolved]

    venv_candidate = Path(sys.executable).with_name("yt-dlp")
    if venv_candidate.exists():
        return [str(venv_candidate)]

    # Fall back to module invocation in the active Python environment.
    return [sys.executable, "-m", "yt_dlp"]


def _normalize_topic_artist(artist: str | None) -> str | None:
    if not artist:
        return artist
    cleaned = artist.strip()
    return re.sub(r"\s*-\s*topic\s*$", "", cleaned, flags=re.IGNORECASE).strip()


def _derive_artist_title(raw_title: str, uploader: str | None) -> tuple[str | None, str | None]:
    uploader_norm = _normalize_topic_artist(uploader)
    title = (raw_title or "").strip()
    if not title:
        return None, uploader_norm

    # Strip bracketed descriptors (e.g. [Official Video], [HD], [XYZ Records]).
    title = re.sub(r"\[[^\]]*\]", "", title).strip()
    title = re.sub(r"\s{2,}", " ", title).strip(" -|:")

    # If quoted text appears, treat the quoted text as title and the remainder as artist.
    quote_match = re.search(r"[\"“']([^\"”']+)[\"”']", title)
    if quote_match:
        quoted = quote_match.group(1).strip()
        remainder = re.sub(r"[\"“']([^\"”']+)[\"”']", "", title).strip(" -|:")
        artist = remainder if remainder else uploader_norm
        return quoted or None, artist or None

    # Common pattern: Artist - Title
    if " - " in title:
        left, right = title.split(" - ", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            return right, left

    return title, uploader_norm


def _extract_youtube_metadata(source: str) -> dict:
    cmd = _yt_dlp_cmd_prefix() + [
        "--no-update",
        "--no-playlist",
        "--extractor-args",
        "youtube:player_client=android,web",
        "--dump-single-json",
        "--no-download",
        source,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return {"title": None, "artist": None, "source_url": source}
    except subprocess.TimeoutExpired:
        return {"title": None, "artist": None, "source_url": source}
    if proc.returncode != 0:
        return {"title": None, "artist": None, "source_url": source}

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"title": None, "artist": None, "source_url": source}

    raw_title = (data.get("track") or data.get("title") or "").strip()
    artist = (data.get("artist") or data.get("creator") or data.get("uploader") or data.get("channel"))
    parsed_title, parsed_artist = _derive_artist_title(raw_title, artist)
    return {
        "title": parsed_title,
        "artist": parsed_artist,
        "source_url": source,
        "youtube_id": data.get("id"),
        "uploader": data.get("uploader"),
        "channel": data.get("channel"),
    }


def ingest_source(
    job_id: str,
    source_type: str,
    source: str,
    stage_hook: Callable[[str], None] | None = None,
) -> tuple[Path, str, dict]:
    job_dir = settings.storage_root / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = job_dir / "audio.wav"

    if source_type == "file":
        if stage_hook:
            stage_hook("download")
        src = Path(source)
        if not src.exists():
            raise SourceError(f"File source not found: {source}")
        copied = job_dir / src.name
        shutil.copy2(src, copied)
        if stage_hook:
            stage_hook("normalize")
        _normalize_to_wav(copied, normalized_path)
        return normalized_path, _sha256_file(normalized_path), {"title": src.stem, "artist": None, "source_url": source}

    if source_type == "youtube":
        if stage_hook:
            stage_hook("download")
        yt_dlp_prefix = _yt_dlp_cmd_prefix()
        cache_path = _youtube_cache_path(source)
        meta_cache_path = _youtube_metadata_cache_path(source)
        if cache_path.exists():
            shutil.copy2(cache_path, normalized_path)
            metadata = {"title": None, "artist": None, "source_url": source}
            if meta_cache_path.exists():
                try:
                    metadata = json.loads(meta_cache_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    metadata = {"title": None, "artist": None, "source_url": source}
            if not metadata.get("title") and not metadata.get("artist"):
                refreshed = _extract_youtube_metadata(source)
                if refreshed.get("title") or refreshed.get("artist"):
                    metadata = refreshed
                    try:
                        meta_cache_path.write_text(json.dumps(metadata), encoding="utf-8")
                    except OSError:
                        pass
            if stage_hook:
                stage_hook("normalize")
            return normalized_path, _sha256_file(normalized_path), metadata

        downloaded = job_dir / "source.%(ext)s"
        attempts = [
            yt_dlp_prefix + [
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
            yt_dlp_prefix + [
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
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
            except FileNotFoundError:
                last_error = (
                    "yt-dlp executable was not found. "
                    "Install yt-dlp or ensure it is available to the worker process."
                )
                continue
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
        if stage_hook:
            stage_hook("normalize")
        _normalize_to_wav(src_media, normalized_path)
        shutil.copy2(normalized_path, cache_path)
        metadata = _extract_youtube_metadata(source)
        try:
            meta_cache_path.write_text(json.dumps(metadata), encoding="utf-8")
        except OSError:
            pass
        return normalized_path, _sha256_file(normalized_path), metadata

    raise SourceError(f"Unsupported source_type: {source_type}")


def cleanup_job_storage(job_id: str) -> None:
    job_dir = settings.storage_root / job_id
    if not job_dir.exists():
        return
    if not job_dir.is_dir():
        return
    shutil.rmtree(job_dir, ignore_errors=True)
