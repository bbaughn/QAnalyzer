# EDM Analysis Service

Async internal-use analysis service for EDM-oriented track metadata and timelines.

## Features
- `POST /v1/analyze` queue job from YouTube URL or local file
- `GET /v1/jobs/{job_id}` status
- `GET /v1/results/{job_id}` structured JSON result
- Global stats: `swing`, `no_drums`, `bars_percussion`
- Unified `sections` timeline
- EDM-oriented `form_sections` mapping

## Requirements
- Python 3.12+
- `ffmpeg`
- (Optional for YouTube sources) `yt-dlp`

## Local run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --reload-dir app --reload-include 'app/**/*.py' --reload-exclude '.venv/**' --reload-exclude '**/.venv/**' --reload-exclude 'data/**' --reload-exclude '*.db' --port 8080
```

In another terminal, run worker:
```bash
source .venv/bin/activate
python -m app.worker
```

Then open [http://localhost:8080/](http://localhost:8080/) for the built-in frontend.

## Example usage
```bash
curl -X POST http://localhost:8080/v1/analyze \
  -H 'content-type: application/json' \
  -d '{"source_type":"youtube","source":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}'
```

Then poll:
```bash
curl http://localhost:8080/v1/jobs/871de222-348e-4f5a-be6d-83f1c249e8f9
curl http://localhost:8080/v1/results/871de222-348e-4f5a-be6d-83f1c249e8f9
```

## Output highlights
- `global.swing`
- `global.no_drums`
- `global.bars_percussion`
- `sections` with change reasons (`tempo`, `key`, or both)

## Notes
- This is an internal prototype and does not enforce rights checks on YouTube URLs.
- Audio artifacts should be configured with TTL cleanup in production.
- If your project folder is synced (Dropbox/iCloud), use the allowlist watcher above (`--reload-include 'app/**/*.py'`) so `.venv` and synced metadata cannot trigger reload loops.
- If YouTube jobs fail with format/signature errors, update downloader tooling: `.venv/bin/python -m pip install -U yt-dlp`.
- Segment boundaries use fuzzy matching across tempo/key changes (`SEGMENT_BOUNDARY_FUZZ_SEC`, default `0.75`).
- Tempo changes must exceed `2.0` BPM to open a new section (override with `TEMPO_SECTION_MIN_DELTA_BPM`).
- Tune key-change sensitivity with `KEY_CHANGE_MIN_PERSIST_WINDOWS`, `KEY_CHANGE_CONF_MARGIN`, and `KEY_CHANGE_MIN_CONFIDENCE`.
