# Deployment

Production URL: **https://qanalyzer-production.up.railway.app**

Hosted on Railway, built from `Dockerfile` (Python 3.11 + ffmpeg + node + the
ONNX path of basic-pitch).  Railway redeploys automatically on push to `main`.

## Endpoints (no auth)

- `GET /` — minimal UI
- `GET /admin` — admin dashboard (HTML, no auth on the page itself)
- `POST /v1/analyze` — submit a job (`source_type: "youtube_url"`, `source: "<url>"`)
- `GET /v1/jobs/{job_id}` — job status / progress
- `GET /v1/results/{job_id}` — full analyzer output, includes `analyzer_version`
  at the top level and `result.provenance.percussion_debug` for no-drums diagnosis

## Endpoints (require `?token=…`)

The admin token lives in **`.admin-token`** (gitignored, single line).
Read it inline: `curl "…/admin/debug?token=$(cat .admin-token)"`.

- `GET /admin/debug` — env diagnostics (app_version, yt-dlp version, disk usage)
- `GET /admin/jobs` — last 50 jobs (id, source, status, analyzer_version)
- `POST /admin/cancel-job/{job_id}` — mark a job failed
- plus: requeue/delete/cleanup endpoints (see `app/main.py`)

## Quick verification commands

```bash
# Latest jobs (find a job id by source URL)
curl -s "https://qanalyzer-production.up.railway.app/admin/jobs?token=$(cat .admin-token)" \
  | jq '.jobs[] | select(.source | test("YOUTUBE_ID"))'

# Full result + version
curl -s "https://qanalyzer-production.up.railway.app/v1/results/$JOB_ID" \
  | jq '{ver: .analyzer_version, sections: .result.sections, perc: .result.provenance.percussion_debug}'
```
