# GCP Balanced Deployment (Cloud Run + Pub/Sub + Cloud SQL + GCS)

## Recommended topology
- API service: Cloud Run (`edm-api`)
- Worker service: Cloud Run (`edm-worker`) subscribed to Pub/Sub push or polling DB
- Queue: Pub/Sub topic `analysis-jobs`
- Metadata DB: Cloud SQL Postgres
- Artifacts/audio: Cloud Storage bucket with lifecycle TTL

## Minimal first deployment
1. Build/push image with Cloud Build.
2. Deploy `edm-api` Cloud Run service.
3. Deploy `edm-worker` Cloud Run service with command `python -m app.worker`.
4. Configure Cloud SQL connection and set `DATABASE_URL`.
5. Create GCS bucket and lifecycle policy to expire objects (e.g., 7 days).

## Environment variables
- `DATABASE_URL`
- `STORAGE_ROOT` (for local fallback; for GCS integration add adapter layer)
- `TEMP_ROOT`
- `ANALYSIS_SAMPLE_RATE`
- `HARMONIC_CONF_THRESHOLD`
- `HARMONIC_CONSECUTIVE_BEATS`

## Notes
- This repository currently uses a DB queue pattern for local/dev simplicity.
- For higher scale, add a Pub/Sub enqueue/dequeue adapter while preserving API contract.
