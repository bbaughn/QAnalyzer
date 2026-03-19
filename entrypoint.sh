#!/bin/sh
set -e

# Start the worker in the background
python -m app.worker &

# Start the API server in the foreground
exec uvicorn app.main:app --host 0.0.0.0 --port 8080
