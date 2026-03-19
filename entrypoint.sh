#!/bin/sh
set -e

echo "Starting worker..."
python -m app.worker 2>&1 &
WORKER_PID=$!
echo "Worker started with PID $WORKER_PID"

# If worker dies, log it
(while kill -0 $WORKER_PID 2>/dev/null; do sleep 10; done; echo "WORKER EXITED with code $?") &

echo "Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8080
