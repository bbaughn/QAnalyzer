#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

API_HOST="127.0.0.1"
API_PORT="8080"
API_URL="http://${API_HOST}:${API_PORT}"

API_LOG="${ROOT_DIR}/data/tmp/api.log"
WORKER_LOG="${ROOT_DIR}/data/tmp/worker.log"
mkdir -p "${ROOT_DIR}/data/tmp"

VENV_PYTHON=""
EXTERNAL_VENV="${HOME}/.local/venvs/qanalyzer"
for candidate in \
  "${ROOT_DIR}/.venv/bin/python" \
  "${ROOT_DIR}/.venv/bin/python3" \
  "${EXTERNAL_VENV}/bin/python" \
  "${EXTERNAL_VENV}/bin/python3"; do
  if [[ -x "${candidate}" ]] && "${candidate}" -c 'import sys; print(sys.version)' >/dev/null 2>&1; then
    VENV_PYTHON="${candidate}"
    break
  fi
done

if [[ -z "${VENV_PYTHON}" ]]; then
  echo "[restart] venv missing or broken — creating at ${EXTERNAL_VENV}..."
  mkdir -p "$(dirname "${EXTERNAL_VENV}")"
  python3 -m venv "${EXTERNAL_VENV}"
  "${EXTERNAL_VENV}/bin/pip" install -r "${ROOT_DIR}/requirements.txt"
  VENV_PYTHON="${EXTERNAL_VENV}/bin/python"
fi

echo "[restart] stopping existing API/worker processes..."
pkill -f "uvicorn app\.main:app" 2>/dev/null || true
pkill -f "app\.worker" 2>/dev/null || true
sleep 0.3

echo "[restart] starting API..."
nohup "${VENV_PYTHON}" -m uvicorn app.main:app --host "${API_HOST}" --port "${API_PORT}" >"${API_LOG}" 2>&1 &
API_PID=$!

echo "[restart] starting worker..."
nohup "${VENV_PYTHON}" -m app.worker >"${WORKER_LOG}" 2>&1 &
WORKER_PID=$!
sleep 0.2
if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
  echo "[restart] Worker failed to stay up. See ${WORKER_LOG}" >&2
  exit 1
fi

echo "[restart] waiting for API health..."
for _ in {1..40}; do
  if curl -fsS "${API_URL}/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

if ! curl -fsS "${API_URL}/healthz" >/dev/null 2>&1; then
  echo "[restart] API failed health check. See ${API_LOG}" >&2
  exit 1
fi

HEALTH_JSON="$(curl -fsS "${API_URL}/healthz")"

echo "[restart] done"
echo "  API:    ${API_URL} (pid ${API_PID})"
echo "  Worker: pid ${WORKER_PID}"
echo "  Health: ${HEALTH_JSON}"
echo "  Logs:"
echo "    - ${API_LOG}"
echo "    - ${WORKER_LOG}"

echo "[restart] process snapshot:"
pgrep -fl "uvicorn app.main:app|app.worker" || true
