#!/usr/bin/env bash
set -Eeuo pipefail

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8002}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${PORT:-${STREAMLIT_PORT:-7860}}"

export TRAFFIC_LAW_API_URL="${TRAFFIC_LAW_API_URL:-http://127.0.0.1:${API_PORT}}"

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI backend on ${API_HOST}:${API_PORT}"
python -m uvicorn api.main:app --host "${API_HOST}" --port "${API_PORT}" &
API_PID="$!"

echo "Waiting for FastAPI health check"
for _ in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${API_PID}" 2>/dev/null; then
    echo "FastAPI exited before becoming healthy" >&2
    wait "${API_PID}"
  fi
  sleep 1
done

echo "Starting Streamlit frontend on ${STREAMLIT_HOST}:${STREAMLIT_PORT}"
exec streamlit run frontend/app.py \
  --server.port "${STREAMLIT_PORT}" \
  --server.address "${STREAMLIT_HOST}" \
  --browser.gatherUsageStats false
