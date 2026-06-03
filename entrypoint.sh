#!/usr/bin/env bash
set -Eeuo pipefail

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8002}"
STREAMLIT_HOST="${STREAMLIT_HOST:-0.0.0.0}"
STREAMLIT_PORT="${PORT:-${STREAMLIT_PORT:-7860}}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if is_truthy "${RAG_DEPLOY_FORCE_LOCAL_MODE:-false}"; then
  export RAG_VECTOR_BACKEND=local
  export RAG_STRICT_VECTOR_BACKEND=false
  export RAG_GRAPH_BACKEND=local
  export RAG_ENABLE_EMBEDDINGS=false
  export RAG_EMBEDDING_MODEL=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder
  export RAG_EMBEDDING_BACKEND=openvino
  export RAG_OPENVINO_MODEL_DIR=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder
  export RAG_OPENVINO_EXPORT=false
  export RAG_ALLOW_MODEL_DOWNLOAD=false
  export RAG_ENABLE_RERANKER=true
  export RAG_RERANKER_BACKEND=openvino
  export RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
  export RAG_OPENVINO_RERANKER_MODEL_DIR=data/models/openvino/BAAI_bge-reranker-v2-m3
  export RAG_OPENVINO_RERANKER_EXPORT=false
  export RAG_ALLOW_RERANKER_DOWNLOAD=false
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

export TRAFFIC_LAW_API_URL="${TRAFFIC_LAW_API_URL:-http://127.0.0.1:${API_PORT}}"

mkdir -p \
  "${HOME:-/tmp}/intel" \
  "${HF_HOME:-/tmp/.cache/huggingface}" \
  "${MPLCONFIGDIR:-/tmp/.cache/matplotlib}" \
  "${XDG_CACHE_HOME:-/tmp/.cache}/fontconfig" \
  2>/dev/null || true

echo "Runtime RAG config: vector=${RAG_VECTOR_BACKEND:-unset}, embeddings=${RAG_ENABLE_EMBEDDINGS:-unset}, reranker=${RAG_ENABLE_RERANKER:-unset}/${RAG_RERANKER_BACKEND:-unset}"
echo "Runtime model dirs: embedding=${RAG_OPENVINO_MODEL_DIR:-unset}, reranker=${RAG_OPENVINO_RERANKER_MODEL_DIR:-unset}"

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
