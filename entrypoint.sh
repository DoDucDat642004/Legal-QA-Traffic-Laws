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
  export RAG_DEPLOY_FAST_MODE="${RAG_DEPLOY_FAST_MODE:-true}"
  export RAG_VECTOR_BACKEND=local
  export RAG_STRICT_VECTOR_BACKEND=false
  export RAG_GRAPH_BACKEND=local
  export RAG_ENABLE_EMBEDDINGS=false
  export RAG_EMBEDDING_MODEL=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder
  export RAG_EMBEDDING_BACKEND=openvino
  export RAG_OPENVINO_MODEL_DIR=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder
  export RAG_OPENVINO_EXPORT=false
  export RAG_ALLOW_MODEL_DOWNLOAD=false
  export RAG_ENABLE_RERANKER="${RAG_ENABLE_RERANKER:-false}"
  export RAG_ENABLE_RERANKER_FOR_HARD="${RAG_ENABLE_RERANKER_FOR_HARD:-true}"
  export RAG_RERANKER_BACKEND=openvino
  export RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
  export RAG_OPENVINO_RERANKER_MODEL_DIR=data/models/openvino/BAAI_bge-reranker-v2-m3
  export RAG_OPENVINO_RERANKER_EXPORT=false
  export RAG_ALLOW_RERANKER_DOWNLOAD=false
  export RAG_ENABLE_AI_PLANNER="${RAG_ENABLE_AI_PLANNER:-false}"
  export RAG_AI_PLANNER_ALWAYS="${RAG_AI_PLANNER_ALWAYS:-false}"
  export RAG_AI_PLANNER_MIN_RULE_CONFIDENCE="${RAG_AI_PLANNER_MIN_RULE_CONFIDENCE:-0.72}"
  export RAG_RETRIEVAL_MAX_ROUNDS="${RAG_RETRIEVAL_MAX_ROUNDS:-2}"
  export RAG_RETRIEVAL_MAX_SLOTS="${RAG_RETRIEVAL_MAX_SLOTS:-12}"
  export RAG_MAX_PLANNED_QUERIES="${RAG_MAX_PLANNED_QUERIES:-6}"
  export RAG_FAST_MAX_CONTEXTS="${RAG_FAST_MAX_CONTEXTS:-12}"
  export RAG_FAST_TOP_K="${RAG_FAST_TOP_K:-12}"
  export RAG_FAST_EXPAND_DEPTH="${RAG_FAST_EXPAND_DEPTH:-1}"
  export RAG_FAST_MAX_IMAGES="${RAG_FAST_MAX_IMAGES:-6}"
  export RAG_FAST_MAX_PROMPT_IMAGES="${RAG_FAST_MAX_PROMPT_IMAGES:-0}"
  export RAG_API_IMAGE_LIMIT="${RAG_API_IMAGE_LIMIT:-6}"
  export RAG_INCLUDE_GRAPH_TRACE="${RAG_INCLUDE_GRAPH_TRACE:-false}"
  export RAG_INCLUDE_ANSWER_TRACE="${RAG_INCLUDE_ANSWER_TRACE:-true}"
  export RAG_AUTO_VERIFY_CLAIMS="${RAG_AUTO_VERIFY_CLAIMS:-true}"
  export RAG_AUTO_VERIFY_MAX_CLAIMS="${RAG_AUTO_VERIFY_MAX_CLAIMS:-12}"
  export RAG_CHAT_TEXT_DEADLINE_SECONDS="${RAG_CHAT_TEXT_DEADLINE_SECONDS:-300}"
  export RAG_EXTRACTIVE_ANSWER_ONLY=false
  export RAG_ENABLE_SIGN_AI_PROBE="${RAG_ENABLE_SIGN_AI_PROBE:-false}"
  export WARMUP_RAG_ON_START="${WARMUP_RAG_ON_START:-true}"
  export RAG_PROMPT_CONTEXT_TEXT_LIMIT="${RAG_PROMPT_CONTEXT_TEXT_LIMIT:-8000}"
  export RAG_PROMPT_STRUCTURED_TEXT_LIMIT="${RAG_PROMPT_STRUCTURED_TEXT_LIMIT:-16000}"
  if [[ ! "${RAG_ANSWER_MAX_OUTPUT_TOKENS:-}" =~ ^[0-9]+$ ]] || [[ "${RAG_ANSWER_MAX_OUTPUT_TOKENS}" -lt 8192 ]]; then
    export RAG_ANSWER_MAX_OUTPUT_TOKENS=8192
  fi
  if [[ ! "${RAG_ANSWER_MAX_CONTINUATIONS:-}" =~ ^[0-9]+$ ]] || [[ "${RAG_ANSWER_MAX_CONTINUATIONS}" -lt 2 ]]; then
    export RAG_ANSWER_MAX_CONTINUATIONS=2
  fi
  export RAG_EXTRACTIVE_MAX_CONTEXTS="${RAG_EXTRACTIVE_MAX_CONTEXTS:-6}"
  export RAG_EXTRACTIVE_TEXT_LIMIT="${RAG_EXTRACTIVE_TEXT_LIMIT:-1200}"
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

echo "Runtime RAG config: vector=${RAG_VECTOR_BACKEND:-unset}, embeddings=${RAG_ENABLE_EMBEDDINGS:-unset}, reranker=${RAG_ENABLE_RERANKER:-unset}/${RAG_RERANKER_BACKEND:-unset}, fast=${RAG_DEPLOY_FAST_MODE:-unset}"
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

if is_truthy "${WARMUP_RAG_ON_START:-false}"; then
  echo "Warming up RAG backend"
  if ! curl -fsS --max-time "${RAG_WARMUP_TIMEOUT_SECONDS:-180}" "http://127.0.0.1:${API_PORT}/system/status" >/dev/null; then
    echo "RAG warmup did not complete before timeout; continuing startup" >&2
  fi
fi

echo "Starting Streamlit frontend on ${STREAMLIT_HOST}:${STREAMLIT_PORT}"
exec streamlit run frontend/app.py \
  --server.port "${STREAMLIT_PORT}" \
  --server.address "${STREAMLIT_HOST}" \
  --browser.gatherUsageStats false
