#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DOTENV_OVERRIDE="${DOTENV_OVERRIDE:-0}"
export RAG_PROFILE="${RAG_PROFILE:-balanced}"
case "$RAG_PROFILE" in
  exhaustive|full|nightly)
    export RAG_ENABLE_RERANKER="${RAG_ENABLE_RERANKER:-true}"
    export RAG_MAX_PLANNED_QUERIES="${RAG_MAX_PLANNED_QUERIES:-10}"
    export RAG_VECTOR_SEARCH_MULTIPLIER="${RAG_VECTOR_SEARCH_MULTIPLIER:-4}"
    export RAG_ENABLE_AI_PLANNER="${RAG_ENABLE_AI_PLANNER:-true}"
    export RAG_AI_PLANNER_ALWAYS="${RAG_AI_PLANNER_ALWAYS:-true}"
    export RAG_AI_PLANNER_MIN_RULE_CONFIDENCE="${RAG_AI_PLANNER_MIN_RULE_CONFIDENCE:-1.01}"
    export RAG_AI_PLANNER_MAX_QUERIES="${RAG_AI_PLANNER_MAX_QUERIES:-10}"
    export TOP_K="${TOP_K:-16}"
    export EXPAND_DEPTH="${EXPAND_DEPTH:-3}"
    ;;
  balanced|core|production|"")
    export RAG_ENABLE_RERANKER="${RAG_ENABLE_RERANKER:-false}"
    export RAG_MAX_PLANNED_QUERIES="${RAG_MAX_PLANNED_QUERIES:-3}"
    export RAG_VECTOR_SEARCH_MULTIPLIER="${RAG_VECTOR_SEARCH_MULTIPLIER:-2}"
    export RAG_ENABLE_AI_PLANNER="${RAG_ENABLE_AI_PLANNER:-true}"
    export RAG_AI_PLANNER_ALWAYS="${RAG_AI_PLANNER_ALWAYS:-false}"
    export RAG_AI_PLANNER_MIN_RULE_CONFIDENCE="${RAG_AI_PLANNER_MIN_RULE_CONFIDENCE:-0.72}"
    export RAG_AI_PLANNER_MAX_QUERIES="${RAG_AI_PLANNER_MAX_QUERIES:-2}"
    export TOP_K="${TOP_K:-8}"
    export EXPAND_DEPTH="${EXPAND_DEPTH:-1}"
    ;;
  *)
    echo "Unknown RAG_PROFILE=$RAG_PROFILE. Use balanced or exhaustive." >&2
    exit 2
    ;;
esac
if [[ "$RAG_PROFILE" =~ ^(balanced|core|production|)$ && "${RAG_AI_PLANNER_ALWAYS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && "${ALLOW_SLOW_AI_PLANNER_ALWAYS:-0}" != "1" ]]; then
  echo "RAG_AI_PLANNER_ALWAYS=true is too slow/noisy for RAG_PROFILE=$RAG_PROFILE; forcing false."
  echo "Use RAG_PROFILE=exhaustive or ALLOW_SLOW_AI_PLANNER_ALWAYS=1 only for full overnight runs."
  export RAG_AI_PLANNER_ALWAYS=false
fi

if [[ -f ".env" && "${SKIP_DOTENV:-0}" != "1" ]]; then
  eval "$(
    DOTENV_OVERRIDE="${DOTENV_OVERRIDE:-0}" python3 - <<'PY'
import os
import shlex
from dotenv import dotenv_values

override = os.getenv("DOTENV_OVERRIDE", "0").lower() in {"1", "true", "yes", "on"}
for key, value in dotenv_values(".env").items():
    if value is None:
        continue
    if not override and key in os.environ:
        continue
    print(f"export {key}={shlex.quote(value)}")
PY
  )"
fi

export RAG_VECTOR_BACKEND="${RAG_VECTOR_BACKEND:-qdrant}"
export RAG_GRAPH_BACKEND="${RAG_GRAPH_BACKEND:-local}"
export RAG_CANONICAL_BACKEND="${RAG_CANONICAL_BACKEND:-local}"
export RAG_OBJECT_BACKEND="${RAG_OBJECT_BACKEND:-local}"
export RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-bkai-foundation-models/vietnamese-bi-encoder}"
export RAG_EMBEDDING_BACKEND="${RAG_EMBEDDING_BACKEND:-openvino}"
export RAG_EMBEDDING_DIMENSION="${RAG_EMBEDDING_DIMENSION:-768}"
export RAG_EMBEDDING_DEVICE="${RAG_EMBEDDING_DEVICE:-cpu}"
export RAG_EMBEDDING_BATCH_SIZE="${RAG_EMBEDDING_BATCH_SIZE:-64}"
export RAG_EMBEDDING_MAX_LENGTH="${RAG_EMBEDDING_MAX_LENGTH:-256}"
export RAG_ENABLE_EMBEDDINGS="${RAG_ENABLE_EMBEDDINGS:-true}"
export RAG_ENABLE_RERANKER="${RAG_ENABLE_RERANKER:-false}"
export QDRANT_COLLECTION="${QDRANT_COLLECTION:-legal_traffic_records_vi}"
export QDRANT_TIMEOUT="${QDRANT_TIMEOUT:-300}"
export QDRANT_EMBED_TEXT_MAX_CHARS="${QDRANT_EMBED_TEXT_MAX_CHARS:-1800}"
export RAG_MAX_PLANNED_QUERIES="${RAG_MAX_PLANNED_QUERIES:-3}"
export RAG_VECTOR_SEARCH_MULTIPLIER="${RAG_VECTOR_SEARCH_MULTIPLIER:-2}"
export RAG_STRICT_VECTOR_BACKEND="${RAG_STRICT_VECTOR_BACKEND:-true}"
export RAG_ENABLE_AI_PLANNER="${RAG_ENABLE_AI_PLANNER:-true}"
export RAG_AI_PLANNER_ALWAYS="${RAG_AI_PLANNER_ALWAYS:-false}"
export RAG_AI_PLANNER_MODEL="${RAG_AI_PLANNER_MODEL:-${RAG_ANSWER_MODEL:-gemini-3.1-flash-lite}}"
export RAG_AI_PLANNER_MIN_RULE_CONFIDENCE="${RAG_AI_PLANNER_MIN_RULE_CONFIDENCE:-0.72}"
export RAG_AI_PLANNER_MAX_QUERIES="${RAG_AI_PLANNER_MAX_QUERIES:-2}"
export EVAL_USE_SOURCE_DOC_HINT="${EVAL_USE_SOURCE_DOC_HINT:-1}"
export RAG_OPENVINO_DEVICE="${RAG_OPENVINO_DEVICE:-CPU}"
export RAG_OPENVINO_MODEL_DIR="${RAG_OPENVINO_MODEL_DIR:-data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder}"
export RAG_OPENVINO_EXPORT="${RAG_OPENVINO_EXPORT:-false}"
export OPENVINO_TELEMETRY_DISABLE="${OPENVINO_TELEMETRY_DISABLE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
if [[ "${RAG_ALLOW_MODEL_DOWNLOAD:-false}" != "true" ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mpl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-data/eval/reports/rag_eval_${STAMP}}"
mkdir -p "$OUT_DIR"

python3 - <<'PY'
import os

keys = [
    "RAG_PROFILE",
    "RAG_VECTOR_BACKEND",
    "RAG_GRAPH_BACKEND",
    "RAG_EMBEDDING_BACKEND",
    "RAG_EMBEDDING_MODEL",
    "RAG_EMBEDDING_DIMENSION",
    "RAG_EMBEDDING_DEVICE",
    "RAG_EMBEDDING_BATCH_SIZE",
    "RAG_EMBEDDING_MAX_LENGTH",
    "RAG_ENABLE_EMBEDDINGS",
    "RAG_ENABLE_RERANKER",
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "QDRANT_TIMEOUT",
    "QDRANT_EMBED_TEXT_MAX_CHARS",
    "RAG_MAX_PLANNED_QUERIES",
    "RAG_VECTOR_SEARCH_MULTIPLIER",
    "RAG_STRICT_VECTOR_BACKEND",
    "RAG_ENABLE_AI_PLANNER",
    "RAG_AI_PLANNER_ALWAYS",
    "RAG_AI_PLANNER_MODEL",
    "RAG_AI_PLANNER_MIN_RULE_CONFIDENCE",
    "RAG_AI_PLANNER_MAX_QUERIES",
    "EVAL_USE_SOURCE_DOC_HINT",
    "TOP_K",
    "EXPAND_DEPTH",
    "ANSWER_SAMPLE_SIZE",
    "ANSWER_SAMPLE_SEED",
    "ANSWER_ON_RETRIEVAL_FAILURES",
    "EVAL_SLEEP_SECONDS",
    "CORE_LEXICAL_ONLY",
    "RAG_OPENVINO_DEVICE",
    "RAG_OPENVINO_MODEL_DIR",
]
print("RAG evaluation config:")
for key in keys:
    value = os.getenv(key, "")
    if "KEY" in key or "TOKEN" in key or "SECRET" in key:
        value = "<redacted>" if value else ""
    print(f"  {key}={value}")
PY

EVAL_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${ANSWER_SAMPLE_SIZE:-}" ]]; then
  EVAL_ARGS+=(--answer-sample-size "$ANSWER_SAMPLE_SIZE")
fi
if [[ -n "${ANSWER_SAMPLE_SEED:-}" ]]; then
  EVAL_ARGS+=(--answer-sample-seed "$ANSWER_SAMPLE_SEED")
fi
if truthy "${ANSWER_ON_RETRIEVAL_FAILURES:-0}"; then
  EVAL_ARGS+=(--answer-on-retrieval-failures)
fi
if [[ -n "${EVAL_SLEEP_SECONDS:-}" ]]; then
  EVAL_ARGS+=(--sleep-seconds "$EVAL_SLEEP_SECONDS")
fi
if truthy "${CORE_LEXICAL_ONLY:-0}"; then
  EVAL_ARGS+=(--lexical-only)
fi

RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-1}"
if truthy "$RETRIEVAL_ONLY"; then
  EVAL_ARGS+=(--retrieval-only)
fi

NO_RERANKER="${NO_RERANKER:-0}"
if truthy "$NO_RERANKER"; then
  EVAL_ARGS+=(--no-reranker)
fi
if truthy "${USE_RERANKER:-0}"; then
  EVAL_ARGS+=(--use-reranker)
fi
if truthy "${EVAL_USE_SOURCE_DOC_HINT:-0}"; then
  EVAL_ARGS+=(--use-source-doc-hint)
fi

if truthy "${RUN_EMBED_BENCHMARK:-0}"; then
  python3 -m src.evaluation.embedding_benchmark \
    --backend "$RAG_EMBEDDING_BACKEND" \
    --model "$RAG_EMBEDDING_MODEL" \
    --batch-size "$RAG_EMBEDDING_BATCH_SIZE" \
    --limit "${EMBED_BENCH_LIMIT:-128}" \
    --out "$OUT_DIR/embedding_benchmark.json"
else
  echo "Skipping embedding benchmark. Set RUN_EMBED_BENCHMARK=1 to enable it."
fi

python3 -m src.evaluation.rag_evaluator \
  --ground-truth "${GROUND_TRUTH:-data/eval/rag_ground_truth.jsonl}" \
  --processed "${PROCESSED_DIR:-data/processed}" \
  --graph "${GRAPH_PATH:-data/graph/legal_graph.json}" \
  --top-k "${TOP_K:-8}" \
  --expand-depth "${EXPAND_DEPTH:-1}" \
  --out-dir "$OUT_DIR" \
  --continue-on-error \
  --progress-every "${PROGRESS_EVERY:-10}" \
  --min-recall-at-k "${MIN_RECALL_AT_K:-0.70}" \
  --min-ref-hit "${MIN_REF_HIT:-0.70}" \
  --min-overall-pass "${MIN_OVERALL_PASS:-0.60}" \
  ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}

echo "RAG evaluation finished: $OUT_DIR"
