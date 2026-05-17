#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export RAG_VECTOR_BACKEND="${RAG_VECTOR_BACKEND:-qdrant}"
export RAG_GRAPH_BACKEND="${RAG_GRAPH_BACKEND:-neo4j}"
export RAG_EMBEDDING_MODEL="${RAG_EMBEDDING_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
export RAG_EMBEDDING_BACKEND="${RAG_EMBEDDING_BACKEND:-auto}"
export RAG_EMBEDDING_DEVICE="${RAG_EMBEDDING_DEVICE:-cpu}"
export RAG_OPENVINO_DEVICE="${RAG_OPENVINO_DEVICE:-CPU}"
export RAG_ENABLE_RERANKER="${RAG_ENABLE_RERANKER:-false}"
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

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-data/eval/reports/rag_eval_${STAMP}}"
mkdir -p "$OUT_DIR"

EVAL_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  EVAL_ARGS+=(--limit "$LIMIT")
fi
if [[ -n "${RETRIEVAL_ONLY:-}" ]]; then
  EVAL_ARGS+=(--retrieval-only)
fi
if [[ -n "${NO_RERANKER:-}" ]]; then
  EVAL_ARGS+=(--no-reranker)
fi

python3 -m src.evaluation.embedding_benchmark \
  --backend "$RAG_EMBEDDING_BACKEND" \
  --model "$RAG_EMBEDDING_MODEL" \
  --batch-size "${RAG_EMBEDDING_BATCH_SIZE:-64}" \
  --limit "${EMBED_BENCH_LIMIT:-512}" \
  --out "$OUT_DIR/embedding_benchmark.json"

python3 -m src.evaluation.rag_evaluator \
  --ground-truth "${GROUND_TRUTH:-data/eval/rag_ground_truth.jsonl}" \
  --processed "${PROCESSED_DIR:-data/processed}" \
  --graph "${GRAPH_PATH:-data/graph/legal_graph.json}" \
  --top-k "${TOP_K:-8}" \
  --expand-depth "${EXPAND_DEPTH:-2}" \
  --out-dir "$OUT_DIR" \
  --continue-on-error \
  --progress-every "${PROGRESS_EVERY:-10}" \
  --min-recall-at-k "${MIN_RECALL_AT_K:-0.70}" \
  --min-ref-hit "${MIN_REF_HIT:-0.70}" \
  --min-overall-pass "${MIN_OVERALL_PASS:-0.60}" \
  "${EVAL_ARGS[@]}"

echo "RAG evaluation finished: $OUT_DIR"
