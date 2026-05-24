#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"

# Hybrid core with AI assistance: full retrieval coverage, answer scoring on a
# deterministic sample, and AI planner only when rule confidence is low.
export DOTENV_OVERRIDE="${DOTENV_OVERRIDE:-0}"
export GROUND_TRUTH="${GROUND_TRUTH:-data/eval/rag_ground_truth.jsonl}"
export RETRIEVAL_ONLY="${RETRIEVAL_ONLY:-0}"
export ANSWER_SAMPLE_SIZE="${ANSWER_SAMPLE_SIZE:-12}"
export ANSWER_SAMPLE_SEED="${ANSWER_SAMPLE_SEED:-20260522}"
export ANSWER_ON_RETRIEVAL_FAILURES="${ANSWER_ON_RETRIEVAL_FAILURES:-0}"
export EVAL_SLEEP_SECONDS="${EVAL_SLEEP_SECONDS:-0}"
export RUN_EMBED_BENCHMARK="${RUN_EMBED_BENCHMARK:-0}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
export OUT_DIR="${OUT_DIR:-data/eval/reports/rag_core_ai_eval_${STAMP}}"
export CORE_LEXICAL_ONLY="${CORE_LEXICAL_ONLY:-0}"

export RAG_VECTOR_BACKEND="${RAG_VECTOR_BACKEND:-qdrant}"
export RAG_GRAPH_BACKEND="${RAG_GRAPH_BACKEND:-local}"
export RAG_ENABLE_RERANKER="${RAG_ENABLE_RERANKER:-false}"
export NO_RERANKER="${NO_RERANKER:-1}"
export RAG_ENABLE_AI_PLANNER="${RAG_ENABLE_AI_PLANNER:-true}"
export RAG_AI_PLANNER_ALWAYS="${RAG_AI_PLANNER_ALWAYS:-false}"
if [[ "${RAG_AI_PLANNER_ALWAYS}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ && "${ALLOW_SLOW_AI_PLANNER_ALWAYS:-0}" != "1" ]]; then
  echo "RAG_AI_PLANNER_ALWAYS=true is too slow for core AI evaluation; forcing false."
  echo "Set ALLOW_SLOW_AI_PLANNER_ALWAYS=1 only for overnight/full runs."
  export RAG_AI_PLANNER_ALWAYS=false
fi
export RAG_AI_PLANNER_MIN_RULE_CONFIDENCE="${RAG_AI_PLANNER_MIN_RULE_CONFIDENCE:-0.72}"
export RAG_AI_PLANNER_MAX_QUERIES="${RAG_AI_PLANNER_MAX_QUERIES:-2}"
export RAG_MAX_PLANNED_QUERIES="${RAG_MAX_PLANNED_QUERIES:-2}"
export RAG_VECTOR_SEARCH_MULTIPLIER="${RAG_VECTOR_SEARCH_MULTIPLIER:-1}"
export TOP_K="${TOP_K:-8}"
export EXPAND_DEPTH="${EXPAND_DEPTH:-0}"

scripts/run_rag_evaluation.sh
