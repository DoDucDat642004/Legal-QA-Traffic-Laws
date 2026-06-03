#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "== Git status =="
git status --short --branch

echo
echo "== Secret scan =="
if rg -n \
  "hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{20,}" \
  -g '!.git' \
  -g '!data/models/**' \
  -g '!data/processed/**' \
  -g '!data/raw/**' \
  -g '!data/graph/**' \
  .; then
  echo "Secret-like token found. Rotate the token and remove it before pushing." >&2
  exit 1
fi

echo "No high-confidence secret tokens found."

echo
echo "== Syntax checks =="
find api frontend scripts src -name '*.py' -print0 | xargs -0 "$PYTHON_BIN" -m py_compile
if [[ -f entrypoint.sh ]]; then
  bash -n entrypoint.sh
fi
find scripts -name '*.sh' -print0 | xargs -0 bash -n

echo
echo "== Unit tests =="
if [[ -d tests ]]; then
  "$PYTHON_BIN" -m unittest discover -s tests
else
  echo "No tests directory found; skipped unit tests."
fi

echo
echo "== Git LFS tracked assets =="
if command -v git-lfs >/dev/null 2>&1; then
  git lfs ls-files | sed -n '1,80p'
else
  echo "git-lfs is not installed; install it before pushing data/model assets." >&2
  exit 1
fi

echo
echo "== Boost retrieval regression =="
BOOST_FILES=(
  "data/eval/35-bgtvt_boost.json"
  "data/eval/336-2025-nd-cp_boost.json"
  "data/eval/qa_qcvn_boost.json"
  "data/eval/168-nd-cp_boost.json"
  "data/eval/real_world_faq_boost.jsonl"
  "data/eval/35-2024-qh15_boost.json"
  "data/eval/36-2024-qh15_boost.json"
)

missing=0
for file in "${BOOST_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    missing=1
    echo "Missing $file; skipping boost regression."
  fi
done

if [[ "$missing" -eq 0 && -d data/processed && -f data/graph/legal_graph.json ]]; then
  joined="$(IFS=,; echo "${BOOST_FILES[*]}")"
  eval_out_dir="/tmp/legal_qa_boost_eval"
  RAG_VECTOR_BACKEND=local \
  RAG_GRAPH_BACKEND=local \
  RAG_ENABLE_EMBEDDINGS=false \
  RAG_STRICT_VECTOR_BACKEND=false \
  RAG_ENABLE_RERANKER=false \
  RAG_ENABLE_AI_PLANNER=false \
  EVAL_USE_SOURCE_DOC_HINT=1 \
  "$PYTHON_BIN" -m src.evaluation.rag_evaluator \
    --ground-truth "$joined" \
    --processed data/processed \
    --graph data/graph/legal_graph.json \
    --top-k 8 \
    --expand-depth 0 \
    --out-dir "$eval_out_dir" \
    --retrieval-only \
    --no-reranker \
    --use-source-doc-hint \
    --continue-on-error

  "$PYTHON_BIN" - "$eval_out_dir/rag_eval_summary.json" <<'PY'
import json
import sys

summary_path = sys.argv[1]
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

failed_cases = summary.get("failed_cases") or []
error_count = int(summary.get("error_count") or 0)
overall_pass = float((summary.get("pass_rates") or {}).get("overall_pass") or 0.0)

if error_count or failed_cases or overall_pass < 1.0:
    failed_ids = ", ".join(str(case.get("id")) for case in failed_cases[:10])
    print(
        f"Boost regression failed: errors={error_count}, "
        f"overall_pass={overall_pass:.4f}, failed_cases=[{failed_ids}]",
        file=sys.stderr,
    )
    sys.exit(1)

print("Boost regression passed with 100% overall pass.")
PY
else
  echo "Required local data is missing; skipped boost regression."
fi

echo
echo "Pre-push checks completed."
