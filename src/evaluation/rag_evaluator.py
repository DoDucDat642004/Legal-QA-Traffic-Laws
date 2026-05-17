import argparse
import json
import os
import statistics
import traceback
import time
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    citation_accuracy,
    claim_support_accuracy,
    context_has_ref,
    modality_flags,
    number_accuracy,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from src.rag.legal_graph_rag import LegalGraphRAG


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cases.append(json.loads(line))
            if limit and len(cases) >= limit:
                break
    return cases


def context_text(contexts: list[dict[str, Any]]) -> str:
    return "\n".join(
        ctx.get("rag_text") or ctx.get("source_body_exact") or ctx.get("content") or ""
        for ctx in contexts
    )


def required_modality(case: dict[str, Any]) -> str | None:
    query_type = case.get("query_type")
    if query_type == "SIGN_MEANING":
        return "has_sign"
    if query_type == "TABLE_LOOKUP":
        return "has_table"
    if query_type == "PENALTY":
        return "has_penalty"
    if query_type == "PROCEDURE":
        return "has_procedure"
    return None


def evaluate_case(
    rag: LegalGraphRAG,
    case: dict[str, Any],
    *,
    top_k: int,
    expand_depth: int,
    retrieval_only: bool,
) -> dict[str, Any]:
    start = time.time()
    contexts = rag.retrieve(case["question"], top_k=top_k, expand_depth=expand_depth)
    answer = "" if retrieval_only else rag.generate_answer(case["question"], contexts)
    latency_ms = int((time.time() - start) * 1000)

    retrieved_ids = list(dict.fromkeys(ctx.get("source_chunk_id") or ctx.get("id") or "" for ctx in contexts))
    expected_chunks = case.get("expected_chunks") or []
    acceptable_chunks = case.get("acceptable_chunks") or []
    expected_all = list(dict.fromkeys(expected_chunks + acceptable_chunks))
    refs = case.get("expected_refs") or []
    flags = modality_flags(contexts)
    modality_key = required_modality(case)

    ctx_text = context_text(contexts)
    quote = case.get("reference_quote") or ""
    quote_in_context = token_f1(ctx_text, quote) if quote else 1.0

    retrieval = {
        "recall_at_1": recall_at_k(retrieved_ids, expected_all, 1),
        "recall_at_3": recall_at_k(retrieved_ids, expected_all, 3),
        "recall_at_k": recall_at_k(retrieved_ids, expected_all, top_k),
        "mrr": reciprocal_rank(retrieved_ids, expected_all),
        "ref_hit": 1.0 if context_has_ref(contexts, refs) else 0.0,
        "quote_context_f1": quote_in_context,
        "required_modality_hit": 1.0 if not modality_key or flags.get(modality_key) else 0.0,
    }

    answer_metrics = {
        "answer_token_f1": 1.0,
        "number_accuracy": 1.0,
        "citation_accuracy": 1.0,
        "claim_support_accuracy": 1.0,
    }
    if not retrieval_only:
        answer_metrics = {
            "answer_token_f1": token_f1(answer, case.get("gold_answer", "")),
            "number_accuracy": number_accuracy(answer, case.get("expected_numbers") or []),
            "citation_accuracy": citation_accuracy(answer, case.get("required_citations") or [], refs),
            "claim_support_accuracy": claim_support_accuracy(answer, case.get("required_claims") or []),
        }

    pass_fail = {
        "retrieval_pass": bool(retrieval["recall_at_k"] or retrieval["ref_hit"]),
        "modality_pass": bool(retrieval["required_modality_hit"]),
        "answer_pass": bool(
            retrieval_only
            or (
                answer_metrics["number_accuracy"] >= 0.99
                and answer_metrics["citation_accuracy"] >= 0.6
                and answer_metrics["claim_support_accuracy"] >= 0.45
            )
        ),
    }
    pass_fail["overall_pass"] = all(pass_fail.values())

    return {
        "id": case.get("id"),
        "query_type": case.get("query_type"),
        "question": case.get("question"),
        "expected_chunks": expected_chunks,
        "retrieved_chunks": retrieved_ids,
        "expected_refs": refs,
        "retrieved_refs": [
            {
                "source_chunk_id": ctx.get("source_chunk_id"),
                "modality": ctx.get("rag_modality"),
                "legal_reference": ctx.get("legal_reference"),
                "rag_metadata": ctx.get("rag_metadata"),
                "retrieval_reasons": ctx.get("retrieval_reasons"),
                "retrieval_score": ctx.get("retrieval_score"),
            }
            for ctx in contexts
        ],
        "retrieval": retrieval,
        "answer_metrics": answer_metrics,
        "pass_fail": pass_fail,
        "answer": answer,
        "latency_ms": latency_ms,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(path: tuple[str, str]) -> float:
        vals = [float(r[path[0]][path[1]]) for r in results if path[0] in r and path[1] in r[path[0]]]
        return statistics.mean(vals) if vals else 0.0

    by_type = {}
    for result in results:
        by_type.setdefault(result.get("query_type") or "UNKNOWN", []).append(result)

    type_summary = {}
    for query_type, items in by_type.items():
        type_summary[query_type] = {
            "count": len(items),
            "recall_at_k": statistics.mean(float(i["retrieval"]["recall_at_k"]) for i in items),
            "mrr": statistics.mean(float(i["retrieval"]["mrr"]) for i in items),
            "ref_hit": statistics.mean(float(i["retrieval"]["ref_hit"]) for i in items),
            "overall_pass_rate": statistics.mean(float(i["pass_fail"]["overall_pass"]) for i in items),
        }

    latencies = [int(r.get("latency_ms") or 0) for r in results if r.get("latency_ms") is not None]
    latencies_sorted = sorted(latencies)

    def percentile(p: float) -> int:
        if not latencies_sorted:
            return 0
        idx = min(len(latencies_sorted) - 1, max(0, int(round((len(latencies_sorted) - 1) * p))))
        return latencies_sorted[idx]

    error_count = sum(1 for r in results if r.get("error"))
    return {
        "case_count": len(results),
        "error_count": error_count,
        "retrieval": {
            "recall_at_1": avg(("retrieval", "recall_at_1")),
            "recall_at_3": avg(("retrieval", "recall_at_3")),
            "recall_at_k": avg(("retrieval", "recall_at_k")),
            "mrr": avg(("retrieval", "mrr")),
            "ref_hit": avg(("retrieval", "ref_hit")),
            "required_modality_hit": avg(("retrieval", "required_modality_hit")),
            "quote_context_f1": avg(("retrieval", "quote_context_f1")),
        },
        "answer": {
            "answer_token_f1": avg(("answer_metrics", "answer_token_f1")),
            "number_accuracy": avg(("answer_metrics", "number_accuracy")),
            "citation_accuracy": avg(("answer_metrics", "citation_accuracy")),
            "claim_support_accuracy": avg(("answer_metrics", "claim_support_accuracy")),
        },
        "pass_rates": {
            "retrieval_pass": statistics.mean(float(r["pass_fail"]["retrieval_pass"]) for r in results) if results else 0.0,
            "modality_pass": statistics.mean(float(r["pass_fail"]["modality_pass"]) for r in results) if results else 0.0,
            "answer_pass": statistics.mean(float(r["pass_fail"]["answer_pass"]) for r in results) if results else 0.0,
            "overall_pass": statistics.mean(float(r["pass_fail"]["overall_pass"]) for r in results) if results else 0.0,
        },
        "latency_ms": {
            "mean": int(statistics.mean(latencies)) if latencies else 0,
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(latencies) if latencies else 0,
        },
        "by_query_type": type_summary,
        "failed_cases": [
            {
                "id": r["id"],
                "query_type": r["query_type"],
                "question": r["question"],
                "pass_fail": r["pass_fail"],
                "retrieval": r["retrieval"],
                "expected_chunks": r["expected_chunks"],
                "retrieved_chunks": r["retrieved_chunks"][:8],
            }
            for r in results
            if not r["pass_fail"]["overall_pass"]
        ][:100],
    }


def write_outputs(results: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rag_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out_dir / "rag_eval_results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    with (out_dir / "rag_eval_failures.json").open("w", encoding="utf-8") as f:
        json.dump(summary["failed_cases"], f, ensure_ascii=False, indent=2)
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "env": {
                    key: os.getenv(key)
                    for key in [
                        "RAG_VECTOR_BACKEND",
                        "RAG_GRAPH_BACKEND",
                        "RAG_EMBEDDING_BACKEND",
                        "RAG_EMBEDDING_MODEL",
                        "RAG_EMBEDDING_DEVICE",
                        "RAG_OPENVINO_DEVICE",
                        "RAG_ENABLE_RERANKER",
                        "RAG_ANSWER_MODEL",
                        "QDRANT_COLLECTION",
                        "NEO4J_URI",
                    ]
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Legal Graph RAG against handmade ground truth.")
    parser.add_argument("--ground-truth", default="data/eval/rag_ground_truth.jsonl")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--graph", default="data/graph/legal_graph.json")
    parser.add_argument("--index-dir", default="data/vector_db/legal_graph_rag")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expand-depth", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-recall-at-k", type=float, default=None)
    parser.add_argument("--min-ref-hit", type=float, default=None)
    parser.add_argument("--min-overall-pass", type=float, default=None)
    args = parser.parse_args()

    cases = load_jsonl(Path(args.ground_truth), args.limit)
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/eval/reports") / time.strftime("%Y%m%d_%H%M%S")

    rag = LegalGraphRAG(
        processed_path=args.processed,
        graph_path=args.graph,
        index_dir=args.index_dir,
        force_reindex=args.force_reindex,
        use_reranker=not args.no_reranker,
    )

    results = []
    try:
        from tqdm import tqdm

        progress_bar = tqdm(cases, desc="Evaluating RAG")
    except ImportError:
        progress_bar = cases
        print(f"Starting evaluation of {len(cases)} cases...")

    for idx, case in enumerate(progress_bar, start=1):
        try:
            result = evaluate_case(
                rag,
                case,
                top_k=args.top_k,
                expand_depth=args.expand_depth,
                retrieval_only=args.retrieval_only,
            )
            if not args.retrieval_only:
                time.sleep(1.0)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = {
                "id": case.get("id"),
                "query_type": case.get("query_type"),
                "question": case.get("question"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "retrieval": {
                    "recall_at_1": 0.0,
                    "recall_at_3": 0.0,
                    "recall_at_k": 0.0,
                    "mrr": 0.0,
                    "ref_hit": 0.0,
                    "quote_context_f1": 0.0,
                    "required_modality_hit": 0.0,
                },
                "answer_metrics": {
                    "answer_token_f1": 0.0,
                    "number_accuracy": 0.0,
                    "citation_accuracy": 0.0,
                    "claim_support_accuracy": 0.0,
                },
                "pass_fail": {
                    "retrieval_pass": False,
                    "modality_pass": False,
                    "answer_pass": False,
                    "overall_pass": False,
                },
                "latency_ms": 0,
            }
        results.append(result)
        if progress_bar is cases and args.progress_every and idx % args.progress_every == 0:
            print(f"Evaluated {idx}/{len(cases)}")

    summary = summarize(results)
    write_outputs(results, summary, out_dir)
    print(json.dumps({"out_dir": str(out_dir), **summary}, ensure_ascii=False, indent=2))
    failed_thresholds = []
    if args.min_recall_at_k is not None and summary["retrieval"]["recall_at_k"] < args.min_recall_at_k:
        failed_thresholds.append(f"recall_at_k<{args.min_recall_at_k}")
    if args.min_ref_hit is not None and summary["retrieval"]["ref_hit"] < args.min_ref_hit:
        failed_thresholds.append(f"ref_hit<{args.min_ref_hit}")
    if args.min_overall_pass is not None and summary["pass_rates"]["overall_pass"] < args.min_overall_pass:
        failed_thresholds.append(f"overall_pass<{args.min_overall_pass}")
    if failed_thresholds:
        raise SystemExit("RAG evaluation thresholds failed: " + ", ".join(failed_thresholds))


if __name__ == "__main__":
    main()
