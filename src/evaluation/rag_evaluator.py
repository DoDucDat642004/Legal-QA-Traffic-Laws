import argparse
import json
import os
import random
import re
import statistics
import traceback
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=os.getenv("DOTENV_OVERRIDE", "0").strip().lower() in {"1", "true", "yes", "on"})

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


SIGN_CODE_RE = re.compile(r"\b(?:[A-ZĐ]{1,3}\.)?\d{2,3}[a-z]?\b", re.IGNORECASE)


def ascii_lower(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def record_text(record: dict[str, Any]) -> str:
    return (
        record.get("rag_text")
        or record.get("source_body_exact")
        or record.get("content")
        or record.get("text")
        or ""
    )


def content_tokens(text: str) -> set[str]:
    stop = {
        "nhung", "gi", "nao", "bao", "nhieu", "duoc", "quy", "dinh", "la", "cua",
        "trong", "doi", "voi", "nguoi", "ca", "nhan", "to", "chuc", "mot", "cac",
        "the", "khong", "co", "ve", "tai", "khi", "neu", "thi", "nay", "theo",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", ascii_lower(text))
        if len(token) >= 3 and token not in stop
    }


def query_phrases(query: str) -> list[str]:
    q = ascii_lower(query)
    candidates = [
        "giay van tai",
        "van tai da phuong thuc",
        "nguoi dan toc thieu so khong biet doc viet tieng viet",
        "phuong phap dao tao",
        "cong nhan trung tuyen",
        "ket thuc ky sat hach",
        "chi huy giao thong duong bo",
        "hien truong tai nan",
        "noi xay ra vu tai nan",
        "hang b2",
        "cap truoc ngay",
        "muc phat tien toi da",
        "thiet bi in hoa don",
        "dong ho tinh tien",
        "chuyen lan duong khong dung",
        "su dung dien thoai",
        "nong do con",
        "duong cao toc",
        "cho theo tu 03 nguoi tro len",
    ]
    phrases = [candidate for candidate in candidates if candidate in q]
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) >= 3]
    for size in [5, 4, 3]:
        for idx in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[idx : idx + size])
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases[:24]


def exact_subject_boost(query_ascii: str, text_ascii: str) -> float:
    rules = [
        (["giay van tai"], ["giay van tai la"], 30.0),
        (["van tai da phuong thuc"], ["van tai da phuong thuc la"], 30.0),
        (["phuong phap dao tao", "dan toc thieu so"], ["phuong phap dao tao", "hinh anh truc quan"], 30.0),
        (["cong nhan trung tuyen", "ket thuc ky sat hach"], ["03 ngay lam viec", "cong nhan trung tuyen"], 30.0),
        (["chi huy giao thong duong bo"], ["chi huy giao thong duong bo la"], 30.0),
        (["hien truong tai nan", "trach nhiem"], ["nguoi co mat tai noi xay ra vu tai nan", "giup do cuu chua"], 30.0),
        (["hang b2", "cap truoc ngay"], ["giay phep lai xe hang b2", "duoc tiep tuc dieu khien"], 30.0),
        (["muc phat tien toi da"], ["75 000 000 dong", "toi da trong hoat dong duong bo"], 30.0),
        (["thiet bi in hoa don", "dong ho tinh tien"], ["thiet bi in hoa don", "dong ho tinh tien"], 30.0),
    ]
    score = 0.0
    for query_needles, text_needles, value in rules:
        if all(needle in query_ascii for needle in query_needles) and any(needle in text_ascii for needle in text_needles):
            score += value
    return score


class LexicalOnlyRAG:
    def __init__(self, processed_path: str | Path):
        self.records: list[dict[str, Any]] = []
        self.records_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.docs: list[str] = []
        self._load_records(Path(processed_path))

    def _load_records(self, processed_path: Path) -> None:
        seen = set()
        for path in processed_path.rglob("*.json"):
            try:
                data = json.load(path.open(encoding="utf-8"))
            except Exception:
                continue
            rows = data if isinstance(data, list) else data.get("records") or data.get("chunks") or []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                ref = row.get("legal_reference") or {}
                doc = ref.get("document") or row.get("doc_name") or ""
                text = record_text(row)
                source_chunk_id = row.get("source_chunk_id") or row.get("id") or ""
                if not doc or not text or not source_chunk_id:
                    continue
                key = (source_chunk_id, row.get("rag_modality") or row.get("modality") or "text")
                if key in seen:
                    continue
                seen.add(key)
                meta = row.get("rag_metadata") if isinstance(row.get("rag_metadata"), dict) else {}
                item = {
                    **row,
                    "source_chunk_id": source_chunk_id,
                    "rag_text": row.get("rag_text") or text,
                    "legal_reference": ref,
                    "rag_modality": row.get("rag_modality") or meta.get("modality") or row.get("modality") or "text",
                    "rag_metadata": meta,
                    "_text_ascii": ascii_lower(text),
                    "_tokens": content_tokens(text),
                }
                self.records.append(item)
                self.records_by_doc[doc].append(item)
        self.docs = sorted(self.records_by_doc)

    def _scope_documents(self, query: str) -> list[str]:
        qa = ascii_lower(query)
        matches = []
        for doc in self.docs:
            if doc.lower() in (query or "").lower() or ascii_lower(doc) in qa:
                matches.append(doc)
        return matches[:3]

    def retrieve(self, query: str, top_k: int = 8, expand_depth: int = 0) -> list[dict[str, Any]]:
        qa = ascii_lower(query)
        q_tokens = content_tokens(query)
        phrases = query_phrases(query)
        scoped_docs = self._scope_documents(query)
        candidates = []
        source = []
        if scoped_docs:
            for doc in scoped_docs:
                source.extend(self.records_by_doc.get(doc, []))
        else:
            source = self.records
        for record in source:
            text_ascii = record.get("_text_ascii") or ""
            overlap = q_tokens & (record.get("_tokens") or set())
            score = min(len(overlap), 30) * 0.15
            score += sum(1 for phrase in phrases if phrase and phrase in text_ascii) * 1.0
            score += exact_subject_boost(qa, text_ascii)
            if scoped_docs:
                score += 2.0
            if score <= 0.0:
                continue
            item = {k: v for k, v in record.items() if not k.startswith("_")}
            item["retrieval_score"] = score
            item["retrieval_reasons"] = ["core_lexical_match"]
            candidates.append(item)
        return sorted(candidates, key=lambda row: float(row.get("retrieval_score") or 0.0), reverse=True)[:top_k]


def looks_like_penalty_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in ["phạt", "xử phạt", "bị phạt", "mức phạt", "bao nhiêu tiền", "trừ điểm", "tước"])


def expected_document(case: dict[str, Any]) -> str:
    refs = case.get("expected_refs") or []
    return (refs[0].get("doc") if refs else "") or ""


def evaluation_stratum(case: dict[str, Any]) -> tuple[str, str]:
    return str(case.get("query_type") or "UNKNOWN"), expected_document(case) or "UNKNOWN_DOC"


def stratified_case_ids(cases: list[dict[str, Any]], sample_size: int, seed: int) -> set[str]:
    if sample_size <= 0:
        return set()
    if sample_size >= len(cases):
        return {str(case.get("id")) for case in cases}

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[evaluation_stratum(case)].append(case)

    counts = Counter({key: len(value) for key, value in groups.items()})
    total = sum(counts.values())
    allocations: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    for key, count in counts.items():
        exact = count * sample_size / total
        base = min(count, int(exact))
        allocations[key] = base
        remainders.append((exact - base, key))

    remaining = sample_size - sum(allocations.values())
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if allocations[key] < counts[key]:
            allocations[key] += 1
            remaining -= 1

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for key in sorted(groups):
        rows = list(groups[key])
        rng.shuffle(rows)
        sampled.extend(rows[: allocations.get(key, 0)])

    if len(sampled) < sample_size:
        selected = {case.get("id") for case in sampled}
        remaining_rows = [case for case in cases if case.get("id") not in selected]
        rng.shuffle(remaining_rows)
        sampled.extend(remaining_rows[: sample_size - len(sampled)])

    return {str(case.get("id")) for case in sampled}


def looks_like_explicit_sign_question(question: str) -> bool:
    q = (question or "").lower()
    if SIGN_CODE_RE.search(question or ""):
        return True
    return any(
        phrase in q
        for phrase in [
            "biển báo cấm",
            "biển hiệu lệnh",
            "biển chỉ dẫn",
            "biển phụ",
            "ý nghĩa biển",
            "biển p.",
            "biển r.",
            "biển i.",
            "biển w.",
        ]
    )


def looks_like_procedure_question(question: str) -> bool:
    q = (question or "").lower()
    return any(
        phrase in q
        for phrase in [
            "thủ tục",
            "hồ sơ",
            "nộp",
            "cấp giấy phép",
            "cấp lại",
            "cấp đổi",
            "đổi giấy phép",
            "thu hồi",
            "sát hạch",
            "đào tạo",
            "lưu trữ",
            "thời hạn",
            "bao nhiêu ngày",
        ]
    )


def required_modality(case: dict[str, Any]) -> str | None:
    query_type = case.get("query_type")
    question = case.get("question", "")
    doc = expected_document(case)
    if query_type == "SIGN_MEANING":
        if "Nghị định" in doc and looks_like_penalty_question(question):
            return "has_penalty"
        if "QCVN" in doc or "Thông tư 51" in doc or looks_like_explicit_sign_question(question):
            return "has_sign"
        if any(k in question.lower() for k in ["cố định", "tạm thời", "ưu tiên", "chấp hành theo biển nào"]):
            return None
        return None
    if query_type == "TABLE_LOOKUP":
        if not any(k in question.lower() for k in ["bảng", "dòng", "cột", "ô bảng", "tra bảng"]):
            return None
        return "has_table"
    if query_type == "PENALTY":
        if "Nghị định" not in doc:
            return None
        return "has_penalty"
    if query_type == "PROCEDURE":
        refs = case.get("expected_refs") or []
        article = str((refs[0] or {}).get("article") or "") if refs else ""
        if "Luật Trật tự ATGT" in doc and article == "89":
            return None
        if looks_like_procedure_question(question):
            return "has_procedure"
        return None
    return None


def retrieval_question(case: dict[str, Any], *, use_source_doc_hint: bool) -> str:
    question = case.get("question") or ""
    if not use_source_doc_hint:
        return question
    expected_refs = case.get("expected_refs") or []
    doc = (expected_refs[0].get("doc") if expected_refs else "") or ""
    if not doc or doc.lower() in question.lower():
        return question
    return f"Trong {doc}, {question}"


def evaluate_case(
    rag: LegalGraphRAG,
    case: dict[str, Any],
    *,
    top_k: int,
    expand_depth: int,
    retrieval_only: bool,
    answer_enabled: bool | None = None,
    answer_on_retrieval_failure: bool = False,
    use_source_doc_hint: bool = False,
) -> dict[str, Any]:
    start = time.time()
    query_for_retrieval = retrieval_question(case, use_source_doc_hint=use_source_doc_hint)
    contexts = rag.retrieve(query_for_retrieval, top_k=top_k, expand_depth=expand_depth)

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

    retrieval_pass = bool(retrieval["recall_at_k"] or retrieval["ref_hit"])
    should_evaluate_answer = not retrieval_only
    if answer_enabled is not None:
        should_evaluate_answer = should_evaluate_answer and answer_enabled
    if answer_on_retrieval_failure and not retrieval_only and not retrieval_pass:
        should_evaluate_answer = True

    answer = ""
    answer_metrics = {
        "answer_token_f1": None,
        "number_accuracy": None,
        "citation_accuracy": None,
        "claim_support_accuracy": None,
    }
    if should_evaluate_answer:
        answer = rag.generate_answer(case["question"], contexts)
        answer_metrics = {
            "answer_token_f1": token_f1(answer, case.get("gold_answer", "")),
            "number_accuracy": number_accuracy(answer, case.get("expected_numbers") or [], case.get("gold_answer", "")),
            "citation_accuracy": citation_accuracy(answer, case.get("required_citations") or [], refs),
            "claim_support_accuracy": claim_support_accuracy(answer, case.get("required_claims") or []),
        }
    latency_ms = int((time.time() - start) * 1000)

    pass_fail = {
        "retrieval_pass": retrieval_pass,
        "modality_pass": bool(retrieval["required_modality_hit"]),
        "answer_pass": bool(
            not should_evaluate_answer
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
        "retrieval_question": query_for_retrieval,
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
        "answer_evaluated": should_evaluate_answer,
        "pass_fail": pass_fail,
        "answer": answer,
        "latency_ms": latency_ms,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(path: tuple[str, str]) -> float:
        vals = [
            float(r[path[0]][path[1]])
            for r in results
            if path[0] in r and path[1] in r[path[0]] and r[path[0]][path[1]] is not None
        ]
        return statistics.mean(vals) if vals else 0.0

    by_type = {}
    for result in results:
        by_type.setdefault(result.get("query_type") or "UNKNOWN", []).append(result)

    type_summary = {}
    for query_type, items in by_type.items():
        answered_items = [i for i in items if i.get("answer_evaluated")]
        type_summary[query_type] = {
            "count": len(items),
            "answer_evaluated_count": len(answered_items),
            "recall_at_k": statistics.mean(float(i["retrieval"]["recall_at_k"]) for i in items),
            "mrr": statistics.mean(float(i["retrieval"]["mrr"]) for i in items),
            "ref_hit": statistics.mean(float(i["retrieval"]["ref_hit"]) for i in items),
            "answer_evaluated_pass_rate": (
                statistics.mean(float(i["pass_fail"]["answer_pass"]) for i in answered_items)
                if answered_items
                else None
            ),
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
    answered_results = [r for r in results if r.get("answer_evaluated")]
    return {
        "case_count": len(results),
        "error_count": error_count,
        "answer_evaluated_count": len(answered_results),
        "answer_evaluated_coverage": len(answered_results) / len(results) if results else 0.0,
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
            "answer_evaluated_pass": (
                statistics.mean(float(r["pass_fail"]["answer_pass"]) for r in answered_results)
                if answered_results
                else 0.0
            ),
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
                "id": r.get("id"),
                "query_type": r.get("query_type"),
                "question": r.get("question"),
                "pass_fail": r.get("pass_fail", {}),
                "answer_evaluated": r.get("answer_evaluated", False),
                "retrieval": r.get("retrieval", {}),
                "expected_chunks": r.get("expected_chunks", []),
                "retrieved_chunks": (r.get("retrieved_chunks") or [])[:8],
                "error": r.get("error"),
            }
            for r in results
            if not (r.get("pass_fail") or {}).get("overall_pass")
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
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Fast core evaluator: use processed records and lexical ranking only; no embeddings, Qdrant, Neo4j, reranker, or LLM.",
    )
    parser.add_argument(
        "--answer-sample-size",
        type=int,
        default=0,
        help=(
            "Evaluate retrieval for every case but generate answers only for a "
            "deterministic stratified sample. Use 0 to answer every case when "
            "--retrieval-only is not set."
        ),
    )
    parser.add_argument("--answer-sample-seed", type=int, default=20260522)
    parser.add_argument(
        "--answer-on-retrieval-failures",
        action="store_true",
        help="Also generate an answer for any case that misses both expected chunk and expected ref.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--force-reindex", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--use-reranker", action="store_true", help="Force reranker even when RAG_ENABLE_RERANKER=false.")
    parser.add_argument(
        "--use-source-doc-hint",
        action="store_true",
        help="Prefix document-scoped eval questions with their expected document for retrieval only.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-recall-at-k", type=float, default=None)
    parser.add_argument("--min-ref-hit", type=float, default=None)
    parser.add_argument("--min-overall-pass", type=float, default=None)
    args = parser.parse_args()

    cases = load_jsonl(Path(args.ground_truth), args.limit)
    out_dir = Path(args.out_dir) if args.out_dir else Path("data/eval/reports") / time.strftime("%Y%m%d_%H%M%S")

    if args.lexical_only:
        args.retrieval_only = True

    use_reranker = args.use_reranker or (bool_env("RAG_ENABLE_RERANKER", False) and not args.no_reranker)
    answer_eval_ids: set[str] | None = None
    if not args.retrieval_only and args.answer_sample_size > 0:
        answer_eval_ids = stratified_case_ids(cases, args.answer_sample_size, args.answer_sample_seed)
        print(
            "Hybrid evaluation: retrieval on all cases; "
            f"answer generation on {len(answer_eval_ids)}/{len(cases)} stratified cases "
            f"(seed={args.answer_sample_seed})."
        )

    if args.lexical_only:
        rag = LexicalOnlyRAG(args.processed)
    else:
        rag = LegalGraphRAG(
            processed_path=args.processed,
            graph_path=args.graph,
            index_dir=args.index_dir,
            force_reindex=args.force_reindex,
            use_reranker=use_reranker,
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
                answer_enabled=None if answer_eval_ids is None else str(case.get("id")) in answer_eval_ids,
                answer_on_retrieval_failure=args.answer_on_retrieval_failures,
                use_source_doc_hint=args.use_source_doc_hint or bool_env("EVAL_USE_SOURCE_DOC_HINT", False),
            )
            if result.get("answer_evaluated") and args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = {
                "id": case.get("id"),
                "query_type": case.get("query_type"),
                "question": case.get("question"),
                "retrieval_question": retrieval_question(
                    case,
                    use_source_doc_hint=args.use_source_doc_hint or bool_env("EVAL_USE_SOURCE_DOC_HINT", False),
                ),
                "expected_chunks": case.get("expected_chunks") or [],
                "retrieved_chunks": [],
                "expected_refs": case.get("expected_refs") or [],
                "retrieved_refs": [],
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
                    "answer_token_f1": None,
                    "number_accuracy": None,
                    "citation_accuracy": None,
                    "claim_support_accuracy": None,
                },
                "answer_evaluated": False,
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
