import argparse
import difflib
import json
import re
from pathlib import Path
from typing import Any


HANDMADE_TO_PROCESSED = {
    "168-nd-cp.signed.json": "168-nd-cp.signed.pdf.extracted.json",
    "336nd.signed.json": "336-2025-nd-cp-22122025-signed-17665482569851736009102.pdf.extracted.json",
    "35-2024-qh15.json": "35-2024-qh15.pdf.extracted.json",
    "35-bgtvt.json": "35-bgtvt.pdf.extracted.json",
    "36-2024-qh15.json": "36-2024-qh15.pdf.extracted.json",
    "36-2024-qh15_tiep.json": "36-2024-qh15_tiep.pdf.extracted.json",
    "51-bgtvt-kem.json": "51-bgtvt-kem.pdf.extracted.json",
}

VND_RE = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d+)\s*(?:đồng|vnd|vnđ)", re.IGNORECASE)
GENERIC_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
SIGN_CODE_RE = re.compile(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def compact(text: str) -> str:
    return re.sub(r"[^\w]+", "", (text or "").lower())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_text(record: dict) -> str:
    return (
        record.get("source_body_exact")
        or record.get("original_text")
        or record.get("content")
        or record.get("violation_content")
        or record.get("meaning_and_usage")
        or ""
    )


def parse_legal_basis(legal_basis: str, doc_name: str) -> dict[str, str]:
    text = legal_basis or ""
    point = ""
    clause = ""
    article = ""
    section = ""

    point_match = re.search(r"điểm\s+([a-zđ])\b", text, re.IGNORECASE)
    clause_match = re.search(r"khoản\s+(\d+[a-z]?)\b", text, re.IGNORECASE)
    article_match = re.search(r"điều\s+(\d+[a-z]?)\b", text, re.IGNORECASE)
    section_match = re.search(r"mục\s+(\d+(?:\.\d+)*)\b", text, re.IGNORECASE)

    if point_match:
        point = point_match.group(1).lower()
    if clause_match:
        clause = clause_match.group(1)
    if article_match:
        article = article_match.group(1).upper()
    elif section_match:
        # QCVN uses numbered sections as legal article-like anchors in extracted data.
        article = section_match.group(1)
        section = section_match.group(1)

    return {
        "doc": doc_name,
        "article": article,
        "clause": clause,
        "point": point,
        "section": section,
        "raw": legal_basis,
    }


def classify_query_type(item: dict) -> str:
    question = str(item.get("question", "")).lower()
    blob = " ".join(str(item.get(k, "")) for k in ["question", "answer", "legal_basis", "reference_text"]).lower()
    if any(k in question for k in ["phạm vi điều chỉnh", "là gì", "bao gồm những gì", "được hiểu như thế nào"]):
        return "DEFINITION"
    if "biển báo" in blob or SIGN_CODE_RE.search(blob):
        return "SIGN_MEANING"
    if any(k in blob for k in ["phạt", "tiền", "trừ điểm", "tước quyền", "đình chỉ"]):
        return "PENALTY"
    if any(k in blob for k in ["hồ sơ", "thủ tục", "trình tự", "thời hạn", "nộp", "đăng ký", "cấp lại", "đổi giấy phép"]):
        return "PROCEDURE"
    if any(k in blob for k in ["trừ trường hợp", "ngoại lệ", "không áp dụng", "trừ hành vi"]):
        return "EXCEPTION"
    if "|" in item.get("reference_text", "") or "bảng" in blob:
        return "TABLE_LOOKUP"
    return "DEFINITION"


def extract_expected_numbers(item: dict) -> list[dict[str, Any]]:
    blob = " ".join(str(item.get(k, "")) for k in ["answer", "reference_text"])
    numbers = []
    seen = set()
    for match in VND_RE.finditer(blob):
        value = int(match.group(1).replace(".", ""))
        if ("VND", value) not in seen:
            numbers.append({"value": value, "unit": "VND", "type": "amount"})
            seen.add(("VND", value))

    query_type = classify_query_type(item)
    if query_type not in {"TABLE_LOOKUP", "SIGN_MEANING", "PROCEDURE"}:
        return numbers

    # Keep non-money numbers only for technical/table/sign/procedure cases.
    for match in GENERIC_NUMBER_RE.finditer(blob):
        raw = match.group(0)
        if "." in raw and re.match(r"\d{1,3}(?:\.\d{3})+", raw):
            continue
        value = raw.replace(",", ".")
        key = ("NUMBER", value)
        if key in seen:
            continue
        if len(numbers) > 20:
            break
        numbers.append({"value": value, "unit": "", "type": "number"})
        seen.add(key)
    return numbers


def score_record(record: dict, expected_ref: dict, reference_text: str) -> tuple[float, str]:
    ref = record.get("legal_reference") or {}
    score = 0.0
    reasons = []
    if expected_ref.get("article") and str(ref.get("article") or "").upper() == expected_ref["article"].upper():
        score += 2.0
        reasons.append("article")
    if expected_ref.get("clause") and str(ref.get("clause") or "") == expected_ref["clause"]:
        score += 1.0
        reasons.append("clause")
    if expected_ref.get("point") and str(ref.get("point") or "").lower() == expected_ref["point"].lower():
        score += 1.0
        reasons.append("point")

    record_text = clean_text(source_text(record))
    quote = clean_text(reference_text)
    if quote:
        compact_quote = compact(quote)
        compact_record = compact(record_text)
        if compact_quote and compact_quote in compact_record:
            score += 4.0
            reasons.append("quote_exact")
        else:
            ratio = difflib.SequenceMatcher(None, compact_quote[:1200], compact_record[:2000]).ratio()
            if ratio > 0.55:
                score += ratio * 2.0
                reasons.append(f"quote_fuzzy:{ratio:.2f}")
    return score, ",".join(reasons)


def match_expected_chunks(records: list[dict], expected_ref: dict, reference_text: str) -> list[dict[str, Any]]:
    candidates = []
    for record in records:
        ref = record.get("legal_reference") or {}
        article_ok = not expected_ref.get("article") or str(ref.get("article") or "").upper() == expected_ref["article"].upper()
        clause_ok = not expected_ref.get("clause") or str(ref.get("clause") or "") == expected_ref["clause"]
        point_ok = not expected_ref.get("point") or str(ref.get("point") or "").lower() == expected_ref["point"].lower()
        if article_ok and clause_ok and point_ok:
            candidates.append(record)

    if not candidates and expected_ref.get("article"):
        for record in records:
            ref = record.get("legal_reference") or {}
            if str(ref.get("article") or "").upper() == expected_ref["article"].upper():
                candidates.append(record)

    if not candidates:
        quote_terms = set(re.findall(r"\w{4,}", (reference_text or "").lower()))
        for record in records:
            record_terms = set(re.findall(r"\w{4,}", source_text(record).lower()))
            if len(quote_terms & record_terms) >= min(5, max(1, len(quote_terms) // 4)):
                candidates.append(record)

    candidates = candidates or records
    scored = []
    for record in candidates:
        score, reason = score_record(record, expected_ref, reference_text)
        if score <= 0:
            continue
        scored.append((score, reason, record))
    scored.sort(key=lambda item: item[0], reverse=True)

    matches = []
    seen = set()
    for score, reason, record in scored[:8]:
        chunk_id = record.get("source_chunk_id")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append({
            "source_chunk_id": chunk_id,
            "record_id": record.get("id"),
            "score": round(score, 3),
            "match_reason": reason,
            "legal_reference": record.get("legal_reference"),
        })
        if len(matches) >= 3:
            break
    return matches


def convert_case(item: dict, index: int, handmade_file: Path, records: list[dict], doc_name: str) -> dict:
    expected_ref = parse_legal_basis(item.get("legal_basis", ""), doc_name)
    matches = match_expected_chunks(records, expected_ref, item.get("reference_text", ""))
    expected_chunks = [m["source_chunk_id"] for m in matches]
    query_type = classify_query_type(item)
    expected_numbers = extract_expected_numbers(item)
    review_required = not matches or (matches[0]["score"] < 3.0)

    return {
        "id": f"{handmade_file.stem}_{index:04d}",
        "source": {
            "type": "handmade",
            "file": str(handmade_file),
            "index": index,
        },
        "question": item.get("question", ""),
        "gold_answer": item.get("answer", ""),
        "query_type": query_type,
        "expected_refs": [expected_ref],
        "expected_chunks": expected_chunks,
        "acceptable_chunks": expected_chunks[1:],
        "reference_quote": item.get("reference_text", ""),
        "expected_numbers": expected_numbers,
        "required_claims": [
            {
                "claim": item.get("answer", ""),
                "support_refs": [expected_ref],
                "support_quote": item.get("reference_text", ""),
            }
        ],
        "required_citations": [item.get("legal_basis", "")] if item.get("legal_basis") else [],
        "retrieval_expectation": {
            "top_k": 8,
            "must_retrieve_any_expected_chunk": bool(expected_chunks),
            "must_include_ref": bool(expected_ref.get("article") or expected_ref.get("section")),
        },
        "answer_expectation": {
            "must_be_faithful_to_context": True,
            "must_include_required_claims": True,
            "must_include_numbers": bool(expected_numbers),
            "must_cite_legal_basis": bool(item.get("legal_basis")),
        },
        "match_debug": {
            "review_required": review_required,
            "matches": matches,
        },
    }


def convert_all(handmade_dir: Path, processed_dir: Path, out_path: Path, review_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    all_cases = []
    review_cases = []

    for handmade_file in sorted(handmade_dir.glob("*.json")):
        processed_name = HANDMADE_TO_PROCESSED.get(handmade_file.name)
        if not processed_name:
            continue
        processed_file = processed_dir / processed_name
        if not processed_file.exists():
            continue
        records = load_json(processed_file)
        doc_name = ""
        for record in records:
            ref = record.get("legal_reference") or {}
            doc_name = record.get("doc_name") or ref.get("document") or doc_name
            if doc_name:
                break

        handmade_items = load_json(handmade_file)
        for i, item in enumerate(handmade_items):
            case = convert_case(item, i, handmade_file, records, doc_name)
            all_cases.append(case)
            if case["match_debug"]["review_required"]:
                review_cases.append(case)

    with out_path.open("w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    with review_path.open("w", encoding="utf-8") as f:
        json.dump(review_cases, f, ensure_ascii=False, indent=2)

    return {
        "cases": len(all_cases),
        "review_required": len(review_cases),
        "output": str(out_path),
        "review_output": str(review_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert handmade QA files into RAG ground-truth JSONL.")
    parser.add_argument("--handmade-dir", default="data/processed/handmade")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out", default="data/eval/rag_ground_truth.jsonl")
    parser.add_argument("--review-out", default="data/eval/rag_ground_truth_review.json")
    args = parser.parse_args()

    summary = convert_all(
        Path(args.handmade_dir),
        Path(args.processed_dir),
        Path(args.out),
        Path(args.review_out),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
