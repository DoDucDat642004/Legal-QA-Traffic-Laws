import re
import unicodedata
from typing import Any


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str) -> str:
    return re.sub(r"[^\w]+", "", normalize_text(text))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = {}
    gold_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    common = sum(min(pred_counts.get(t, 0), gold_counts.get(t, 0)) for t in gold_counts)
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_fuzzy(container: str, needle: str, *, min_ratio: float = 0.82) -> bool:
    container_norm = compact_text(container)
    needle_norm = compact_text(needle)
    if not needle_norm:
        return True
    if needle_norm in container_norm:
        return True
    return token_f1(container, needle) >= min_ratio


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    expected = set(x for x in expected_ids if x)
    if not expected:
        return 1.0
    return 1.0 if expected & set(retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    expected = set(x for x in expected_ids if x)
    if not expected:
        return 1.0
    for idx, item in enumerate(retrieved_ids, start=1):
        if item in expected:
            return 1.0 / idx
    return 0.0


def reference_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    if expected.get("doc"):
        actual_doc = (actual.get("document") or actual.get("doc") or "").replace("_", " ")
        if normalize_text(expected["doc"]) not in normalize_text(actual_doc):
            return False
    for key in ["article", "clause", "point"]:
        if expected.get(key) and str(actual.get(key) or "").lower() != str(expected[key]).lower():
            return False
    return True


def source_chunk_ref(source_chunk_id: str) -> dict[str, Any]:
    parts = (source_chunk_id or "").rsplit("_", 4)
    if len(parts) != 5:
        return {}
    doc, article, clause, point, _suffix = parts
    return {
        "doc": doc.replace("_", " "),
        "article": article if article != "0" else "",
        "clause": clause if clause != "0" else "",
        "point": point if point != "0" else "",
    }


def context_has_ref(contexts: list[dict[str, Any]], expected_refs: list[dict[str, Any]]) -> bool:
    if not expected_refs:
        return True
    for expected in expected_refs:
        for ctx in contexts:
            source_ref = source_chunk_ref(ctx.get("source_chunk_id") or ctx.get("id") or "")
            if source_ref and reference_matches(source_ref, expected):
                return True
            ref = ctx.get("legal_reference") or {}
            if reference_matches(ref, expected):
                return True
            meta = ctx.get("rag_metadata") or {}
            if reference_matches(
                {
                    "doc": meta.get("doc"),
                    "article": meta.get("article"),
                    "clause": meta.get("clause"),
                    "point": meta.get("point"),
                },
                expected,
            ):
                return True
    return False


def extract_answer_numbers(text: str) -> set[str]:
    values = set()
    for match in re.finditer(r"(\d{1,3}(?:\.\d{3})+|\d+)\s*(?:đồng|vnd|vnđ)", text or "", re.IGNORECASE):
        values.add(str(int(match.group(1).replace(".", ""))))
    for match in re.finditer(r"\b\d+(?:[.,]\d+)?\b", text or ""):
        values.add(match.group(0).replace(",", "."))
    return values


def number_accuracy(answer: str, expected_numbers: list[dict[str, Any]]) -> float:
    if not expected_numbers:
        return 1.0
    actual = extract_answer_numbers(answer)
    expected = {str(item.get("value")) for item in expected_numbers if item.get("value") is not None}
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def citation_accuracy(answer: str, required_citations: list[str], expected_refs: list[dict[str, Any]]) -> float:
    if not required_citations and not expected_refs:
        return 1.0
    answer_norm = normalize_text(answer)
    hits = 0
    total = 0
    for citation in required_citations:
        if not citation:
            continue
        total += 1
        if contains_fuzzy(answer, citation, min_ratio=0.65):
            hits += 1
    for ref in expected_refs:
        parts = []
        if ref.get("point"):
            parts.append(f"điểm {ref['point']}")
        if ref.get("clause"):
            parts.append(f"khoản {ref['clause']}")
        if ref.get("article"):
            parts.append(f"điều {ref['article']}")
        ref_text = " ".join(parts)
        if ref_text:
            total += 1
            if normalize_text(ref_text) in answer_norm:
                hits += 1
    return hits / total if total else 1.0


def claim_support_accuracy(answer: str, claims: list[dict[str, Any]]) -> float:
    if not claims:
        return 1.0
    hits = 0
    total = 0
    for claim in claims:
        text = claim.get("claim") or ""
        if not text:
            continue
        total += 1
        if contains_fuzzy(answer, text, min_ratio=0.55):
            hits += 1
    return hits / total if total else 1.0


def modality_flags(contexts: list[dict[str, Any]]) -> dict[str, bool]:
    flags = {
        "has_table": False,
        "has_figure": False,
        "has_sign": False,
        "has_penalty": False,
        "has_procedure": False,
    }
    for ctx in contexts:
        meta = ctx.get("rag_metadata") or {}
        modality = ctx.get("rag_modality")
        flags["has_table"] = flags["has_table"] or bool(meta.get("has_table") or modality == "table" or ctx.get("tables"))
        flags["has_figure"] = flags["has_figure"] or bool(meta.get("has_figure") or modality == "figure" or ctx.get("figures") or ctx.get("image_path"))
        flags["has_sign"] = flags["has_sign"] or bool(meta.get("has_sign") or modality == "sign" or meta.get("sign_codes"))
        flags["has_penalty"] = flags["has_penalty"] or bool(meta.get("has_penalty") or ctx.get("penalties"))
        flags["has_procedure"] = flags["has_procedure"] or bool(meta.get("has_procedure"))
    return flags
