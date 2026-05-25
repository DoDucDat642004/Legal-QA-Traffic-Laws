import re
import unicodedata
from typing import Any


DOC_ALIAS_RULES = [
    (
        "nd168",
        [
            "nghi dinh 168",
            "nghi dinh so 168",
            "168/2024",
            "168 2024",
            "168-2024",
        ],
    ),
    (
        "nd336",
        [
            "nghi dinh 336",
            "nghi dinh so 336",
            "336/2025",
            "336 2025",
            "336-2025",
        ],
    ),
    (
        "luat35",
        [
            "luat duong bo",
            "luat duong bo so 35",
            "35/2024/qh15",
            "35 2024 qh15",
            "35-2024-qh15",
            "luat 35",
        ],
    ),
    (
        "luat36",
        [
            "luat trat tu atgt",
            "luat trat tu an toan giao thong",
            "trat tu an toan giao thong duong bo",
            "ttatgt",
            "36/2024/qh15",
            "36 2024 qh15",
            "36-2024-qh15",
            "luat 36",
        ],
    ),
    (
        "qcvn41",
        [
            "qcvn 41",
            "qcvn 41:2024",
            "thong tu 51",
            "51/2024",
            "51 2024",
            "51-2024",
            "quy chuan ve bao hieu duong bo",
            "quy chuan bbdb",
        ],
    ),
    (
        "tt35",
        [
            "thong tu 35",
            "35/2024/tt-bgtvt",
            "35 2024 tt bgtvt",
            "35-2024-tt-bgtvt",
            "tt35",
        ],
    ),
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ascii_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def normalize_doc_text(value: str) -> str:
    text = ascii_lower(value or "").replace("_", " ")
    text = re.sub(r"[/():.,]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def doc_alias_key(value: str) -> str:
    text = normalize_doc_text(value)
    if not text:
        return ""
    for key, aliases in DOC_ALIAS_RULES:
        if any(alias in text for alias in aliases):
            return key
    return text


def doc_matches(actual_doc: str, expected_doc: str) -> bool:
    actual_norm = normalize_doc_text(actual_doc)
    expected_norm = normalize_doc_text(expected_doc)
    if not expected_norm:
        return True
    if not actual_norm:
        return False
    actual_key = doc_alias_key(actual_norm)
    expected_key = doc_alias_key(expected_norm)
    if actual_key and expected_key and actual_key == expected_key:
        return True
    return expected_norm in actual_norm or actual_norm in expected_norm


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


def token_recall(container: str, needle: str) -> float:
    container_tokens = normalize_text(container).split()
    needle_tokens = normalize_text(needle).split()
    if not needle_tokens:
        return 1.0
    if not container_tokens:
        return 0.0
    container_counts = {}
    needle_counts = {}
    for token in container_tokens:
        container_counts[token] = container_counts.get(token, 0) + 1
    for token in needle_tokens:
        needle_counts[token] = needle_counts.get(token, 0) + 1
    common = sum(min(container_counts.get(t, 0), needle_counts.get(t, 0)) for t in needle_counts)
    return common / len(needle_tokens)


def contains_fuzzy(container: str, needle: str, *, min_ratio: float = 0.82) -> bool:
    container_norm = compact_text(container)
    needle_norm = compact_text(needle)
    if not needle_norm:
        return True
    if needle_norm in container_norm:
        return True
    return token_f1(container, needle) >= min_ratio or token_recall(container, needle) >= min_ratio


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
        if not doc_matches(actual_doc, str(expected["doc"])):
            return False
    if not ref_part_matches(actual.get("article"), expected.get("article")):
        return False
    if not clause_matches(actual.get("article"), actual.get("clause"), expected.get("article"), expected.get("clause")):
        return False
    if not ref_part_matches(actual.get("point"), expected.get("point")):
        return False
    if expected.get("section") and not ref_part_matches(
        actual.get("section") or actual.get("appendix") or actual.get("article"),
        expected.get("section"),
    ):
        # QCVN appendix references are often represented as article/appendix in
        # extracted records. Keep section matching best-effort so ordinary
        # article/clause refs remain strict.
        if expected.get("article") or expected.get("clause") or expected.get("point"):
            return False
    return True


def normalize_ref_part(value: Any) -> str:
    text = ascii_lower(str(value or "")).replace("_", " ")
    text = re.sub(r"\b(điều|dieu|khoản|khoan|điểm|diem)\b", " ", text)
    text = re.sub(r"[^a-z0-9.,;/\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def ref_part_values(value: Any) -> set[str]:
    text = normalize_ref_part(value)
    if not text:
        return set()
    values = {text}
    values.update(part for part in re.split(r"[,;/]|\s+va\s+|\s+hoac\s+", text) if part)
    for match in re.finditer(r"\b\d+(?:\.\d+)?\b|[a-zđ]", text, flags=re.IGNORECASE):
        values.add(match.group(0).lower())
    return values


def ref_part_matches(actual: Any, expected: Any) -> bool:
    expected_norm = normalize_ref_part(expected)
    if not expected_norm:
        return True
    actual_norm = normalize_ref_part(actual)
    if not actual_norm:
        return False
    if actual_norm == expected_norm:
        return True
    return expected_norm in ref_part_values(actual_norm)


def clause_matches(actual_article: Any, actual_clause: Any, expected_article: Any, expected_clause: Any) -> bool:
    expected_norm = normalize_ref_part(expected_clause)
    if not expected_norm:
        return True
    if ref_part_matches(actual_clause, expected_clause):
        return True
    # QCVN sections are commonly encoded either as article=4, clause=1 or
    # article=4.1. Accept both for expected clause 4.1 under Article 4.
    if re.fullmatch(r"\d+\.\d+", expected_norm):
        left, right = expected_norm.split(".", 1)
        if normalize_ref_part(actual_article) == expected_norm:
            return True
        if ref_part_matches(actual_article, left) and ref_part_matches(actual_clause, right):
            return True
        if ref_part_matches(expected_article, left) and ref_part_matches(actual_article, expected_norm):
            return True
    return False


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


def context_matches_ref(ctx: dict[str, Any], expected: dict[str, Any]) -> bool:
    source_ref = source_chunk_ref(ctx.get("source_chunk_id") or ctx.get("id") or "")
    if source_ref and reference_matches(source_ref, expected):
        return True
    ref = ctx.get("legal_reference") or {}
    if reference_matches(ref, expected):
        return True
    meta = ctx.get("rag_metadata") or {}
    return reference_matches(
        {
            "doc": meta.get("doc"),
            "article": meta.get("article"),
            "clause": meta.get("clause"),
            "point": meta.get("point"),
            "section": meta.get("section") or meta.get("appendix"),
        },
        expected,
    )


def chunk_matches_context(ctx: dict[str, Any], expected_id: str) -> bool:
    actual_id = ctx.get("source_chunk_id") or ctx.get("id") or ""
    if expected_id and actual_id == expected_id:
        return True
    expected_ref = source_chunk_ref(expected_id)
    return bool(expected_ref and context_matches_ref(ctx, expected_ref))


def recall_at_k_contexts(contexts: list[dict[str, Any]], expected_ids: list[str], k: int) -> float:
    expected = [x for x in expected_ids if x]
    if not expected:
        return 1.0
    for ctx in contexts[:k]:
        if any(chunk_matches_context(ctx, expected_id) for expected_id in expected):
            return 1.0
    return 0.0


def reciprocal_rank_contexts(contexts: list[dict[str, Any]], expected_ids: list[str]) -> float:
    expected = [x for x in expected_ids if x]
    if not expected:
        return 1.0
    for idx, ctx in enumerate(contexts, start=1):
        if any(chunk_matches_context(ctx, expected_id) for expected_id in expected):
            return 1.0 / idx
    return 0.0


def context_has_ref(contexts: list[dict[str, Any]], expected_refs: list[dict[str, Any]]) -> bool:
    if not expected_refs:
        return True
    for expected in expected_refs:
        for ctx in contexts:
            if context_matches_ref(ctx, expected):
                return True
    return False


def extract_answer_numbers(text: str) -> set[str]:
    values = set()
    for match in re.finditer(r"(\d{1,3}(?:\.\d{3})+|\d+)\s*(?:đồng|vnd|vnđ)", text or "", re.IGNORECASE):
        values.add(str(int(match.group(1).replace(".", ""))))
    for match in re.finditer(r"\b\d+(?:[.,]\d+)*\b", text or ""):
        raw = match.group(0)
        normalized = raw.replace(",", ".")
        values.add(normalized)
        if re.fullmatch(r"\d+", raw):
            values.add(str(int(raw)))
        elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", raw):
            values.add(str(int(raw.replace(".", ""))))
    return values


def number_accuracy(answer: str, expected_numbers: list[dict[str, Any]], gold_answer: str = "") -> float:
    if not expected_numbers:
        return 1.0
    actual = extract_answer_numbers(answer)
    expected = {str(item.get("value")) for item in expected_numbers if item.get("value") is not None}
    if gold_answer:
        gold_expected = expected & extract_answer_numbers(gold_answer)
        if gold_expected:
            expected = gold_expected
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def legal_ref_variants(ref: dict[str, Any]) -> list[str]:
    detail_parts = []
    if ref.get("point"):
        detail_parts.append(f"điểm {ref['point']}")
    if ref.get("clause"):
        detail_parts.append(f"khoản {ref['clause']}")
    if ref.get("article"):
        detail_parts.append(f"điều {ref['article']}")

    detail = " ".join(detail_parts).strip()
    doc = (ref.get("doc") or "").strip()
    raw = (ref.get("raw") or "").strip()

    variants = []
    if raw:
        variants.append(raw)
    if detail and doc:
        variants.append(f"{detail} {doc}")
    if detail:
        variants.append(detail)
    if ref.get("clause") and ref.get("article") and not ref.get("point"):
        variants.append(f"khoản {ref['clause']} điều {ref['article']}")
    if ref.get("article") and doc and not ref.get("clause") and not ref.get("point"):
        variants.append(f"điều {ref['article']} {doc}")

    return list(dict.fromkeys(v for v in variants if v))


def ref_in_answer(answer: str, ref: dict[str, Any]) -> bool:
    answer_norm = normalize_text(answer)
    for variant in legal_ref_variants(ref):
        variant_norm = normalize_text(variant)
        if variant_norm and variant_norm in answer_norm:
            return True
        if contains_fuzzy(answer, variant, min_ratio=0.90):
            return True
    return False


def citation_accuracy(answer: str, required_citations: list[str], expected_refs: list[dict[str, Any]]) -> float:
    if not required_citations and not expected_refs:
        return 1.0

    # Ground truth usually stores both a citation string and the same structured
    # reference. Score the citation obligation once, while accepting either form.
    if required_citations:
        hits = 0
        total = 0
        for idx, citation in enumerate(required_citations):
            if not citation:
                continue
            total += 1
            ref = expected_refs[idx] if idx < len(expected_refs) else None
            citation_hit = False if ref else contains_fuzzy(answer, citation, min_ratio=0.82)
            structured_ref_hit = ref_in_answer(answer, ref) if ref else False
            if citation_hit or structured_ref_hit:
                hits += 1
        return hits / total if total else 1.0

    hits = 0
    total = 0
    for ref in expected_refs:
        if legal_ref_variants(ref):
            total += 1
            if ref_in_answer(answer, ref):
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
        support_quote = claim.get("support_quote") or ""
        if contains_fuzzy(answer, text, min_ratio=0.55) or (
            support_quote and contains_fuzzy(answer, support_quote, min_ratio=0.45)
        ):
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
        text = normalize_text(ctx.get("rag_text") or ctx.get("source_body_exact") or ctx.get("content") or "")
        flags["has_table"] = flags["has_table"] or bool(meta.get("has_table") or modality == "table" or ctx.get("tables"))
        flags["has_figure"] = flags["has_figure"] or bool(meta.get("has_figure") or modality == "figure" or ctx.get("figures") or ctx.get("image_path"))
        flags["has_sign"] = flags["has_sign"] or bool(meta.get("has_sign") or modality == "sign" or meta.get("sign_codes"))
        flags["has_penalty"] = flags["has_penalty"] or bool(
            meta.get("has_penalty")
            or ctx.get("penalties")
            or any(phrase in text for phrase in ["phạt tiền", "trừ điểm", "tước quyền", "mức phạt"])
        )
        flags["has_procedure"] = flags["has_procedure"] or bool(
            meta.get("has_procedure")
            or any(
                phrase in text
                for phrase in [
                    "thủ tục",
                    "hồ sơ",
                    "thời hạn",
                    "cấp đổi",
                    "cấp lại",
                    "thu hồi",
                    "sát hạch",
                    "nộp",
                    "ra quyết định",
                ]
            )
        )
    return flags
