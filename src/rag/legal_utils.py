import re
from typing import Any


SIGN_CODE_RE = re.compile(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", re.IGNORECASE)
LEGAL_REF_RE = re.compile(
    r"(?:(?:điểm)\s+(?P<point>[a-zđ])\s+)?"
    r"(?:(?:khoản)\s+(?P<clause>\d+)\s+)?"
    r"(?:điều)\s+(?P<article>\d+[a-z]?)",
    re.IGNORECASE,
)


def normalize_sign_code(value: str) -> str:
    return re.sub(r"\s+", "", value or "").replace(".", "").upper()


def sign_group_from_code(code: str) -> str:
    normalized = normalize_sign_code(code)
    if normalized.startswith("P"):
        return "prohibition"
    if normalized.startswith("W"):
        return "warning"
    if normalized.startswith("R"):
        return "mandatory"
    if normalized.startswith("I"):
        return "guide"
    if normalized.startswith("S"):
        return "supplementary"
    if normalized.startswith(("DP", "IE", "E")):
        return "highway"
    return "unknown"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", (text or "").lower())


def format_reference(record: dict[str, Any]) -> str:
    ref = normalized_legal_reference(record)
    parts = []
    if ref.get("point"):
        parts.append(f"Điểm {ref.get('point')}")
    if ref.get("clause"):
        parts.append(f"Khoản {ref.get('clause')}")
    if ref.get("article"):
        parts.append(f"Điều {ref.get('article')}")
    if ref.get("section"):
        parts.append(f"Mục {ref.get('section')}")
    if ref.get("chapter"):
        parts.append(f"Chương {ref.get('chapter')}")
    doc_name = record.get("doc_name") or ref.get("document") or "Văn bản pháp luật"
    parts.append(doc_name)
    return ", ".join(str(p) for p in parts if p)


def ref_key(document: str, article: str = "", clause: str = "", point: str = "") -> str:
    parts = [document or ""]
    if article:
        parts.append(f"D{article}")
    if clause:
        parts.append(f"K{clause}")
    if point:
        parts.append(f"P{point}")
    return "|".join(parts)


def record_ref_key(record: dict[str, Any]) -> str:
    ref = record.get("legal_reference") or {}
    return ref_key(
        ref.get("document") or record.get("doc_name") or "",
        str(ref.get("article") or ""),
        str(ref.get("clause") or ""),
        str(ref.get("point") or ""),
    )


def source_text(record: dict[str, Any]) -> str:
    return (
        record.get("rag_text")
        or record.get("source_body_exact")
        or record.get("violation_content")
        or record.get("original_text")
        or record.get("meaning_and_usage")
        or record.get("content")
        or ""
    )


def source_chunk_ref(source_chunk_id: str) -> dict[str, str]:
    parts = (source_chunk_id or "").rsplit("_", 4)
    if len(parts) != 5:
        return {}
    doc, article, clause, point, _suffix = parts
    if not re.fullmatch(r"\d+[a-z]?", article or "", re.IGNORECASE):
        return {}
    if clause and clause != "0" and not re.fullmatch(r"\d+", clause):
        return {}
    if point and point != "0" and not re.fullmatch(r"[a-zđ]", point, re.IGNORECASE):
        return {}
    return {
        "doc": doc.replace("_", " "),
        "article": article if article != "0" else "",
        "clause": clause if clause != "0" else "",
        "point": point if point != "0" else "",
    }


def normalized_legal_reference(record: dict[str, Any]) -> dict[str, Any]:
    ref = dict(record.get("legal_reference") or {})
    source_ref = source_chunk_ref(record.get("source_chunk_id") or record.get("id") or "")
    if not source_ref:
        return ref
    if source_ref.get("article"):
        ref["article"] = source_ref["article"]
    if source_ref.get("clause") or str(ref.get("clause") or "").isalpha():
        ref["clause"] = source_ref.get("clause", "")
    if source_ref.get("point") or str(ref.get("point") or "") == "0":
        ref["point"] = source_ref.get("point", "")
    return ref


def record_text_for_index(record: dict[str, Any]) -> str:
    table_text = "\n".join(
        table.get("text", "")
        for table in record.get("tables", []) or []
        if isinstance(table, dict) and table.get("text")
    )
    figure_text = "\n".join(
        " ".join(str(x) for x in [fig.get("code"), fig.get("name"), fig.get("caption")] if x)
        for fig in record.get("figures", []) or []
        if isinstance(fig, dict)
    )
    parent_text = "\n".join(
        f"{item.get('kind', '')} {item.get('num', '')}: {item.get('title', '')}".strip()
        for item in record.get("parent_hierarchy", []) or []
        if isinstance(item, dict)
    )
    return "\n".join(
        x
        for x in [
            format_reference(record),
            parent_text,
            record.get("semantic_context") or "",
            source_text(record),
            table_text,
            figure_text,
        ]
        if x
    ).strip()


def penalty_summary(record: dict[str, Any]) -> dict[str, Any]:
    penalties = record.get("penalties") or {}
    main = penalties.get("main_penalty") or {}
    return {
        "penalty_type": main.get("type"),
        "fine_min_vnd": main.get("min_amount_vnd") or main.get("individual_min_vnd"),
        "fine_max_vnd": main.get("max_amount_vnd") or main.get("individual_max_vnd"),
        "individual_min_vnd": main.get("individual_min_vnd"),
        "individual_max_vnd": main.get("individual_max_vnd"),
        "organization_min_vnd": main.get("organization_min_vnd"),
        "organization_max_vnd": main.get("organization_max_vnd"),
        "point_deduction": penalties.get("point_deduction"),
        "license_suspension": penalties.get("license_suspension"),
        "vehicle_impoundment": penalties.get("vehicle_impoundment"),
        "additional_penalties": penalties.get("additional_penalties") or [],
        "remedial_measures": penalties.get("remedial_measures") or [],
        "raw_penalty_text": main.get("raw_penalty_text") or main.get("description"),
    }


def looks_like_procedure(record: dict[str, Any]) -> bool:
    text = source_text(record).lower()
    metadata = record.get("metadata") or {}
    rule_type = str(metadata.get("rule_type") or metadata.get("domain") or "").lower()
    keywords = [
        "thủ tục",
        "hồ sơ",
        "trình tự",
        "thời hạn",
        "nộp",
        "đăng ký",
        "tiếp nhận hồ sơ",
        "trả kết quả",
        "cơ quan tiếp nhận",
        "dịch vụ bưu chính",
        "môi trường điện tử",
    ]
    return any(k in text for k in keywords) or any(k in rule_type for k in ["thủ tục", "hồ sơ", "quy trình"])


def build_vector_metadata(record: dict[str, Any], modality: str = "text", image_path: str = "") -> dict[str, Any]:
    ref = normalized_legal_reference(record)
    tables = [t for t in record.get("tables", []) or [] if isinstance(t, dict)]
    figures = [f for f in record.get("figures", []) or [] if isinstance(f, dict)]
    figure_obj = record.get("figure") if isinstance(record.get("figure"), dict) else None
    table_obj = record.get("table") if isinstance(record.get("table"), dict) else None
    if figure_obj:
        figures = figures + [figure_obj]
    if table_obj:
        tables = tables + [table_obj]
    sign_codes = sorted({
        normalize_sign_code(x)
        for fig in figures
        for x in [fig.get("code")]
        if x
    } | {
        normalize_sign_code(match.group(0))
        for match in SIGN_CODE_RE.finditer(source_text(record))
    })
    images = sorted({
        x
        for x in [image_path, record.get("image_path"), *(fig.get("image_path") for fig in figures), *(tbl.get("image_path") for tbl in tables)]
        if x
    })
    penalty = penalty_summary(record)
    has_penalty = bool(record.get("penalties")) or any(
        k in source_text(record).lower()
        for k in ["phạt tiền", "trừ điểm", "tước quyền", "đình chỉ", "khắc phục hậu quả"]
    )
    return {
        "source_chunk_id": record.get("source_chunk_id") or record.get("id") or "",
        "doc": record.get("doc_name") or ref.get("document") or "",
        "document_id": ref.get("document_id"),
        "document_type": ref.get("document_type"),
        "article": str(ref.get("article") or ""),
        "clause": str(ref.get("clause") or ""),
        "point": str(ref.get("point") or ""),
        "chapter": str(ref.get("chapter") or ""),
        "section": str(ref.get("section") or ""),
        "page_start": ref.get("page_start") if ref.get("page_start") is not None else (record.get("chunk_meta") or {}).get("page_start"),
        "page_end": ref.get("page_end") if ref.get("page_end") is not None else (record.get("chunk_meta") or {}).get("page_end"),
        "record_id": record.get("id"),
        "record_type": record.get("record_type"),
        "modality": modality,
        "has_table": bool(tables),
        "has_figure": bool(figures or record.get("image_path")),
        "has_sign": bool(sign_codes),
        "has_penalty": has_penalty,
        "has_procedure": looks_like_procedure(record),
        "table_ids": sorted({str(t.get("id")) for t in tables if t.get("id")}),
        "figure_ids": sorted({str(f.get("id")) for f in figures if f.get("id")}),
        "sign_codes": sign_codes,
        "image_paths": images,
        "fine_min_vnd": penalty.get("fine_min_vnd"),
        "fine_max_vnd": penalty.get("fine_max_vnd"),
        "point_deduction": penalty.get("point_deduction"),
        "source_text_sha256": record.get("source_text_sha256"),
    }


def extract_explicit_legal_refs(query: str) -> list[dict[str, str]]:
    refs = []
    for match in LEGAL_REF_RE.finditer(query or ""):
        refs.append({
            "article": (match.group("article") or "").upper(),
            "clause": match.group("clause") or "",
            "point": match.group("point") or "",
        })
    return refs
