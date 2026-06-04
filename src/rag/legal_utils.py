import re
import unicodedata
import hashlib
from typing import Any, Dict, List, Optional, Union

# --- Regex Patterns ---
SIGN_CODE_RE = re.compile(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", re.IGNORECASE)
SIGN_CODE_WITH_PREFIX_RE = re.compile(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", re.IGNORECASE)
LEGAL_REF_RE = re.compile(
    r"(?:(?:điểm)\s+(?P<point>[a-zđ])\s+)?"
    r"(?:(?:khoản)\s+(?P<clause>\d+)\s+)?"
    r"(?:điều)\s+(?P<article>\d+[a-z]?)",
    re.IGNORECASE,
)

def ascii_lower(value: str) -> str:
    """Standardizes strings for consistent search by removing diacritics and normalizing whitespace."""
    if not value: return ""
    normalized = unicodedata.normalize("NFD", value)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()

def looks_like_statutory_fine_cap_query(query: str) -> bool:
    """Distinguishes a stated legal fine cap from a ranking of high-fine violations."""
    qa = ascii_lower(query)
    if "muc phat tien toi da" not in qa:
        return False
    if not any(term in qa for term in ["ca nhan", "to chuc", "doi tuong"]):
        return False
    ranking_terms = [
        "hanh vi nao",
        "cao nhat",
        "nang nhat",
        "top",
        "xep hang",
        "thong ke",
        "tong hop",
    ]
    return not any(term in qa for term in ranking_terms)

def normalize_sign_code(value: str) -> str:
    """Canonicalizes sign codes for uniform lookup."""
    if not value: return ""
    return re.sub(r"[\s.]+", "", value).upper()

def sign_group_from_code(code: str) -> str:
    """Identifies the legal category of a sign based on its code prefix."""
    norm = normalize_sign_code(code)
    mapping = {
        "P": "prohibition", "W": "warning", "R": "mandatory",
        "I": "guide", "S": "supplementary", "DP": "highway", "IE": "highway"
    }
    for prefix, group in mapping.items():
        if norm.startswith(prefix): return group
    return "unknown"

def public_asset_path(path: str) -> str:
    """Converts internal file paths to public-facing URLs."""
    if not path:
        return ""
    normalized = str(path).strip().replace("\\", "/")
    if not normalized:
        return ""
    if normalized.startswith(("http://", "https://")):
        return normalized
    if normalized.startswith("/processed/"):
        return normalized
    if normalized.startswith("processed/"):
        return "/" + normalized
    marker = "data/processed/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
    elif normalized.startswith("/"):
        return normalized
    else:
        relative = normalized
    relative = relative.lstrip("/")
    return "/processed/" + relative

def record_image_paths(record: Dict[str, Any]) -> List[str]:
    """Collects all source/table/figure image paths attached to one RAG record."""
    paths: List[str] = []
    seen = set()

    def add(path: Any) -> None:
        if not path:
            return
        value = str(path).strip()
        if not value or value in seen:
            return
        seen.add(value)
        paths.append(value)

    add(record.get("image_path"))
    for path in record.get("image_paths") or []:
        add(path)

    meta = record.get("rag_metadata") or {}
    for path in meta.get("image_paths") or []:
        add(path)

    table = record.get("table")
    if isinstance(table, dict):
        add(table.get("image_path"))
    for table in record.get("tables") or []:
        if isinstance(table, dict):
            add(table.get("image_path"))

    figure = record.get("figure")
    if isinstance(figure, dict):
        add(figure.get("image_path"))
    for figure in record.get("figures") or []:
        if isinstance(figure, dict):
            add(figure.get("image_path"))

    return paths

def public_record_image_paths(record: Dict[str, Any]) -> List[str]:
    """Returns public URLs for all images attached to one record."""
    return [public_asset_path(path) for path in record_image_paths(record)]

def merge_record_assets(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merges duplicate chunk records without dropping table/figure/page images."""
    image_paths: List[str] = []
    seen_paths = set()
    for record in (base, incoming):
        for path in record_image_paths(record):
            if path not in seen_paths:
                seen_paths.add(path)
                image_paths.append(path)
    if image_paths:
        base["image_paths"] = image_paths
        if not base.get("image_path"):
            base["image_path"] = image_paths[0]

    for field, key_fields in {
        "tables": ("id", "page", "image_path"),
        "figures": ("id", "code", "image_path"),
    }.items():
        merged = []
        seen = set()
        for record in (base, incoming):
            for item in record.get(field) or []:
                if not isinstance(item, dict):
                    continue
                key = tuple(str(item.get(key) or "") for key in key_fields)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        if merged:
            base[field] = merged

    for singular, plural in [("table", "tables"), ("figure", "figures")]:
        item = incoming.get(singular)
        if isinstance(item, dict):
            existing = base.get(plural) if isinstance(base.get(plural), list) else []
            if item not in existing:
                base[plural] = [*existing, item]
            if not isinstance(base.get(singular), dict):
                base[singular] = item

    base["retrieval_reasons"] = sorted(set((base.get("retrieval_reasons") or []) + (incoming.get("retrieval_reasons") or [])))
    base_score = float(base.get("retrieval_score") or 0)
    incoming_score = float(incoming.get("retrieval_score") or 0)
    base["retrieval_score"] = max(base_score, incoming_score) + 0.05 * min(base_score, incoming_score)

    if incoming.get("matched_table_rows"):
        rows = base.get("matched_table_rows") if isinstance(base.get("matched_table_rows"), list) else []
        for row in incoming.get("matched_table_rows") or []:
            if row not in rows:
                rows.append(row)
        base["matched_table_rows"] = rows

    return base

def looks_like_sign_query(query: str) -> bool:
    """Heuristic check for sign-related queries."""
    q = query.lower()
    return any(k in q for k in ["biển báo", "biển cấm", "hình dạng"]) or bool(SIGN_CODE_WITH_PREFIX_RE.search(query))

def looks_like_table_query(query: str) -> bool:
    """Heuristic check for technical table-related queries."""
    q = ascii_lower(query)
    if any(k in q for k in ["bang hieu", "bang ten"]):
        return False
    if any(
        k in q
        for k in [
            "bang lai",
            "bang a1",
            "bang a2",
            "bang b",
            "bang c",
            "bang d",
            "bang e",
            "bang gplx",
            "bang giay phep lai xe",
            "tuoc bang",
        ]
    ):
        return False
    table_terms = [
        "bang phu luc",
        "bang trong phu luc",
        "tra bang",
        "bang thong so",
        "bang quy chuan",
        "dong cot",
        "dong du lieu",
        "cot du lieu",
        "tong so gio",
        "tong thoi gian dao tao",
    ]
    return any(term in q for term in table_terms) or bool(re.search(r"\bbang\b", q))

def format_reference(record: Dict[str, Any]) -> str:
    """Formats a legal citation (e.g., 'Khoản 1 Điều 5 Nghị định 168/2024/NĐ-CP')."""
    ref = normalized_legal_reference(record)
    parts = []
    if ref.get("point"): parts.append(f"Điểm {ref.get('point')}")
    if ref.get("clause"): parts.append(f"Khoản {ref.get('clause')}")
    if ref.get("article"): parts.append(f"Điều {ref.get('article')}")
    if ref.get("section"): parts.append(f"Mục {ref.get('section')}")
    if ref.get("chapter"): parts.append(f"Chương {ref.get('chapter')}")
    
    doc_name = record.get("doc_name") or ref.get("document") or "Văn bản pháp luật"
    parts.append(doc_name)
    return ", ".join(str(p) for p in parts if p)

def source_chunk_ref(source_chunk_id: str) -> Dict[str, str]:
    """Parses structural components from a standardized source chunk ID."""
    parts = (source_chunk_id or "").rsplit("_", 4)
    if len(parts) != 5: return {}
    doc, article, clause, point, _suffix = parts
    return {
        "doc": doc.replace("_", " "),
        "article": article if article != "0" else "",
        "clause": clause if clause != "0" else "",
        "point": point if point != "0" else "",
    }

def normalized_legal_reference(record: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures consistent legal metadata across sources, prioritizing ID-based structure."""
    ref = dict(record.get("legal_reference") or {})
    if record.get("prefer_legal_reference"):
        return ref
    source_ref = source_chunk_ref(record.get("source_chunk_id") or record.get("id") or "")
    if not source_ref: return ref
    if source_ref.get("article"): ref["article"] = source_ref["article"]
    if source_ref.get("clause"): ref["clause"] = source_ref["clause"]
    if source_ref.get("point"): ref["point"] = source_ref["point"]
    return ref

def source_text(record: Dict[str, Any]) -> str:
    """Extracts the best available source body for LLM reasoning."""
    return (
        record.get("source_body_exact") or 
        record.get("rag_text") or 
        record.get("violation_content") or
        record.get("original_text") or
        record.get("content") or ""
    )

def tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer."""
    return re.findall(r"\w+", (text or "").lower())

def extract_explicit_legal_refs(query: str) -> List[Dict[str, str]]:
    """Extracts Article/Clause patterns from a natural language query."""
    refs = []
    for match in LEGAL_REF_RE.finditer(query or ""):
        refs.append({
            "article": (match.group("article") or "").upper(),
            "clause": match.group("clause") or "",
            "point": match.group("point") or "",
        })
    return refs

def penalty_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Summarizes penalty data for consistent indexing and display."""
    penalties = record.get("penalties") or {}
    main = penalties.get("main_penalty") or {}
    return {
        "fine_min_vnd": main.get("min_amount_vnd") or main.get("individual_min_vnd"),
        "fine_max_vnd": main.get("max_amount_vnd") or main.get("individual_max_vnd"),
        "point_deduction": penalties.get("point_deduction"),
        "license_suspension": penalties.get("license_suspension"),
        "raw_penalty_text": main.get("raw_penalty_text") or main.get("description"),
    }

def looks_like_procedure(record: Dict[str, Any]) -> bool:
    """Detects if a record describes an administrative procedure."""
    text = source_text(record).lower()
    keywords = ["thủ tục", "hồ sơ", "trình tự", "thời hạn", "nộp", "đăng ký"]
    return any(k in text for k in keywords)

def record_text_for_index(record: Dict[str, Any]) -> str:
    """Aggregates all searchable text components for vector indexing."""
    return "\n".join(x for x in [
        format_reference(record),
        record.get("semantic_context") or "",
        source_text(record),
    ] if x).strip()

def ref_key(document: str, article: str = "", clause: str = "", point: str = "") -> str:
    """Generates a standardized lookup key for legal references."""
    parts = [document or ""]
    if article: parts.append(f"D{article}")
    if clause: parts.append(f"K{clause}")
    if point: parts.append(f"P{point}")
    return "|".join(parts)

def record_ref_key(record: Dict[str, Any]) -> str:
    """Extracts a standardized lookup key from a record's legal metadata."""
    ref = normalized_legal_reference(record)
    return ref_key(
        ref.get("document") or record.get("doc_name") or "",
        str(ref.get("article") or ""),
        str(ref.get("clause") or ""),
        str(ref.get("point") or ""),
    )

def build_vector_metadata(record: Dict[str, Any], modality: str = "text", image_path: str = "") -> Dict[str, Any]:
    """
    Constructs a flattened metadata dictionary for vector database storage.
    Ensures high-speed filtering by legal attributes.
    """
    ref = normalized_legal_reference(record)
    penalty = penalty_summary(record)
    
    # Extract sign codes from text and figures
    sign_codes = sorted({
        normalize_sign_code(match.group(0))
        for match in SIGN_CODE_RE.finditer(source_text(record))
    })
    
    return {
        "source_chunk_id": record.get("source_chunk_id") or record.get("id") or "",
        "doc": record.get("doc_name") or ref.get("document") or "",
        "article": str(ref.get("article") or ""),
        "clause": str(ref.get("clause") or ""),
        "point": str(ref.get("point") or ""),
        "modality": modality,
        "has_sign": bool(sign_codes),
        "has_penalty": bool(penalty.get("raw_penalty_text")),
        "sign_codes": sign_codes,
        "image_paths": sorted({x for x in [image_path, record.get("image_path")] if x}),
        "fine_min_vnd": penalty.get("fine_min_vnd"),
        "fine_max_vnd": penalty.get("fine_max_vnd"),
    }
