import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

from src.rag.legal_utils import (
    build_vector_metadata,
    normalize_sign_code,
    normalized_legal_reference,
    record_text_for_index,
)


_CACHE_VERSION = "expanded-records-v5_asset_scoped_qcvn_sign_code_guard"
_NORMALIZED_SIGN_CODE_RE = re.compile(r"^(?:DP|IE|[PWRISE])\d{2,3}[A-ZĐ]?$", re.IGNORECASE)


def _truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _processed_paths(processed_path: str | Path) -> list[Path]:
    path = Path(processed_path)
    if path.is_dir():
        return sorted(path.glob("*.extracted.json"))
    if path.exists():
        return [path]
    return []


def _expanded_cache_path(processed_path: str | Path) -> Path | None:
    paths = _processed_paths(processed_path)
    if not paths:
        return None
    signature = {
        "version": _CACHE_VERSION,
        "paths": [
            {
                "path": str(path.resolve()),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ],
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_dir = Path(os.getenv("RAG_EXPANDED_RECORDS_CACHE_DIR", "data/vector_db/expanded_records_cache"))
    return cache_dir / f"{digest}.pkl"


def load_processed_records(processed_path: str | Path = "data/processed") -> list[dict[str, Any]]:
    """Load canonical extracted legal records from a directory or a single file."""
    paths = _processed_paths(processed_path)

    records: list[dict[str, Any]] = []
    for item in paths:
        try:
            with item.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                records.extend(record for record in data if isinstance(record, dict))
        except Exception:
            continue
    return records


def _dedupe_assets(items: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        key = tuple(str(item.get(k) or "") for k in keys)
        if not any(key):
            key = (json.dumps(item, ensure_ascii=False, sort_keys=True),)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _record_sign_codes(record: dict[str, Any]) -> set[str]:
    codes: set[str] = set()
    sign_info = record.get("sign_info") if isinstance(record.get("sign_info"), dict) else {}
    for value in [
        sign_info.get("sign_code"),
        (record.get("legal_reference") or {}).get("figure"),
        record.get("sign_code"),
    ]:
        normalized = normalize_sign_code(str(value or ""))
        if normalized:
            codes.add(normalized)

    # QCVN LLM extraction usually makes IDs such as "P.101_1_5ca31e".
    doc_name = str(record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "").lower()
    record_id = str(record.get("id") or "")
    if record_id and ("qcvn" in doc_name or "thông tư 51" in doc_name):
        first = record_id.split("_", 1)[0]
        normalized = normalize_sign_code(first)
        if normalized and _NORMALIZED_SIGN_CODE_RE.fullmatch(normalized):
            codes.add(normalized)
    return codes


def _figures_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    figures = _dedupe_assets(record.get("figures", []) or [], keys=("id", "code", "image_path"))
    target_codes = _record_sign_codes(record)
    if not target_codes:
        return figures

    matched = [
        fig
        for fig in figures
        if normalize_sign_code(str(fig.get("code") or "")) in target_codes
    ]
    # If the record is sign-specific but no crop matched, keep no unrelated sign
    # assets. The text record remains available and the catalog can still use
    # sign_info/name fields.
    return matched


def _tables_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    return _dedupe_assets(record.get("tables", []) or [], keys=("id", "page", "image_path", "text"))


def expand_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one canonical legal record into independently retrievable modalities."""
    out: list[dict[str, Any]] = []
    record = dict(record)
    record["legal_reference"] = normalized_legal_reference(record)
    record["figures"] = _figures_for_record(record)
    record["figure_refs"] = [fig.get("id") for fig in record["figures"] if fig.get("id")]
    record["tables"] = _tables_for_record(record)
    record["table_refs"] = [table.get("id") for table in record["tables"] if table.get("id")]
    base_text = record_text_for_index(record)
    if base_text:
        base = dict(record)
        base["rag_modality"] = "text"
        base["rag_text"] = base_text
        base["rag_parent_id"] = record.get("id")
        base["rag_metadata"] = build_vector_metadata(record, modality="text", image_path=base.get("image_path", ""))
        out.append(base)

    for table in record.get("tables", []) or []:
        if not isinstance(table, dict):
            continue
        table_text = table.get("text") or table.get("caption") or ""
        if not table_text and not table.get("image_path"):
            continue
        rec = dict(record)
        rec["rag_modality"] = "table"
        rec["rag_parent_id"] = record.get("id")
        rec["table"] = table
        rec["image_path"] = table.get("image_path") or record.get("image_path", "")
        rec["rag_text"] = "\n".join(
            x for x in [record_text_for_index(record), f"Bảng {table.get('id', '')}", table_text] if x
        )
        rec["rag_metadata"] = build_vector_metadata(rec, modality="table", image_path=rec.get("image_path", ""))
        out.append(rec)

    emitted_sign_codes: set[str] = set()
    for fig in record.get("figures", []) or []:
        if not isinstance(fig, dict):
            continue
        rec = dict(record)
        rec["rag_modality"] = "figure"
        rec["rag_parent_id"] = record.get("id")
        rec["figure"] = fig
        rec["image_path"] = fig.get("image_path") or record.get("image_path", "")
        rec["rag_text"] = "\n".join(
            x
            for x in [
                record_text_for_index(record),
                f"Biển báo/Hình {fig.get('code', '')} {fig.get('name', '')}".strip(),
                fig.get("caption") or "",
            ]
            if x
        )
        rec["rag_metadata"] = build_vector_metadata(rec, modality="figure", image_path=rec.get("image_path", ""))
        out.append(rec)
        if fig.get("code"):
            emitted_sign_codes.add(normalize_sign_code(str(fig.get("code") or "")))
            sign_rec = dict(record)
            sign_rec["rag_modality"] = "sign"
            sign_rec["rag_parent_id"] = record.get("id")
            sign_rec["figure"] = fig
            sign_rec["image_path"] = fig.get("image_path") or record.get("image_path", "")
            sign_rec["rag_text"] = "\n".join(
                x
                for x in [
                    f"Mã biển báo: {fig.get('code', '')}",
                    f"Tên/nhãn biển báo: {fig.get('name', '')}",
                    f"Mô tả hình ảnh: {fig.get('caption', '')}",
                    record_text_for_index(record),
                ]
                if x
            )
            sign_rec["rag_metadata"] = build_vector_metadata(
                sign_rec,
                modality="sign",
                image_path=sign_rec.get("image_path", ""),
            )
            out.append(sign_rec)

    sign_info = record.get("sign_info") if isinstance(record.get("sign_info"), dict) else {}
    for code in sorted(_record_sign_codes(record) - emitted_sign_codes):
        display_code = sign_info.get("sign_code") or code
        synthetic_fig = {
            "id": f"synthetic_{code}",
            "code": display_code,
            "name": sign_info.get("sign_name") or "",
            "caption": record.get("caption") or record.get("meaning_and_usage") or "",
            "image_path": record.get("image_path") or "",
            "source": "sign_info",
        }
        sign_rec = dict(record)
        sign_rec["rag_modality"] = "sign"
        sign_rec["rag_parent_id"] = record.get("id")
        sign_rec["figure"] = synthetic_fig
        sign_rec["image_path"] = synthetic_fig["image_path"]
        sign_rec["rag_text"] = "\n".join(
            x
            for x in [
                f"Mã biển báo: {display_code}",
                f"Tên/nhãn biển báo: {synthetic_fig['name']}",
                f"Mô tả/ý nghĩa: {synthetic_fig['caption']}",
                record_text_for_index(record),
            ]
            if x
        )
        sign_rec["rag_metadata"] = build_vector_metadata(
            sign_rec,
            modality="sign",
            image_path=sign_rec.get("image_path", ""),
        )
        out.append(sign_rec)
    return out


def load_expanded_records(processed_path: str | Path = "data/processed") -> list[dict[str, Any]]:
    use_cache = not _truthy_env("RAG_DISABLE_EXPANDED_RECORDS_CACHE", False)
    cache_path = _expanded_cache_path(processed_path) if use_cache else None
    if cache_path and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, list):
                return cached
        except Exception:
            pass

    expanded: list[dict[str, Any]] = []
    for record in load_processed_records(processed_path):
        expanded.extend(expand_record(record))
    if cache_path:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(expanded, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return expanded
