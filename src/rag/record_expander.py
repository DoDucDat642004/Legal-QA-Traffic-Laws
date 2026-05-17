import json
from pathlib import Path
from typing import Any

from src.rag.legal_utils import build_vector_metadata, normalized_legal_reference, record_text_for_index


def load_processed_records(processed_path: str | Path = "data/processed") -> list[dict[str, Any]]:
    """Load canonical extracted legal records from a directory or a single file."""
    path = Path(processed_path)
    paths: list[Path] = []
    if path.is_dir():
        paths = sorted(path.glob("*.extracted.json"))
    elif path.exists():
        paths = [path]

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


def expand_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one canonical legal record into independently retrievable modalities."""
    out: list[dict[str, Any]] = []
    record = dict(record)
    record["legal_reference"] = normalized_legal_reference(record)
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
    return out


def load_expanded_records(processed_path: str | Path = "data/processed") -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for record in load_processed_records(processed_path):
        expanded.extend(expand_record(record))
    return expanded
