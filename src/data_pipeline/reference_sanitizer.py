import argparse
import json
from pathlib import Path


GARBAGE_TERMS = ("không xác định", "n/a", "none", "unknown", "null", "khh", "không biết")
CORE_REF_FIELDS = ("chapter", "article", "clause", "point")


def is_garbage(value) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or any(term in text for term in GARBAGE_TERMS)


def has_garbage_marker(value) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return any(term in text for term in GARBAGE_TERMS)


def clean_value(value, fallback="") -> tuple[str, bool]:
    if is_garbage(value):
        return str(fallback or "").strip(), True
    return str(value).strip(), False


def load_chunks(chunks_path: Path) -> dict[str, dict]:
    if not chunks_path.exists():
        return {}
    chunks = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_id = chunk.get("source_chunk_id")
            if chunk_id:
                chunks[chunk_id] = chunk
    return chunks


def sanitize_record_reference(record: dict, chunk: dict | None = None, doc_name: str = "") -> tuple[dict, list[str]]:
    ref = dict(record.get("legal_reference") or {})
    chunk = chunk or {}
    warnings = []

    doc_fallback = doc_name or record.get("doc_name") or chunk.get("doc_name") or ref.get("document") or ""
    document, changed = clean_value(ref.get("document"), doc_fallback)
    if changed:
        warnings.append("REF_DOCUMENT_RECOVERED_OR_CLEARED")
    ref["document"] = document

    fallback_map = {
        "chapter": chunk.get("chapter_num", ""),
        "article": chunk.get("article_num", ""),
        "clause": chunk.get("clause_num", ""),
        "point": chunk.get("point_key", ""),
    }
    for field in CORE_REF_FIELDS:
        raw_value = ref.get(field)
        raw_text = "" if raw_value is None else str(raw_value).strip()
        fallback_text = str(fallback_map[field] or "").strip()
        has_marker = has_garbage_marker(raw_value)

        if raw_text and not has_marker:
            cleaned = raw_text
            changed = False
        elif fallback_text:
            cleaned = fallback_text
            changed = raw_text != fallback_text
        else:
            cleaned = ""
            # Empty article/clause/point fields are valid for higher-level chunks.
            # Only warn when we actually removed a garbage marker such as unknown/n/a.
            changed = has_marker

        if changed and cleaned:
            warnings.append(f"REF_{field.upper()}_RECOVERED_FROM_CHUNK")
        elif has_marker:
            warnings.append(f"REF_{field.upper()}_EMPTY_AFTER_SANITIZE")
        ref[field] = cleaned

    if chunk:
        if chunk.get("page_start") is not None:
            ref["page_start"] = chunk.get("page_start")
        if chunk.get("page_end") is not None:
            ref["page_end"] = chunk.get("page_end")

    record["legal_reference"] = ref
    if document:
        record["doc_name"] = record.get("doc_name") or document
    if chunk:
        record.setdefault("source_body_exact", chunk.get("text", ""))
        record.setdefault("semantic_context", chunk.get("semantic_context", ""))
        record.setdefault("parent_hierarchy", chunk.get("parent_hierarchy", []))
        record.setdefault("chunk_meta", {
            "chapter_num": chunk.get("chapter_num", ""),
            "article_num": chunk.get("article_num", ""),
            "clause_num": chunk.get("clause_num", ""),
            "point_key": chunk.get("point_key", ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "kind": chunk.get("kind", ""),
        })

    if warnings:
        meta = record.get("extraction_meta")
        if not isinstance(meta, dict):
            meta = {}
        existing = meta.get("warnings")
        if not isinstance(existing, list):
            existing = []
        new_warnings = []
        for warning in warnings:
            if warning not in existing:
                existing.append(warning)
                new_warnings.append(warning)
        meta["warnings"] = existing
        record["extraction_meta"] = meta
        warnings = new_warnings

    return record, warnings


def sanitize_extracted_file(processed_path: Path, chunks_path: Path, doc_name: str = "", write: bool = True) -> dict:
    if not processed_path.exists():
        return {"file": processed_path.name, "error": "missing processed file"}
    with processed_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    chunks = load_chunks(chunks_path)
    changed_records = 0
    warning_counts = {}
    garbage_before = 0
    garbage_after = 0

    for record in records:
        ref_before = record.get("legal_reference") or {}
        if any(has_garbage_marker(ref_before.get(field)) for field in ("document", *CORE_REF_FIELDS)):
            garbage_before += 1

        chunk = chunks.get(record.get("source_chunk_id"))
        before_json = json.dumps(record.get("legal_reference") or {}, ensure_ascii=False, sort_keys=True)
        _, warnings = sanitize_record_reference(record, chunk, doc_name)
        after_json = json.dumps(record.get("legal_reference") or {}, ensure_ascii=False, sort_keys=True)
        if before_json != after_json or warnings:
            changed_records += 1
        for warning in warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1

        ref_after = record.get("legal_reference") or {}
        if any(has_garbage_marker(ref_after.get(field)) for field in ("document", *CORE_REF_FIELDS)):
            garbage_after += 1

    if write and changed_records:
        tmp_path = processed_path.with_suffix(processed_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        tmp_path.replace(processed_path)

    return {
        "file": processed_path.name,
        "records": len(records),
        "changed_records": changed_records,
        "garbage_reference_records_before": garbage_before,
        "garbage_reference_records_after": garbage_after,
        "warning_counts": warning_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize legal_reference fields using chunk ground truth.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--file", help="Processed file fragment.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    chunks_dir = Path(args.chunks_dir)
    files = sorted(processed_dir.glob("*.extracted.json"))
    if args.file:
        files = [p for p in files if args.file in p.name]

    reports = []
    for processed_path in files:
        base = processed_path.name.removesuffix(".pdf.extracted.json")
        chunks_path = chunks_dir / f"{base}.chunks.jsonl"
        reports.append(sanitize_extracted_file(processed_path, chunks_path, write=not args.dry_run))

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(
            f"{report['file']}: changed={report.get('changed_records')} "
            f"garbage_before={report.get('garbage_reference_records_before')} "
            f"garbage_after={report.get('garbage_reference_records_after')}"
        )


if __name__ == "__main__":
    main()
