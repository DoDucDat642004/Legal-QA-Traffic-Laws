import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from src.data_pipeline.coverage_validator import CoverageValidator
from src.data_pipeline.reference_sanitizer import has_garbage_marker


GARBAGE_TERMS = ("không xác định", "n/a", "none", "unknown", "null", "khh", "không biết")
SIGN_CODE_RE = re.compile(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", re.IGNORECASE)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _bad(value) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or any(term in text for term in GARBAGE_TERMS)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _doc_base_from_processed(path: Path) -> str:
    name = path.name
    if name.endswith(".pdf.extracted.json"):
        return name[: -len(".pdf.extracted.json")]
    return name.replace(".extracted.json", "")


def audit_file(processed_path: Path, chunks_dir: Path) -> dict:
    base = _doc_base_from_processed(processed_path)
    chunks_path = chunks_dir / f"{base}.chunks.jsonl"
    records = _load_json(processed_path)
    chunks = _load_jsonl(chunks_path) if chunks_path.exists() else []

    bad_refs = []
    garbage_ref_values = []
    original_not_in_source = []
    very_short_original = []
    record_types = Counter(record.get("record_type") for record in records)
    sign_codes_in_text = set()
    sign_codes_with_linked_asset = set()
    table_link_count = 0
    figure_link_count = 0

    for record in records:
        ref = record.get("legal_reference") or {}
        for field in ("document", "chapter", "article", "clause", "point"):
            if has_garbage_marker(ref.get(field)):
                garbage_ref_values.append({
                    "id": record.get("id") or record.get("source_chunk_id") or "unknown",
                    "field": field,
                    "value": ref.get(field),
                })
        if _bad(ref.get("document")) or (_bad(ref.get("article")) and _bad(ref.get("clause")) and _bad(ref.get("point"))):
            bad_refs.append(record.get("id") or record.get("source_chunk_id") or "unknown")

        if record.get("record_type") == "source_legal_unit":
            continue

        raw_source = record.get("source_body_exact") or record.get("content") or ""
        source = _norm(raw_source)
        extracted = _norm(record.get("original_text") or record.get("violation_content") or record.get("meaning_and_usage") or "")
        for match in SIGN_CODE_RE.finditer(raw_source):
            sign_codes_in_text.add(re.sub(r"\s+", "", match.group(0)).upper())

        table_link_count += len(record.get("tables") or [])
        figures = record.get("figures") or []
        figure_link_count += len(figures)
        for fig in figures:
            if isinstance(fig, dict) and fig.get("code"):
                sign_codes_with_linked_asset.add(re.sub(r"\s+", "", str(fig["code"])).upper())

        if source and extracted:
            if extracted not in source:
                original_not_in_source.append(record.get("id") or record.get("source_chunk_id") or "unknown")
            if len(extracted) < len(source) * 0.5:
                very_short_original.append(record.get("id") or record.get("source_chunk_id") or "unknown")

    validator = CoverageValidator()
    coverage = validator.validate(records, "", chunks if chunks else None)
    chunk_coverage = coverage.get("source_chunk_coverage") or {}

    return {
        "file": processed_path.name,
        "chunks": len(chunks),
        "records": len(records),
        "record_types": dict(record_types),
        "bad_reference_count": len(bad_refs),
        "bad_reference_examples": bad_refs[:10],
        "garbage_reference_value_count": len(garbage_ref_values),
        "garbage_reference_value_examples": garbage_ref_values[:20],
        "source_only_chunk_count": chunk_coverage.get("source_only_chunk_count", 0),
        "missing_chunk_count": len(chunk_coverage.get("missing_chunk_ids", [])),
        "missing_coordinate_count": len(chunk_coverage.get("missing_coordinates", [])),
        "exact_text_hash_issue_count": chunk_coverage.get("exact_text_hash_issue_count", 0),
        "original_not_in_source_count": len(original_not_in_source),
        "original_not_in_source_examples": original_not_in_source[:10],
        "very_short_original_count": len(very_short_original),
        "very_short_original_examples": very_short_original[:10],
        "table_link_count": table_link_count,
        "figure_link_count": figure_link_count,
        "sign_code_mention_count": len(sign_codes_in_text),
        "sign_code_with_asset_count": len(sign_codes_with_linked_asset),
        "sign_code_asset_gap_count": max(0, len(sign_codes_in_text - sign_codes_with_linked_asset)),
        "sign_code_asset_gap_examples": sorted(sign_codes_in_text - sign_codes_with_linked_asset)[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit legal extraction coverage and traceability.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--file", help="Audit one processed file name or basename.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    chunks_dir = Path(args.chunks_dir)
    files = sorted(processed_dir.glob("*.extracted.json"))
    if args.file:
        files = [p for p in files if args.file in p.name]

    reports = [audit_file(path, chunks_dir) for path in files]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(f"\n{report['file']}")
        print(f"  chunks={report['chunks']} records={report['records']} record_types={report['record_types']}")
        print(
            "  bad_refs={bad_reference_count} source_only_chunks={source_only_chunk_count} "
            "garbage_ref_values={garbage_reference_value_count} "
            "missing_chunks={missing_chunk_count} missing_coords={missing_coordinate_count} "
            "hash_issues={exact_text_hash_issue_count}".format(**report)
        )
        print(
            "  original_not_in_source={original_not_in_source_count} "
            "very_short_original={very_short_original_count}".format(**report)
        )
        print(
            "  table_links={table_link_count} figure_links={figure_link_count} "
            "sign_codes={sign_code_mention_count} sign_codes_with_assets={sign_code_with_asset_count} "
            "sign_asset_gap={sign_code_asset_gap_count}".format(**report)
        )
        if report["bad_reference_examples"]:
            print(f"  bad_ref_examples={report['bad_reference_examples']}")
        if report["original_not_in_source_examples"]:
            print(f"  original_not_in_source_examples={report['original_not_in_source_examples']}")


if __name__ == "__main__":
    main()
