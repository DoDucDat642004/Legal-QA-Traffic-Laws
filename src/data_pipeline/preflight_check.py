import argparse
import json
import sys
import time
from pathlib import Path

from src.data_pipeline.legal_extraction import CHUNK_CACHE_VERSION, EXTRACTION_CACHE_VERSION
from src.data_pipeline.pdf_quality_audit import audit_pdf_quality
from src.data_pipeline.qa_generator import QA_PIPELINE_VERSION


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _lock_report(lock_dir: Path) -> list[dict]:
    reports = []
    now = time.time()
    for path in sorted(lock_dir.glob("*.lock")):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        started = None
        for token in text.split():
            if token.startswith("started="):
                try:
                    started = float(token.split("=", 1)[1])
                except ValueError:
                    started = None
        reports.append({
            "file": path.name,
            "content": text,
            "age_seconds": int(now - started) if started else None,
        })
    return reports


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _chunk_meta_report(chunks_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(chunks_dir.glob("*.chunks.jsonl")):
        meta_path = Path(str(path) + ".meta.json")
        meta = _load_json(meta_path) if meta_path.exists() else {}
        reports.append({
            "file": path.name,
            "chunk_count": _jsonl_count(path),
            "version": meta.get("chunk_cache_version"),
            "ok": meta.get("chunk_cache_version") == CHUNK_CACHE_VERSION,
        })
    return reports


def _processed_meta_report(processed_dir: Path) -> list[dict]:
    reports = []
    expected = {
        "extraction_cache_version": EXTRACTION_CACHE_VERSION,
        "chunk_cache_version": CHUNK_CACHE_VERSION,
    }
    for path in sorted(processed_dir.glob("*.extracted.json")):
        meta_path = Path(str(path) + ".meta.json")
        meta = _load_json(meta_path) if meta_path.exists() else {}
        reports.append({
            "file": path.name,
            "has_meta": meta_path.exists(),
            "chunk_count": meta.get("chunk_count"),
            "ok": all(meta.get(k) == v for k, v in expected.items()),
            "version": {k: meta.get(k) for k in expected},
        })
    return reports


def _qa_meta_report(qa_dir: Path) -> list[dict]:
    reports = []
    for path in sorted(qa_dir.glob("*.qa.json")):
        meta_path = Path(str(path) + ".meta.json")
        meta = _load_json(meta_path) if meta_path.exists() else {}
        reports.append({
            "file": path.name,
            "has_meta": meta_path.exists(),
            "qa_count": meta.get("qa_count"),
            "record_count": meta.get("record_count"),
            "version": meta.get("qa_pipeline_version"),
            "ok": meta.get("qa_pipeline_version") == QA_PIPELINE_VERSION,
        })
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-flight checks before long legal data pipeline runs.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--qa-dir", default="data/qa_pairs")
    parser.add_argument("--lock-dir", default="data/locks")
    parser.add_argument(
        "--before-run",
        action="store_true",
        help="Only gate inputs/cache/locks before starting a run; processed/QA outputs may be stale because the run will rebuild them.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    interim_dir = Path(args.interim_dir)
    chunks_dir = Path(args.chunks_dir)
    processed_dir = Path(args.processed_dir)
    qa_dir = Path(args.qa_dir)
    lock_dir = Path(args.lock_dir)

    pdf_reports = [
        audit_pdf_quality(path, interim_dir, chunks_dir)
        for path in sorted(raw_dir.glob("*.pdf"))
    ]
    locks = _lock_report(lock_dir)
    chunks = _chunk_meta_report(chunks_dir)
    processed = _processed_meta_report(processed_dir)
    qa = _qa_meta_report(qa_dir)

    failures = []
    if locks:
        failures.append(f"{len(locks)} lock file(s) exist; do not start another pipeline run until confirmed stale.")
    for report in pdf_reports:
        if not report.get("page_count_match"):
            failures.append(f"{report['file']}: page count mismatch")
        if report.get("pages_with_tables_not_linked_to_chunks"):
            failures.append(f"{report['file']}: unlinked table pages remain")
    for report in chunks:
        if not report["ok"]:
            failures.append(f"{report['file']}: stale chunk cache version")
    if not args.before_run:
        for report in processed:
            if not report["ok"]:
                failures.append(f"{report['file']}: missing/stale processed meta")
        for report in qa:
            if not report["ok"]:
                failures.append(f"{report['file']}: missing/stale QA meta")

    output = {
        "ok": not failures,
        "mode": "before_run" if args.before_run else "after_run",
        "failures": failures,
        "locks": locks,
        "pdf_summary": [
            {
                "file": r["file"],
                "engine_counts": r["engine_counts"],
                "recommended_engine": (r.get("engine_recommendation") or {}).get("recommended_engine"),
                "table_pages_not_linked": len(r.get("pages_with_tables_not_linked_to_chunks") or []),
                "low_corrected_text_pages": len(r.get("low_corrected_text_pages") or []),
            }
            for r in pdf_reports
        ],
        "chunks": chunks,
        "processed": processed,
        "qa": qa,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print("PRE-FLIGHT:", "OK" if output["ok"] else "FAILED")
        for failure in failures:
            print(f" - {failure}")
        if locks:
            print("\nLocks:")
            for lock in locks:
                print(f" - {lock['file']}: {lock['content']} age={lock['age_seconds']}s")
        print("\nPDF summary:")
        for report in output["pdf_summary"]:
            print(
                f" - {report['file']}: engine={report['engine_counts']} "
                f"recommended={report['recommended_engine']} "
                f"unlinked_tables={report['table_pages_not_linked']} "
                f"low_pages={report['low_corrected_text_pages']}"
            )

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
