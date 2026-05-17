import argparse
import json
from collections import Counter
from pathlib import Path

import fitz


def _recommend_engine(pdf_path: Path, page_text_lengths: list[int], page_image_counts: list[int]) -> dict:
    pages = len(page_text_lengths) or 1
    avg_text_len = sum(page_text_lengths) / pages
    text_rich_ratio = sum(1 for length in page_text_lengths if length >= 500) / pages
    low_text_ratio = sum(1 for length in page_text_lengths if length < 80) / pages
    image_heavy_ratio = sum(
        1 for length, images in zip(page_text_lengths, page_image_counts) if images > 0 and length < 250
    ) / pages
    images_per_page = sum(page_image_counts) / pages
    name = pdf_path.name.lower()

    if ("qcvn" in name or "51-bgtvt" in name) and image_heavy_ratio >= 0.20 and images_per_page >= 5:
        return {
            "recommended_engine": "llamaparse",
            "reason": "QCVN/biển báo có nhiều trang ảnh; cần OCR/layout để giữ bảng và hình biển báo.",
            "avg_text_len": round(avg_text_len, 1),
            "text_rich_ratio": round(text_rich_ratio, 3),
            "low_text_ratio": round(low_text_ratio, 3),
            "image_heavy_ratio": round(image_heavy_ratio, 3),
            "images_per_page": round(images_per_page, 1),
        }

    if text_rich_ratio >= 0.30 and low_text_ratio >= 0.15:
        return {
            "recommended_engine": "hybrid",
            "reason": "PDF có phần đầu/giữa là text tốt nhưng nhiều trang text layer yếu; dùng fitz cho trang rõ và OCR/LlamaParse cho trang yếu.",
            "avg_text_len": round(avg_text_len, 1),
            "text_rich_ratio": round(text_rich_ratio, 3),
            "low_text_ratio": round(low_text_ratio, 3),
            "image_heavy_ratio": round(image_heavy_ratio, 3),
            "images_per_page": round(images_per_page, 1),
        }

    if avg_text_len >= 500 or text_rich_ratio >= 0.30:
        return {
            "recommended_engine": "fitz",
            "reason": "PDF có text layer đủ mạnh; dùng fitz để tránh OCR/LlamaParse làm méo chữ luật.",
            "avg_text_len": round(avg_text_len, 1),
            "text_rich_ratio": round(text_rich_ratio, 3),
            "low_text_ratio": round(low_text_ratio, 3),
            "image_heavy_ratio": round(image_heavy_ratio, 3),
            "images_per_page": round(images_per_page, 1),
        }

    return {
        "recommended_engine": "llamaparse",
        "reason": "Text layer yếu hoặc nhiều trang ít chữ; cần OCR/layout parser.",
        "avg_text_len": round(avg_text_len, 1),
        "text_rich_ratio": round(text_rich_ratio, 3),
        "low_text_ratio": round(low_text_ratio, 3),
        "image_heavy_ratio": round(image_heavy_ratio, 3),
        "images_per_page": round(images_per_page, 1),
    }


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _table_issue(table: dict) -> list[str]:
    issues = []
    rows = table.get("rows") or []
    if not rows:
        return ["EMPTY_TABLE_ROWS"]

    widths = [len(row or []) for row in rows]
    if len(set(widths)) > 1:
        issues.append("RAGGED_ROWS")

    total_cells = sum(widths)
    empty_cells = sum(1 for row in rows for cell in (row or []) if not str(cell or "").strip())
    if total_cells and empty_cells / total_cells > 0.45:
        issues.append("HIGH_EMPTY_CELL_RATIO")

    if len(rows) < 2 and "|" not in (table.get("text") or ""):
        issues.append("TOO_FEW_ROWS")

    if table.get("bbox") and not table.get("image_path"):
        issues.append("MISSING_TABLE_CROP")

    return issues


def audit_pdf_quality(pdf_path: Path, interim_dir: Path, chunks_dir: Path) -> dict:
    doc_base = pdf_path.name.replace(".pdf", "")
    manifest_path = interim_dir / doc_base / "manifest.json"
    chunks_path = chunks_dir / f"{doc_base}.chunks.jsonl"

    with fitz.open(pdf_path) as doc:
        page_count = len(doc)
        fitz_page_text = [len(doc[i].get_text("text").strip()) for i in range(len(doc))]
        fitz_page_images = [len(doc[i].get_images(full=True)) for i in range(len(doc))]

    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    doc_map = manifest.get("doc_map") or {}

    engine_counts = Counter()
    empty_pages = []
    low_text_pages = []
    table_count = 0
    table_issue_count = 0
    table_issue_examples = []
    pages_with_tables = set()

    for page_key, page in doc_map.items():
        page_num = int(page_key)
        engine_counts[page.get("extraction_engine", "unknown")] += 1
        corrected_len = len((page.get("corrected") or "").strip())
        if corrected_len == 0:
            empty_pages.append(page_num)
        if corrected_len < 80:
            low_text_pages.append(page_num)

        tables = page.get("tables") or []
        if tables:
            pages_with_tables.add(page_num)
        for table in tables:
            table_count += 1
            issues = _table_issue(table)
            if issues:
                table_issue_count += 1
                if len(table_issue_examples) < 20:
                    table_issue_examples.append({
                        "page": page_num,
                        "table_id": table.get("id"),
                        "issues": issues,
                    })

    chunk_table_pages = set()
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                for table in chunk.get("tables") or []:
                    if isinstance(table, dict) and table.get("page") is not None:
                        chunk_table_pages.add(int(table["page"]))

    return {
        "file": pdf_path.name,
        "pdf_page_count": page_count,
        "doc_map_page_count": len(doc_map),
        "page_count_match": page_count == len(doc_map),
        "engine_recommendation": _recommend_engine(pdf_path, fitz_page_text, fitz_page_images),
        "fitz_low_text_pages": [i for i, length in enumerate(fitz_page_text) if length < 80],
        "empty_corrected_pages": empty_pages,
        "low_corrected_text_pages": low_text_pages,
        "engine_counts": dict(engine_counts),
        "table_count": table_count,
        "table_issue_count": table_issue_count,
        "table_issue_examples": table_issue_examples,
        "pages_with_tables": sorted(pages_with_tables),
        "pages_with_tables_not_linked_to_chunks": sorted(pages_with_tables - chunk_table_pages),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PDF extraction, OCR/LlamaParse, and table quality.")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--chunks-dir", default="data/chunks")
    parser.add_argument("--file", help="PDF filename fragment.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.pdf"))
    if args.file:
        files = [p for p in files if args.file in p.name]

    reports = [audit_pdf_quality(path, Path(args.interim_dir), Path(args.chunks_dir)) for path in files]
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return

    for report in reports:
        print(f"\n{report['file']}")
        print(
            f"  pages={report['doc_map_page_count']}/{report['pdf_page_count']} "
            f"engines={report['engine_counts']} empty_pages={len(report['empty_corrected_pages'])} "
            f"low_text_pages={len(report['low_corrected_text_pages'])}"
        )
        engine_rec = report["engine_recommendation"]
        print(
            f"  recommended_engine={engine_rec['recommended_engine']} "
            f"avg_text={engine_rec['avg_text_len']} rich_ratio={engine_rec['text_rich_ratio']} "
            f"image_heavy={engine_rec['image_heavy_ratio']} images/page={engine_rec['images_per_page']}"
        )
        print(
            f"  tables={report['table_count']} table_issues={report['table_issue_count']} "
            f"table_pages_not_linked={len(report['pages_with_tables_not_linked_to_chunks'])}"
        )


if __name__ == "__main__":
    main()
