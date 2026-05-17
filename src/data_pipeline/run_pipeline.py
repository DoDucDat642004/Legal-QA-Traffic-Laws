import argparse
import asyncio
from contextlib import contextmanager
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

load_dotenv(os.path.join(project_root, ".env"))

from src.data_pipeline.extraction_audit_report import audit_file
from src.data_pipeline.graph_exporter import export_graph
from src.data_pipeline.legal_extraction import process_document
from src.data_pipeline.pdf_quality_audit import audit_pdf_quality
from src.data_pipeline.qa_audit_report import audit_qa_file
from src.data_pipeline.qa_generator import process_generate_qa
from src.data_pipeline.reference_sanitizer import sanitize_extracted_file


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Pipeline")


DOCUMENTS = {
    "168-nd-cp.signed.pdf": "Nghị định 168/2024/NĐ-CP",
    "336-2025-nd-cp-22122025-signed-17665482569851736009102.pdf": "Nghị định 336/2025/NĐ-CP",
    "35-2024-qh15.pdf": "Luật Đường bộ 2024",
    "35-bgtvt.pdf": "Thông tư 35/2024/TT-BGTVT",
    "36-2024-qh15.pdf": "Luật Trật tự ATGT 2024",
    "36-2024-qh15_tiep.pdf": "Luật Trật tự ATGT 2024 (Tiếp)",
    "51-bgtvt-kem.pdf": "QCVN 41:2024 (Thông tư 51/2024)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end legal data pipeline: extract, repair/enrich, QA, audit, and RAG index."
    )
    parser.add_argument("target", nargs="?", help="Optional filename fragment, e.g. 35-2024-qh15 or 51-bgtvt-kem.")
    parser.add_argument("--list-docs", action="store_true", help="Print configured documents and exit.")
    parser.add_argument("--skip-extraction", action="store_true", help="Use existing extracted JSON.")
    parser.add_argument("--skip-qa", action="store_true", help="Skip QA generation/repair.")
    parser.add_argument(
        "--defer-qa-until-all-extracted",
        action="store_true",
        help="Run QA only after all selected documents finish extraction. By default QA is generated immediately after each document.",
    )
    parser.add_argument("--skip-rag-index", action="store_true", help="Skip vector DB rebuild.")
    parser.add_argument("--force-rag-index", action="store_true", help="Force vector DB rebuild even if cache exists.")
    parser.add_argument("--build-rag-index", action="store_true", help="Build vector DB even when running a single target.")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph JSON export.")
    parser.add_argument("--sync-rag-stores", action="store_true", help="Sync data into PostgreSQL, Qdrant, Neo4j, and MinIO after graph/RAG build.")
    parser.add_argument("--lock-timeout", type=int, default=21600, help="Seconds to wait for per-file pipeline lock.")
    parser.add_argument("--audit-only", action="store_true", help="Only run extraction and QA audits.")
    return parser.parse_args()


def select_docs(target: str | None) -> dict[str, str]:
    if not target:
        return DOCUMENTS

    target_norm = target.strip().lower()

    def normalized_names(filename: str, doc_name: str) -> set[str]:
        stem = filename[:-4] if filename.lower().endswith(".pdf") else filename
        return {
            filename.lower(),
            stem.lower(),
            doc_name.lower(),
        }

    # Prefer exact file/stem/document matches. This prevents target "36-2024-qh15"
    # from also selecting "36-2024-qh15_tiep".
    exact_docs = {
        filename: doc
        for filename, doc in DOCUMENTS.items()
        if target_norm in normalized_names(filename, doc)
    }
    if exact_docs:
        return exact_docs

    target_docs = {
        filename: doc
        for filename, doc in DOCUMENTS.items()
        if target_norm in filename.lower() or target_norm in doc.lower()
    }
    if not target_docs:
        raise SystemExit(f"Không tìm thấy file nào khớp với: {target}")
    return target_docs


def write_report(report_dir: Path, name: str, data) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Wrote report: %s", path)


@contextmanager
def file_lock(lock_path: Path, timeout_seconds: int):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} started={time.time()}\n".encode("utf-8"))
            break
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            logger.info("Waiting for lock: %s", lock_path)
            time.sleep(10)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def run_pdf_audits(target_docs: dict[str, str], raw_dir: Path, interim_dir: Path, chunks_dir: Path, report_dir: Path) -> list[dict]:
    reports = []
    for filename in target_docs:
        pdf_path = raw_dir / filename
        if pdf_path.exists():
            reports.append(audit_pdf_quality(pdf_path, interim_dir, chunks_dir))
        else:
            reports.append({"file": filename, "error": "missing raw PDF"})
    write_report(report_dir, "pdf_quality_audit.json", reports)
    return reports


def run_extraction_audits(target_docs: dict[str, str], processed_dir: Path, chunks_dir: Path, report_dir: Path) -> list[dict]:
    reports = []
    for filename in target_docs:
        processed_path = processed_dir / f"{filename}.extracted.json"
        if processed_path.exists():
            reports.append(audit_file(processed_path, chunks_dir))
        else:
            reports.append({"file": processed_path.name, "error": "missing extracted JSON"})
    write_report(report_dir, "extraction_audit.json", reports)
    return reports


def run_reference_sanitizer(target_docs: dict[str, str], processed_dir: Path, chunks_dir: Path, report_dir: Path) -> list[dict]:
    reports = []
    for filename, doc_name in target_docs.items():
        processed_path = processed_dir / f"{filename}.extracted.json"
        chunks_path = chunks_dir / f"{filename.replace('.pdf', '')}.chunks.jsonl"
        reports.append(sanitize_extracted_file(processed_path, chunks_path, doc_name=doc_name, write=True))
    write_report(report_dir, "reference_sanitizer.json", reports)
    return reports


def run_qa_audits(target_docs: dict[str, str], qa_dir: Path, chunks_dir: Path, processed_dir: Path, report_dir: Path) -> list[dict]:
    reports = []
    for filename in target_docs:
        qa_path = qa_dir / f"{filename}.qa.json"
        if qa_path.exists():
            reports.append(audit_qa_file(qa_path, chunks_dir, processed_dir))
        else:
            reports.append({"file": qa_path.name, "error": "missing QA JSON"})
    write_report(report_dir, "qa_audit.json", reports)
    return reports


async def run_single_doc_postprocess(
    filename: str,
    doc_name: str,
    processed_dir: Path,
    chunks_dir: Path,
    qa_dir: Path,
    skip_qa: bool,
) -> None:
    """Sanitize references and generate QA as soon as one document is extracted."""
    processed_path = processed_dir / f"{filename}.extracted.json"
    chunks_path = chunks_dir / f"{filename.replace('.pdf', '')}.chunks.jsonl"
    if not processed_path.exists():
        logger.warning("Postprocess skipped; missing extracted JSON: %s", processed_path)
        return

    sanitize_report = sanitize_extracted_file(processed_path, chunks_path, doc_name=doc_name, write=True)
    if sanitize_report.get("changed_records"):
        logger.info(
            "Sanitized references in %s: changed=%s garbage_before=%s garbage_after=%s",
            sanitize_report["file"],
            sanitize_report["changed_records"],
            sanitize_report["garbage_reference_records_before"],
            sanitize_report["garbage_reference_records_after"],
        )

    if not skip_qa:
        output_file = qa_dir / f"{filename}.qa.json"
        logger.info("Generating/repairing QA immediately for %s", filename)
        await process_generate_qa(str(processed_path), str(output_file))


def print_audit_summary(extraction_reports: list[dict], qa_reports: list[dict]) -> None:
    print("\n[SUMMARY] Extraction audit")
    for report in extraction_reports:
        if report.get("error"):
            print(f" - {report['file']}: {report['error']}")
            continue
        print(
            " - {file}: chunks={chunks}, bad_refs={bad_reference_count}, "
            "garbage_ref_values={garbage_reference_value_count}, "
            "source_only={source_only_chunk_count}, missing_coords={missing_coordinate_count}, "
            "sign_asset_gap={sign_code_asset_gap_count}".format(**report)
        )

    print("\n[SUMMARY] QA audit")
    for report in qa_reports:
        if report.get("error"):
            print(f" - {report['file']}: {report['error']}")
            continue
        print(
            " - {file}: qa={qa_count}, covered={covered_chunk_count}/{expected_chunk_count}, "
            "missing={missing_chunk_count}, invalid_quotes={invalid_quote_count}, duplicates={duplicate_question_count}".format(**report)
        )


def print_pdf_summary(pdf_reports: list[dict]) -> None:
    print("\n[SUMMARY] PDF/LlamaParse/Table audit")
    for report in pdf_reports:
        if report.get("error"):
            print(f" - {report['file']}: {report['error']}")
            continue
        print(
            " - {file}: pages={doc_map_page_count}/{pdf_page_count}, engines={engine_counts}, "
            "recommended={recommended}, low_text={low}, tables={table_count}, "
            "table_issues={table_issue_count}, unlinked_table_pages={unlinked}".format(
                file=report["file"],
                doc_map_page_count=report["doc_map_page_count"],
                pdf_page_count=report["pdf_page_count"],
                engine_counts=report["engine_counts"],
                recommended=(report.get("engine_recommendation") or {}).get("recommended_engine", "unknown"),
                low=len(report["low_corrected_text_pages"]),
                table_count=report["table_count"],
                table_issue_count=report["table_issue_count"],
                unlinked=len(report["pages_with_tables_not_linked_to_chunks"]),
            )
        )

def rebuild_rag_index(processed_dir: Path, graph_path: Path, force: bool) -> None:
    try:
        from src.rag.legal_graph_rag import LegalGraphRAG

        LegalGraphRAG(str(processed_dir), graph_path=str(graph_path), force_reindex=force)
        logger.info("Hybrid Legal Graph RAG index is ready.")
    except Exception as e:
        logger.error("RAG index rebuild failed: %s", e)
        logger.error("Extraction/QA outputs are still available; fix embedding dependencies or run with --skip-rag-index.")


async def main() -> None:
    args = parse_args()
    if args.list_docs:
        print("Configured documents:")
        for filename, doc_name in DOCUMENTS.items():
            print(f" - {filename}: {doc_name}")
        return
    target_docs = select_docs(args.target)

    raw_dir = Path(project_root) / "data" / "raw"
    processed_dir = Path(project_root) / "data" / "processed"
    chunks_dir = Path(project_root) / "data" / "chunks"
    interim_dir = Path(project_root) / "data" / "interim"
    qa_dir = Path(project_root) / "data" / "qa_pairs"
    graph_dir = Path(project_root) / "data" / "graph"
    lock_dir = Path(project_root) / "data" / "locks"
    report_dir = Path(project_root) / "data" / "reports" / f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting pipeline for %s document(s).", len(target_docs))

    if not args.audit_only and not args.skip_extraction:
        print("\n[1/6] Trích xuất, parse, enrich và tự repair fallback nếu LLM lỗi...")
        doc_sem = asyncio.Semaphore(1)
        for filename, doc_name in target_docs.items():
            pdf_path = raw_dir / filename
            if pdf_path.exists():
                safe_lock_name = filename.replace("/", "_") + ".lock"
                with file_lock(lock_dir / safe_lock_name, args.lock_timeout):
                    await process_document(str(pdf_path), doc_name, str(processed_dir), doc_sem)
                if not args.defer_qa_until_all_extracted:
                    await run_single_doc_postprocess(
                        filename,
                        doc_name,
                        processed_dir,
                        chunks_dir,
                        qa_dir,
                        skip_qa=args.skip_qa,
                    )
            else:
                logger.warning("Missing raw PDF: %s", pdf_path)
    else:
        print("\n[1/6] Bỏ qua extraction, dùng dữ liệu hiện có.")

    print("\n[2/6] Audit PDF/LlamaParse/table trước khi đánh giá chunk...")
    pdf_reports = run_pdf_audits(target_docs, raw_dir, interim_dir, chunks_dir, report_dir)

    print("\n[3/6] Chuẩn hóa legal_reference và audit extraction coverage...")
    sanitizer_reports = run_reference_sanitizer(target_docs, processed_dir, chunks_dir, report_dir)
    for report in sanitizer_reports:
        if report.get("changed_records"):
            logger.info(
                "Sanitized references in %s: changed=%s garbage_before=%s garbage_after=%s",
                report["file"],
                report["changed_records"],
                report["garbage_reference_records_before"],
                report["garbage_reference_records_after"],
            )
    extraction_reports = run_extraction_audits(target_docs, processed_dir, chunks_dir, report_dir)

    should_run_deferred_qa = args.defer_qa_until_all_extracted or args.skip_extraction
    if not args.audit_only and not args.skip_qa and should_run_deferred_qa:
        print("\n[4/6] Sinh/repair QA theo chunk còn thiếu hoặc QA cũ không đạt chuẩn...")
        for filename in target_docs:
            input_file = processed_dir / f"{filename}.extracted.json"
            output_file = qa_dir / f"{filename}.qa.json"
            if input_file.exists():
                await process_generate_qa(str(input_file), str(output_file))
            else:
                logger.warning("Missing extracted JSON for QA: %s", input_file)
    elif not args.audit_only and not args.skip_qa:
        print("\n[4/6] QA đã được sinh/repair ngay sau từng tài liệu trong bước extraction.")
    else:
        print("\n[4/6] Bỏ qua QA generation.")

    print("\n[5/6] Audit QA coverage, quote, intent/difficulty và độ trùng lặp...")
    qa_reports = run_qa_audits(target_docs, qa_dir, chunks_dir, processed_dir, report_dir)
    print_pdf_summary(pdf_reports)
    print_audit_summary(extraction_reports, qa_reports)

    if not args.skip_graph:
        processed_files = [
            processed_dir / f"{filename}.extracted.json"
            for filename in target_docs
            if (processed_dir / f"{filename}.extracted.json").exists()
        ]
        graph_out = graph_dir / ("legal_graph.json" if not args.target else f"{args.target}_graph.json")
        graph = export_graph(processed_files, graph_out)
        logger.info("Exported graph JSON: nodes=%s edges=%s path=%s", len(graph["nodes"]), len(graph["edges"]), graph_out)
    else:
        graph_out = graph_dir / "legal_graph.json"

    should_build_rag = not args.audit_only and not args.skip_rag_index and (not args.target or args.build_rag_index or args.force_rag_index)
    if should_build_rag:
        print("\n[6/6] Rebuild RAG vector DB từ extracted records, bảng và ảnh đã liên kết...")
        with file_lock(lock_dir / "rag_index.lock", args.lock_timeout):
            rebuild_rag_index(processed_dir, graph_path=graph_out, force=args.force_rag_index)
    else:
        print("\n[6/6] Bỏ qua RAG vector DB rebuild. Khi chạy song song từng file, hãy build index một lần sau cùng.")

    if args.sync_rag_stores:
        print("\n[7/7] Sync dữ liệu sang PostgreSQL, Qdrant, Neo4j và MinIO...")
        from src.data_pipeline.rag_store_sync import (
            MinioAssetRepository,
            Neo4jGraphRepository,
            PostgresLegalRepository,
            _asset_paths,
        )
        from src.rag.rag_store_config import RAGStoreConfig
        from src.rag.qdrant_vector_store import QdrantLegalVectorStore
        from src.rag.record_expander import load_expanded_records, load_processed_records
        from src.rag.traffic_sign_catalog import TrafficSignCatalog

        config = RAGStoreConfig()
        canonical_records = load_processed_records(processed_dir)
        expanded_records = load_expanded_records(processed_dir)
        catalog = TrafficSignCatalog(expanded_records)
        PostgresLegalRepository(config).sync(canonical_records, expanded_records, catalog)
        QdrantLegalVectorStore(processed_path=processed_dir, force_reindex=True, config=config)
        Neo4jGraphRepository(config).sync_graph(graph_out)
        MinioAssetRepository(config).sync_assets(_asset_paths(expanded_records, catalog))

    print(f"\n[HOÀN TẤT] Reports nằm ở: {report_dir}")


if __name__ == "__main__":
    asyncio.run(main())
