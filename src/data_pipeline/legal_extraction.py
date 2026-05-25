import asyncio
import glob
import json
import os
import re
import logging
import hashlib
import time
from dotenv import load_dotenv

from src.data_pipeline.parsers.router import DocumentRouter
from src.data_pipeline.pdf_engine import PDFEngine
from src.data_pipeline.figure_extractor import FigureExtractor
from src.data_pipeline.text_normalizer import TextNormalizer
from src.data_pipeline.coverage_validator import CoverageValidator
from src.data_pipeline.reference_sanitizer import sanitize_record_reference
from src.rag.model_policy import first_text_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExpertExtraction")
load_dotenv(override=True)

MONEY_RE = re.compile(r'(\d{1,3}(?:\.\d{3})*)\s*(?:đồng|VNĐ|VND)', re.IGNORECASE)
SIGN_CODE_RE = re.compile(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", re.IGNORECASE)
EXTRACTION_CACHE_VERSION = "pdf_engine_v9_control_space_repair"
CHUNK_CACHE_VERSION = "legal_parser_v15_qcvn_caption_spacing"

def verify_penalties(rule_dict: dict, source_text: str) -> list[str]:
    """So sánh số tiền AI trích với số tiền trong text gốc."""
    warnings = []
    text_amounts = set()
    for m in MONEY_RE.findall(source_text):
        try:
            text_amounts.add(int(m.replace(".", "")))
        except ValueError:
            continue
    
    p = rule_dict.get("penalties", {}) or {}
    main = p.get("main_penalty", {}) or {}
    
    fields = ["min_amount_vnd", "max_amount_vnd", "individual_min_vnd", 
              "individual_max_vnd", "organization_min_vnd", "organization_max_vnd"]
    
    for field in fields:
        val = main.get(field)
        if val and text_amounts and val not in text_amounts:
            warnings.append(f"PENALTY_MISMATCH: {field}={val} không thấy trong text")
    return warnings

def check_penalty_integrity(records: list) -> list[dict]:
    """Kiểm tra logic nghiệp vụ: min <= max, tổ chức ~ 2x cá nhân."""
    issues = []
    for rec in records:
        p = (rec.get("penalties") or {}).get("main_penalty") or {}
        
        for mn_f, mx_f in [("min_amount_vnd", "max_amount_vnd"), 
                           ("individual_min_vnd", "individual_max_vnd"), 
                           ("organization_min_vnd", "organization_max_vnd")]:
            mn, mx = p.get(mn_f), p.get(mx_f)
            if mn and mx and mn > mx:
                issues.append({"id": rec.get("id"), "error": f"{mn_f} > {mx_f}"})
        
        i_max = p.get("individual_max_vnd")
        o_max = p.get("organization_max_vnd")
        if i_max and o_max and abs(o_max - 2 * i_max) > i_max * 0.15:
            issues.append({"id": rec.get("id"), "warning": "Tỉ lệ phạt tổ chức/cá nhân bất thường"})
    return issues

def is_penalty_chunk(text: str) -> bool:
    keywords = ["phạt", "xử phạt", "tiền", "tước", "trừ điểm", "đối với", "hành vi"]
    return any(k in text.lower() for k in keywords)

def validate_rule(rule, doc_name: str) -> bool:
    ref = getattr(rule, "legal_reference", None)
    if not ref: return False
    if hasattr(rule, "violation_content") and rule.violation_content: return True
    if hasattr(rule, "original_text") and rule.original_text: return True
    if hasattr(rule, "meaning_and_usage") and rule.meaning_and_usage: return True
    return False

def save_to_standard_json(path: str, records: list):
    data = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Could not load existing extraction JSON %s: %s", path, exc)

    def record_priority(record: dict) -> int:
        record_type = record.get("record_type")
        if record_type == "enriched_rule":
            return 3
        if record_type == "source_legal_unit":
            return 1
        # Older files may not have record_type even though they came from LLM output.
        # Keep them above raw fallback, but allow a new explicit enriched record to replace them.
        return 2

    # Store records in a dict keyed by ID to deduplicate.
    # Prefer explicit enriched records, then legacy enriched-looking records, then raw source fallback.
    lookup = {r.get('id'): r for r in data if 'id' in r}
    for rec in records:
        rec_id = rec.get('id')
        if rec_id:
            existing = lookup.get(rec_id)
            if existing:
                if record_priority(rec) > record_priority(existing):
                    lookup[rec_id] = rec
            else:
                lookup[rec_id] = rec

    final_data = list(lookup.values())
    enriched_chunk_ids = {
        r.get("source_chunk_id")
        for r in final_data
        if r.get("source_chunk_id") and r.get("record_type") != "source_legal_unit"
    }
    final_data = [
        r for r in final_data
        if not (
            r.get("record_type") == "source_legal_unit"
            and (r.get("metadata") or {}).get("enrichment_status") != "source_table_preserved"
            and r.get("source_chunk_id") in enriched_chunk_ids
        )
    ]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def _load_json_records(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning("Could not load processed checkpoint %s: %s", path, exc)
        return []

def _resume_record_paths(out_path: str) -> list[str]:
    paths = []
    if os.path.exists(out_path):
        paths.append(out_path)
    backups = sorted(
        glob.glob(out_path + ".bak.*"),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    paths.extend(backups)
    return paths

def _chunk_hashes(chunks: list[dict]) -> dict[str, str]:
    hashes = {}
    for chunk in chunks:
        chunk_id = chunk.get("source_chunk_id")
        if chunk_id:
            hashes[chunk_id] = hashlib.sha256(chunk.get("text", "").encode("utf-8")).hexdigest()
    return hashes

def _record_matches_current_chunks(record: dict, current_hashes: dict[str, str]) -> bool:
    chunk_id = record.get("source_chunk_id")
    if not chunk_id or chunk_id not in current_hashes:
        return False
    source_hash = record.get("source_text_sha256")
    return not source_hash or source_hash == current_hashes[chunk_id]

def load_resume_records(out_path: str, chunks: list[dict]) -> list[dict]:
    """Load resumable processed records from the live output and its .bak checkpoints."""
    current_hashes = _chunk_hashes(chunks)
    records = []
    seen_ids = set()
    paths = _resume_record_paths(out_path)
    for path in paths:
        for record in _load_json_records(path):
            if not isinstance(record, dict):
                continue
            if not _record_matches_current_chunks(record, current_hashes):
                continue
            record_id = record.get("id") or f"{record.get('source_chunk_id')}:{len(records)}"
            if record_id in seen_ids:
                continue
            seen_ids.add(record_id)
            records.append(record)

    if paths:
        covered = len({r.get("source_chunk_id") for r in records if r.get("source_chunk_id")})
        logger.info(
            " - Loaded %s resumable records from %s checkpoint file(s), covering %s/%s chunks.",
            len(records),
            len(paths),
            covered,
            len(current_hashes),
        )
    return records

def _processed_meta_path(out_path: str) -> str:
    return out_path + ".meta.json"

def _processed_versions() -> dict:
    return {
        "extraction_cache_version": EXTRACTION_CACHE_VERSION,
        "chunk_cache_version": CHUNK_CACHE_VERSION,
    }

def ensure_processed_output_version(out_path: str) -> None:
    if not os.path.exists(out_path):
        return
    meta_path = _processed_meta_path(out_path)
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if all(meta.get(k) == v for k, v in _processed_versions().items()):
            return
    except Exception:
        pass

    logger.warning(
        "Processed output metadata is missing or stale; keeping %s in place for resume "
        "and validating records by source_chunk_id/source_text_sha256.",
        out_path,
    )

def write_processed_meta(out_path: str, doc_name: str, chunk_count: int) -> None:
    meta = {
        **_processed_versions(),
        "doc_name": doc_name,
        "chunk_count": chunk_count,
        "timestamp": time.time(),
    }
    with open(_processed_meta_path(out_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_figure_assets(project_root: str, pdf_path: str) -> list[dict]:
    doc_base_name = os.path.basename(pdf_path).replace(".pdf", "")
    meta_path = os.path.join(project_root, "data/processed/sign_assets", f"{doc_base_name}_meta.json")
    if not os.path.exists(meta_path): return []
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load figure metadata {meta_path}: {e}")
        return []

def figures_for_chunk(figures: list[dict], chunk: dict) -> list[dict]:
    if not figures: return []
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start is None: return []
    if end is None: end = start

    text = chunk.get("text") or ""
    chunk_codes = {
        re.sub(r"[\s.]+", "", m.group(0)).upper()
        for m in SIGN_CODE_RE.finditer(text)
    }
    related = []
    seen = set()
    for fig in figures:
        if not (start <= fig.get("page", -1) <= end):
            continue
        fig_code = re.sub(r"[\s.]+", "", str(fig.get("code", ""))).upper()
        if chunk_codes and fig_code and fig_code not in chunk_codes:
            continue
        key = (fig.get("id"), fig_code, fig.get("image_path"))
        if key in seen:
            continue
        seen.add(key)
        fig_copy = fig.copy()
        fig_copy["linked_chunk_id"] = chunk.get("source_chunk_id")
        fig_copy["linked_article"] = chunk.get("article_num")
        related.append(fig_copy)
    return related

def attach_virtual_sign_figures(chunk: dict) -> dict:
    """Attach page-image fallbacks for sign codes when exact crop assets are unavailable."""
    text = chunk.get("text") or ""
    codes = sorted({re.sub(r"\s+", "", m.group(0)).upper() for m in SIGN_CODE_RE.finditer(text)})
    if not codes:
        return chunk

    existing = chunk.get("figures") or []
    existing_codes = {re.sub(r"\s+", "", str(fig.get("code", ""))).upper() for fig in existing if isinstance(fig, dict)}
    page = chunk.get("page_start")
    page_image = chunk.get("image_path") or ""
    fallbacks = []
    for code in codes:
        if code in existing_codes:
            continue
        fallback_id = f"{chunk.get('source_chunk_id', 'chunk')}_sign_{code}".replace(".", "_")
        fallbacks.append({
            "id": fallback_id,
            "code": code,
            "name": "",
            "doc_name": chunk.get("doc_name", ""),
            "page": page,
            "image_path": page_image,
            "source": "page_image_fallback",
            "linked_chunk_id": chunk.get("source_chunk_id"),
            "linked_article": chunk.get("article_num"),
            "caption": f"Mã biển/vạch {code} được nhắc trong đoạn luật; dùng ảnh trang gốc để đối chiếu nếu chưa có crop riêng.",
        })
    chunk["figures"] = existing + fallbacks
    chunk["sign_codes"] = sorted(set(codes))
    return chunk

def _table_key(table: dict) -> tuple:
    return (int(table.get("page", -1)), str(table.get("id") or ""))

def _nearest_legal_context(chunks: list[dict], page: int) -> dict:
    candidates = [
        c for c in chunks
        if c.get("page_start") is not None
        and int(c.get("page_start")) <= page
        and (c.get("article_num") or c.get("chapter_num"))
    ]
    if not candidates:
        return {}
    return max(candidates, key=lambda c: (int(c.get("page_start") or 0), int(c.get("page_end") or c.get("page_start") or 0)))

def append_unlinked_table_chunks(chunks: list[dict], doc_map: dict, doc_name: str) -> list[dict]:
    """Preserve every detected table as a retrievable unit, even if layout linking misses it."""
    linked_tables = {
        _table_key(table)
        for chunk in chunks
        for table in (chunk.get("tables") or [])
        if isinstance(table, dict)
    }
    table_chunks = []

    for page_key, page_data in sorted(doc_map.items(), key=lambda item: int(item[0])):
        page = int(page_key)
        for table in page_data.get("tables") or []:
            if _table_key(table) in linked_tables:
                continue

            table_text = TextNormalizer.normalize_vietnamese(table.get("text") or "")
            if not table_text and not (table.get("image_path") or table.get("bbox")):
                continue
            if not table_text:
                table_text = f"[Bảng/hình bảng trang {page + 1}; cần truy xuất ảnh crop để đọc chi tiết]"

            context = _nearest_legal_context(chunks, page)
            hash_source = json.dumps({"page": page, "id": table.get("id"), "text": table_text, "bbox": table.get("bbox")}, ensure_ascii=False)
            short_hash = hashlib.sha256(hash_source.encode("utf-8")).hexdigest()[:8]
            table_id = str(table.get("id") or f"p{page}_table")
            source_chunk_id = f"{doc_name}_table_{page}_{table_id}_{short_hash}".lower().replace(" ", "_").replace(".", "")
            context_label = ""
            if context.get("article_num"):
                context_label = f"Điều {context.get('article_num')}"
                if context.get("article_title"):
                    context_label += f". {context.get('article_title')}"

            table_chunks.append({
                "doc_name": doc_name,
                "source_file": "",
                "kind": "table",
                "text": f"{context_label}\n[BẢNG {table_id} - trang {page + 1}]\n{table_text}".strip(),
                "chapter_num": context.get("chapter_num", ""),
                "article_num": context.get("article_num", ""),
                "article_title": context.get("article_title", ""),
                "clause_num": context.get("clause_num", ""),
                "point_key": context.get("point_key", ""),
                "tables": [table],
                "source_chunk_id": source_chunk_id,
                "page_start": page,
                "page_end": page,
                "source_body_exact": table_text,
                "semantic_context": context.get("semantic_context", context_label),
                "parent_hierarchy": context.get("parent_hierarchy", []),
                "image_path": table.get("image_path") or page_data.get("img", ""),
                "is_sign_page": page_data.get("is_sign_page", False),
                "figures": [],
                "is_table_only": True,
            })

    if table_chunks:
        logger.info(" - Added %s table-only chunks for unlinked tables.", len(table_chunks))
    return chunks + table_chunks

def make_deterministic_record(chunk: dict, doc_name: str, image_path: str = "", figures: list[dict] | None = None) -> dict:
    text = chunk.get("text", "")
    tables = chunk.get("tables", []) or []
    figures = figures or []
    source_chunk_id = chunk.get("source_chunk_id", "")
    raw_id = source_chunk_id or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    record_id = "SRC_" + re.sub(r"[^0-9A-Za-zÀ-ỹ_]+", "_", raw_id).strip("_")
    return {
        "id": record_id,
        "record_type": "source_legal_unit",
        "legal_reference": {
            "document": doc_name,
            "chapter": chunk.get("chapter_num", ""),
            "article": chunk.get("article_num", ""),
            "clause": chunk.get("clause_num", ""),
            "point": chunk.get("point_key", ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
        },
        "metadata": {
            "enrichment_status": "pending_or_failed",
            "chunk_kind": chunk.get("kind", ""),
        },
        "violation_content": text,
        "original_text": text,
        "meaning_and_usage": text,
        "source_chunk_id": source_chunk_id,
        "doc_name": doc_name,
        "image_path": image_path,
        "tables": tables,
        "table_refs": [t.get("id") for t in tables if isinstance(t, dict) and t.get("id")],
        "figures": figures,
        "figure_refs": [f.get("id") for f in figures if f.get("id")],
        "content": text,
        "source_body_exact": text,
        "semantic_context": chunk.get("semantic_context", ""),
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "parent_hierarchy": chunk.get("parent_hierarchy", []),
        "chunk_meta": {
            "chapter_num": chunk.get("chapter_num", ""),
            "article_num": chunk.get("article_num", ""),
            "clause_num": chunk.get("clause_num", ""),
            "point_key": chunk.get("point_key", ""),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "kind": chunk.get("kind", ""),
        }
    }

async def stage_extract(pdf_path: str, doc_name: str, interim_dir: str) -> dict:
    """Stage 1: PDF to Raw TXT per page (L2)."""
    doc_base = os.path.basename(pdf_path).replace(".pdf", "")
    page_dir = os.path.join(interim_dir, doc_base)
    os.makedirs(page_dir, exist_ok=True)
    
    manifest_path = os.path.join(page_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
                if manifest.get("extraction_cache_version") == EXTRACTION_CACHE_VERSION:
                    return PDFEngine()._strip_headers_footers(PDFEngine()._apply_text_filters(manifest.get("doc_map", {})))
                logger.info(f" - Extraction manifest is stale; rebuilding page map for {doc_base}.")
        except Exception as exc:
            logger.warning("Could not read extraction manifest %s: %s", manifest_path, exc)

    engine = PDFEngine()
    doc_map = await engine.convert_to_markdown_simple(pdf_path, doc_name=doc_name)
    
    for p_idx, p_data in doc_map.items():
        with open(os.path.join(page_dir, f"page_{int(p_idx):03d}.txt"), "w", encoding="utf-8") as f:
            f.write(p_data.get("raw", ""))
            
    manifest = {
        "doc_name": doc_name, "timestamp": time.time(), "engine": "PDFEngine_Multi",
        "extraction_cache_version": EXTRACTION_CACHE_VERSION,
        "page_count": len(doc_map), "doc_map": doc_map
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        
    return doc_map

def stage_chunk(doc_map: dict, doc_name: str, pdf_path: str, chunks_dir: str, figure_assets: list = None) -> list[dict]:
    """Stage 2: Parse text into deterministic legal chunks (L3)."""
    doc_base = os.path.basename(pdf_path).replace(".pdf", "")
    chunks_path = os.path.join(chunks_dir, f"{doc_base}.chunks.jsonl")
    chunks_meta_path = chunks_path + ".meta.json"
    
    if os.path.exists(chunks_path):
        cache_ok = False
        if os.path.exists(chunks_meta_path):
            try:
                with open(chunks_meta_path, "r", encoding="utf-8") as f:
                    cache_ok = json.load(f).get("chunk_cache_version") == CHUNK_CACHE_VERSION
            except Exception:
                cache_ok = False
        if cache_ok:
            chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f: chunks.append(json.loads(line))
            return chunks
        logger.info(f" - Chunk cache is stale or unversioned; reparsing {doc_base}.")

    full_corrected = "\n\n".join([p['corrected'] for p in sorted(doc_map.values(), key=lambda x: x.get('page_meta', {}).get('page', 0))])
    parser = DocumentRouter.get_parser(os.path.basename(pdf_path), full_corrected, doc_name)
    chunks = parser.parse(full_corrected, doc_name, doc_map)
    
    for c in chunks:
        c['figures'] = figures_for_chunk(figure_assets, c)
        attach_virtual_sign_figures(c)
    chunks = append_unlinked_table_chunks(chunks, doc_map, doc_name)
    
    os.makedirs(chunks_dir, exist_ok=True)
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(chunks_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "chunk_cache_version": CHUNK_CACHE_VERSION,
            "doc_name": doc_name,
            "chunk_count": len(chunks),
            "timestamp": time.time(),
        }, f, ensure_ascii=False, indent=2)
            
    return chunks

async def stage_enrich(chunks: list[dict], doc_name: str, pdf_path: str, out_path: str):
    """Stage 3: LLM enrichment (L4)."""
    ensure_processed_output_version(out_path)
    processed_chunk_ids = set()
    garbage_terms = ["không xác định", "n/a", "none", "unknown", "khh", "không biết", "nghị định liên quan"]
    retry_source_fallback = os.getenv("RETRY_SOURCE_FALLBACK_CHUNKS", "false").lower() in {"1", "true", "yes", "y"}
    
    existing = load_resume_records(out_path, chunks)
    if existing:
        # Restore valid records from .bak checkpoints into the live file before continuing.
        save_to_standard_json(out_path, existing)

    for item in existing:
        ref = item.get('legal_reference', {})

        def is_garbage(v):
            return not v or any(g in str(v).lower() for g in garbage_terms)

        is_broken = (
            is_garbage(item.get('doc_name')) or
            is_garbage(ref.get('document')) or
            (is_garbage(ref.get('article')) and is_garbage(ref.get('clause')) and is_garbage(ref.get('point'))) or
            is_garbage(item.get('violation_content'))
        )

        record_type = item.get('record_type')
        is_enriched = record_type == 'enriched_rule'
        is_source_fallback = record_type == 'source_legal_unit'
        is_preserved_table = (
            is_source_fallback
            and (item.get("metadata") or {}).get("enrichment_status") == "source_table_preserved"
        )

        # Source fallback records preserve chunks and prevent failed LLM calls from retrying forever.
        if item.get('source_chunk_id') and (
            (is_enriched and not is_broken)
            or is_preserved_table
            or (is_source_fallback and not retry_source_fallback)
        ):
            processed_chunk_ids.add(item['source_chunk_id'])

    dummy_text = "\n".join([c.get('text', '') for c in chunks[:5]])
    parser = DocumentRouter.get_parser(os.path.basename(pdf_path), dummy_text, doc_name)
    pending = [c for c in chunks if c.get('source_chunk_id') not in processed_chunk_ids]
    
    if not pending: 
        logger.info(f" - All chunks for {doc_name} are already successfully enriched.")
        write_processed_meta(out_path, doc_name, len(chunks))
        return

    logger.info(f" - Pending chunks to enrich/repair: {len(pending)}")
    for i, c in enumerate(pending):
        if c.get("is_table_only") or c.get("kind") == "table":
            fallback = make_deterministic_record(c, doc_name, image_path=c.get('image_path', ''), figures=c.get('figures', []))
            fallback["metadata"]["enrichment_status"] = "source_table_preserved"
            fallback["metadata"]["reason"] = "table-only chunk preserved for RAG/table retrieval without LLM mutation"
            save_to_standard_json(out_path, [fallback])
            logger.info(f" - Preserved table chunk without LLM: {i+1}/{len(pending)} for {doc_name}")
            continue

        structured_data = await asyncio.to_thread(parser.extract_structured_data, c, doc_name)
        chunk_records = []
        
        if structured_data and hasattr(structured_data, 'rules') and structured_data.rules:
            for rule in structured_data.rules:
                rule_dict = rule.model_dump()
                
                ref = rule_dict.get('legal_reference', {})
                def clean_val(v, fallback=""):
                    if not v or any(x in str(v).lower() for x in ["không xác định", "n/a", "none", "unknown", "null", "khh", "không biết"]):
                        return fallback
                    return str(v).strip()

                ref['document'] = doc_name
                ref['article'] = clean_val(ref.get('article'), c.get('article_num', ''))
                ref['clause'] = clean_val(ref.get('clause'), c.get('clause_num', ''))
                ref['point'] = clean_val(ref.get('point'), c.get('point_key', ''))
                ref['chapter'] = clean_val(ref.get('chapter'), c.get('chapter_num', ''))
                ref['page_start'] = c.get('page_start')
                ref['page_end'] = c.get('page_end')
                
                for k in ['article', 'clause', 'point', 'chapter']:
                    if k in ref and clean_val(ref.get(k)) == "":
                        ref[k] = ""
                
                rule_dict['legal_reference'] = ref
                rule_dict['doc_name'] = doc_name
                rule_dict['record_type'] = "enriched_rule"
                sanitize_record_reference(rule_dict, c, doc_name)
                
                if not clean_val(rule_dict.get('violation_content')):
                    rule_dict['violation_content'] = c.get('text', '')[:1000]

                safe_id = rule_dict.get('id', '')
                chunk_id_suffix = str(c.get('source_chunk_id', ''))[-8:]
                
                rule_dict.update({
                    "id": f"{safe_id}_{chunk_id_suffix}".strip("_"),
                    "source_chunk_id": c.get('source_chunk_id'),
                    "source_body_exact": c.get('text'),
                    "semantic_context": c.get('semantic_context', ''),
                    "source_text_sha256": hashlib.sha256(c.get('text', '').encode("utf-8")).hexdigest(),
                    "parent_hierarchy": c.get('parent_hierarchy', []),
                    "tables": c.get('tables', []),
                    "figures": c.get('figures', []),
                    "image_path": c.get('image_path', ''),
                    "extraction_meta": {
                        "engine": first_text_model("EXTRACTION_MODEL", "EXTRACTION_PRIMARY_MODEL"),
                        "timestamp": time.time(),
                        "confidence": 0.95,
                        "warnings": verify_penalties(rule_dict, c.get('text', ''))
                    }
                })
                chunk_records.append(rule_dict)
        
        if not chunk_records:
            # Fallback to source record if LLM fails completely, but still sanitize
            fallback = make_deterministic_record(c, doc_name, image_path=c.get('image_path', ''), figures=c.get('figures', []))
            chunk_records.append(fallback)
            
        save_to_standard_json(out_path, chunk_records)
        logger.info(f" - Progress: {i+1}/{len(pending)} for {doc_name}")
        if i % 3 == 0: await asyncio.sleep(2)
    write_processed_meta(out_path, doc_name, len(chunks))

async def process_document(pdf_path: str, doc_name: str, out_dir: str, doc_sem: asyncio.Semaphore):
    async with doc_sem:
        logger.info(f"🚀 Processing: {doc_name}")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        interim_dir = os.path.join(project_root, "data/interim")
        chunks_dir = os.path.join(project_root, "data/chunks")
        
        if "QCVN" in doc_name:
            try:
                fig_extractor = FigureExtractor(project_root)
                fig_extractor.extract_figures(pdf_path, doc_name)
            except Exception as e: logger.error(f"Figure extraction failed: {e}")
        
        figure_assets = load_figure_assets(project_root, pdf_path)
        doc_map = await stage_extract(pdf_path, doc_name, interim_dir)
        chunks = stage_chunk(doc_map, doc_name, pdf_path, chunks_dir, figure_assets=figure_assets)
        out_path = os.path.join(out_dir, f"{os.path.basename(pdf_path)}.extracted.json")
        await stage_enrich(chunks, doc_name, pdf_path, out_path)

        try:
            full_raw_text = "\n\n".join([p['raw'] for p in sorted(doc_map.values(), key=lambda x: x.get('page_meta', {}).get('page', 0))])
            with open(out_path, "r", encoding="utf-8") as f:
                all_extracted = json.load(f)
            validator = CoverageValidator()
            report = validator.validate(all_extracted, full_raw_text, chunks)
            logger.info(f"📊 Legal-Grade Coverage Report for {doc_name}:")
            logger.info(f"  - Status: {report['status']} (Overall Score: {report['overall_score']*100:.1f}%)")
            for level, data in report['levels'].items():
                if data["expected_count"]:
                    logger.info(
                        "  - %s: %s/%s covered (%.1f%%), extracted unique=%s",
                        level.capitalize(),
                        data["covered_count"],
                        data["expected_count"],
                        data["score"] * 100,
                        data["extracted_count"],
                    )
                else:
                    logger.info(
                        "  - %s: extracted unique=%s (no raw-text expectation)",
                        level.capitalize(),
                        data["extracted_count"],
                    )
            chunk_cov = report.get("source_chunk_coverage")
            if chunk_cov:
                logger.info(
                    "  - Source chunks: %s/%s (%.1f%%), coordinates: %s/%s (%.1f%%), source-only fallback chunks: %s",
                    chunk_cov["covered_chunk_count"],
                    chunk_cov["expected_chunk_count"],
                    chunk_cov["chunk_coverage_score"] * 100,
                    chunk_cov["covered_coordinate_count"],
                    chunk_cov["expected_coordinate_count"],
                    chunk_cov["coordinate_coverage_score"] * 100,
                    chunk_cov["source_only_chunk_count"],
                )
            integrity_issues = check_penalty_integrity(all_extracted)
            if integrity_issues: logger.warning(f"  - ⚠️ Found {len(integrity_issues)} business integrity issues")
        except Exception as e: logger.error(f"Coverage validation failed: {e}")

async def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    out_dir = os.path.join(project_root, "data/processed")
    raw_dir = os.path.join(project_root, "data/raw")
    os.makedirs(out_dir, exist_ok=True)
    target_files = {
        "168-nd-cp.signed.pdf": "Nghị định 168/2024/NĐ-CP",
        "336-2025-nd-cp-22122025-signed-17665482569851736009102.pdf": "Nghị định 336/2025/NĐ-CP",
        "35-2024-qh15.pdf": "Luật Đường bộ 2024",
        "35-bgtvt.pdf": "Thông tư 35/2024/TT-BGTVT",
        "36-2024-qh15.pdf": "Luật Trật tự ATGT 2024",
        "36-2024-qh15_tiep.pdf": "Luật Trật tự ATGT 2024 (Tiếp)",
        "51-bgtvt-kem.pdf": "QCVN 41:2024 (Thông tư 51/2024)"
    }
    doc_sem = asyncio.Semaphore(1) 
    tasks = [process_document(os.path.join(raw_dir, f), d, out_dir, doc_sem) for f, d in target_files.items()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
