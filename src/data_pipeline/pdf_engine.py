import os
import json
import logging
import asyncio
import fitz
import pdfplumber
import hashlib
import re
from src.data_pipeline.text_normalizer import TextNormalizer
try:
    from llama_parse import LlamaParse
except ImportError:
    LlamaParse = None

logger = logging.getLogger("PDFEngine")

class PDFEngine:
    def __init__(self, client=None):
        self.parser = client

    def _get_llamaparse(self):
        if self.parser:
            return self.parser
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key or LlamaParse is None:
            return None
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            language="vi",
            user_prompt=(
                "Đây là văn bản pháp luật Việt Nam. Hãy OCR chính xác, giữ cấu trúc "
                "Điều/Khoản/Điểm, tiêu đề, bảng biểu và chú thích hình. Không cần diễn giải."
            )
        )
        return self.parser

    def inspect_pdf(self, pdf_path: str) -> dict:
        """Collect page-level quality signals before choosing an extraction engine."""
        try:
            doc = fitz.open(pdf_path)
            page_stats = []
            for i in range(len(doc)):
                page = doc[i]
                text = page.get_text("text").strip()
                page_stats.append({
                    "page": i,
                    "text_len": len(text),
                    "word_count": len(text.split()),
                    "embedded_images": len(page.get_images(full=True)),
                })
            doc.close()
            low_text_pages = sum(1 for p in page_stats if p["text_len"] < 80)
            image_heavy_pages = sum(1 for p in page_stats if p["embedded_images"] > 0 and p["text_len"] < 250)
            total_pages = len(page_stats) or 1
            return {
                "pages": total_pages,
                "page_stats": page_stats,
                "low_text_ratio": low_text_pages / total_pages,
                "image_heavy_ratio": image_heavy_pages / total_pages,
                "total_text_len": sum(p["text_len"] for p in page_stats),
                "total_embedded_images": sum(p["embedded_images"] for p in page_stats),
            }
        except Exception as e:
            logger.error(f"Error inspecting PDF: {e}")
            return {"pages": 0, "page_stats": [], "low_text_ratio": 1.0, "image_heavy_ratio": 1.0}

    def is_scanned_pdf(self, pdf_path: str, doc_name: str = "") -> bool:
        """Mandatory OCR only if the document is consistently poor quality.

        Some official PDFs contain embedded images/signatures on many pages while still
        having a good text layer. In that case LlamaParse can degrade Vietnamese legal
        text badly, so prefer PyMuPDF when the embedded text volume is strong. QCVN/sign
        appendices are the exception because the visual layer is part of the legal data.
        """
        audit = self.inspect_pdf(pdf_path)
        pages = audit.get("pages") or 1
        avg_text_len = audit.get("total_text_len", 0) / pages
        text_rich_pages = sum(1 for p in audit.get("page_stats", []) if p.get("text_len", 0) >= 500)
        text_rich_ratio = text_rich_pages / pages
        images_per_page = audit.get("total_embedded_images", 0) / pages
        name_signal = f"{pdf_path} {doc_name}".lower()
        sign_heavy_document = (
            ("qcvn" in name_signal or "51-bgtvt" in name_signal)
            and audit.get("image_heavy_ratio", 0) >= 0.20
            and images_per_page >= 5
        )

        if sign_heavy_document:
            return True

        if avg_text_len >= 500 or text_rich_ratio >= 0.30:
            return False
        return audit["low_text_ratio"] > 0.15 or audit["image_heavy_ratio"] > 0.25

    def _is_sign_heavy_document(self, pdf_path: str, doc_name: str, audit: dict) -> bool:
        pages = audit.get("pages") or 1
        images_per_page = audit.get("total_embedded_images", 0) / pages
        name_signal = f"{pdf_path} {doc_name}".lower()
        return (
            ("qcvn" in name_signal or "51-bgtvt" in name_signal)
            and audit.get("image_heavy_ratio", 0) >= 0.20
            and images_per_page >= 5
        )

    def _is_mixed_text_document(self, pdf_path: str, doc_name: str, audit: dict) -> bool:
        if self._is_sign_heavy_document(pdf_path, doc_name, audit):
            return False
        pages = audit.get("pages") or 1
        text_rich_pages = sum(1 for p in audit.get("page_stats", []) if p.get("text_len", 0) >= 500)
        text_rich_ratio = text_rich_pages / pages
        return text_rich_ratio >= 0.30 and audit.get("low_text_ratio", 0) >= 0.15

    async def _load_or_parse_llamaparse_map(self, pdf_path: str, audit: dict) -> dict:
        cache_path = pdf_path + ".llamaparse.v4.json"
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return self._strip_headers_footers(self._apply_text_filters(json.load(f)))

        parser = self._get_llamaparse()
        if not parser:
            return {}

        logger.info(f" - [OCR FALLBACK] LlamaParse: {os.path.basename(pdf_path)}")
        pages_map = {}
        documents = await parser.aload_data(pdf_path)
        for i, doc in enumerate(documents):
            txt = f"\n[INTERNAL_PAGE_MARKER_{i}]\n" + doc.text
            norm = TextNormalizer.process(txt)
            pages_map[str(i)] = {
                "raw": txt,
                "corrected": norm["normalized"],
                "raw_sha256": norm["raw_sha256"],
                "norm_sha256": norm["norm_sha256"],
                "img": "",
                "tables": [],
                "layout": None,
                "is_sign_page": False,
                "page_meta": audit["page_stats"][i] if i < len(audit["page_stats"]) else {"page": i},
                "extraction_engine": "llamaparse",
            }
        pages_map = self._strip_headers_footers(pages_map)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(pages_map, f, ensure_ascii=False, indent=2)
        return pages_map

    def _table_to_text(self, rows: list[list[str | None]]) -> str:
        return "\n".join(
            " | ".join((cell or "").strip() for cell in row)
            for row in rows
            if any((cell or "").strip() for cell in row)
        ).strip()

    def _normalize_table_rows(self, rows: list[list[str | None]]) -> list[list[str]]:
        """Normalize pdfplumber/Camelot table cells while preserving row shape."""
        normalized = []
        max_cols = max((len(row) for row in rows), default=0)
        previous = [""] * max_cols
        for row in rows:
            clean_row = []
            for idx in range(max_cols):
                value = row[idx] if idx < len(row) else ""
                cell = "" if value is None else TextNormalizer.normalize_vietnamese(str(value)).strip()
                if not cell and idx < len(previous):
                    cell = ""
                clean_row.append(cell)
            if any(clean_row):
                normalized.append(clean_row)
                previous = clean_row
        return normalized

    def _infer_table_headers(self, rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
        if len(rows) < 2:
            return [], rows
        first = rows[0]
        filled = [cell for cell in first if cell.strip()]
        if not filled:
            return [], rows
        alpha_cells = sum(1 for cell in filled if re.search(r"[A-Za-zÀ-ỹ]", cell))
        numeric_cells = sum(1 for cell in filled if re.fullmatch(r"[\d\s.,/%-]+", cell))
        looks_like_header = alpha_cells >= max(1, len(filled) // 2) and numeric_cells <= len(filled) // 2
        if not looks_like_header:
            return [], rows
        return first, rows[1:]

    def _build_table_payload(self, *, table_id: str, page_idx: int, bbox: list | None, rows_raw: list, image_path: str, layout: dict | None = None, note: str = "") -> dict:
        rows = self._normalize_table_rows(rows_raw or [])
        headers, body_rows = self._infer_table_headers(rows)
        caption = TextNormalizer.normalize_vietnamese(self._find_nearest_caption(bbox, layout) or "") if bbox and layout else ""
        table_text = self._table_to_text([headers] + body_rows if headers else body_rows)
        payload = {
            "id": table_id,
            "page": page_idx,
            "bbox": bbox,
            "caption": caption,
            "headers": headers,
            "rows": body_rows,
            "text": table_text,
            "image_path": image_path,
        }
        if note:
            payload["note"] = note
        return payload

    def _has_merged_cells(self, table: dict) -> bool:
        rows = table.get("rows", [])
        return any(cell is None for row in rows for cell in row)

    def _crop_table_image(self, pdf_path: str, page_idx: int, bbox: list, out_path: str):
        """Crop và save ảnh bảng với 2× zoom để LLM đọc rõ."""
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_idx)
        rect = fitz.Rect(bbox)
        rect.x0 = max(0, rect.x0 - 4)
        rect.y0 = max(0, rect.y0 - 4)
        rect.x1 = min(page.rect.width,  rect.x1 + 4)
        rect.y1 = min(page.rect.height, rect.y1 + 4)
        pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2))
        pix.save(out_path)
        doc.close()

    def _extract_tables(self, pdf_path: str, table_img_dir: str) -> dict[int, list[dict]]:
        page_tables: dict[int, list[dict]] = {}
        doc_base_name = os.path.basename(pdf_path).replace(".pdf", "")
        pages_needing_camelot = []
        fitz_doc = fitz.open(pdf_path)

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages):
                if (page_idx + 1) % 20 == 0 or page_idx == 0:
                    logger.info(f"   - [TABLE PROGRESS] Page {page_idx + 1}/{total_pages} analyzed...")

                tables = []
                page_has_merged_cells = False
                layout = self._clean_bytes(fitz_doc.load_page(page_idx).get_text("dict"))
                try:
                    found_tables = page.find_tables()
                except Exception:
                    found_tables = []

                if found_tables:
                    for table_idx, table in enumerate(found_tables):
                        rows = table.extract() or []
                        if not rows: continue
                        if any(cell is None for row in rows for cell in row):
                            page_has_merged_cells = True
                        bbox = list(table.bbox) if table.bbox else None
                        img_path_rel = ""
                        if bbox:
                            table_id = f"p{page_idx}_t{table_idx}"
                            img_path_abs = os.path.join(table_img_dir, f"{table_id}.png")
                            try:
                                self._crop_table_image(pdf_path, page_idx, bbox, img_path_abs)
                                img_path_rel = os.path.join("data/processed/table_imgs", doc_base_name, f"{table_id}.png")
                            except Exception as exc:
                                logger.debug("Could not crop table image %s page=%s table=%s: %s", pdf_path, page_idx, table_idx, exc)
                            
                        table_id = f"p{page_idx}_t{table_idx}"
                        tables.append(self._build_table_payload(
                            table_id=table_id,
                            page_idx=page_idx,
                            bbox=bbox,
                            rows_raw=rows,
                            image_path=img_path_rel,
                            layout=layout,
                        ))
                
                if page_has_merged_cells:
                    pages_needing_camelot.append(page_idx)
                page_tables[page_idx] = tables
        
        if pages_needing_camelot:
            try:
                import camelot
                pages_str = ",".join([str(p + 1) for p in pages_needing_camelot[:50]])
                logger.info(f"   - [CAMELOT FALLBACK] Processing {len(pages_needing_camelot)} complex pages in batch...")
                camelot_tables_all = camelot.read_pdf(pdf_path, pages=pages_str, flavor='lattice')
                for ct in camelot_tables_all:
                    p_idx = ct.page - 1
                    if p_idx in page_tables:
                        layout = self._clean_bytes(fitz_doc.load_page(p_idx).get_text("dict"))
                        table_id = f"p{p_idx}_ct_{len(page_tables[p_idx])}"
                        page_tables[p_idx].append(self._build_table_payload(
                            table_id=table_id,
                            page_idx=p_idx,
                            bbox=list(ct._bbox) if getattr(ct, "_bbox", None) else None,
                            rows_raw=ct.df.values.tolist(),
                            image_path="",
                            layout=layout,
                            note="Extracted via Camelot Lattice",
                        ))
            except Exception as exc:
                logger.warning("Camelot fallback failed for %s pages=%s: %s", pdf_path, pages_needing_camelot[:50], exc)
        fitz_doc.close()
        return page_tables

    def _find_nearest_caption(self, bbox, layout_dict):
        if not bbox or not layout_dict: return None
        tx0, ty0, tx1, ty1 = bbox
        best_text, min_dist = None, float('inf')
        for block in layout_dict.get("blocks", []):
            if block.get("type") != 0: continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text: continue
                    sx0, sy0, sx1, sy1 = span.get("bbox")
                    dist = min(abs(ty0 - sy1), abs(ty1 - sy0))
                    overlap = max(0, min(tx1, sx1) - max(tx0, sx0))
                    if dist < 50 and (overlap > 0 or abs(tx0 - sx0) < 100):
                        if dist < min_dist:
                            min_dist, best_text = dist, text
        return best_text

    def _clean_bytes(self, obj):
        if isinstance(obj, dict): return {k: self._clean_bytes(v) for k, v in obj.items() if not isinstance(v, bytes)}
        elif isinstance(obj, list): return [self._clean_bytes(i) for i in obj if not isinstance(i, bytes)]
        return obj

    def _apply_text_filters(self, pages_map: dict) -> dict:
        """Apply current text normalization to fresh or cached page maps."""
        for p_data in pages_map.values():
            text = p_data.get("corrected") or p_data.get("raw") or ""
            norm = TextNormalizer.process(text)
            p_data["corrected"] = norm["normalized"]
            p_data["norm_sha256"] = norm["norm_sha256"]
        return pages_map

    def _strip_headers_footers(self, pages_map: dict) -> dict:
        """Loại bỏ số trang và tiêu đề lặp lại (boilerplate) xuất hiện ở >60% số trang.
        Sử dụng kiểm tra 5 dòng đầu và 5 dòng cuối để bắt được boilerplate nhiều dòng."""
        from collections import Counter
        line_counts = Counter()

        for p_data in pages_map.values():
            lines = p_data.get("raw", "").strip().split("\n")
            if not lines: continue

            # Check first 5 and last 5 lines
            header_candidates = lines[:5]
            footer_candidates = lines[-5:]

            for line in header_candidates + footer_candidates:
                clean_line = line.strip()
                if clean_line:
                    line_counts[clean_line] += 1

        threshold = len(pages_map) * 0.6
        boilerplate = {l for l, c in line_counts.items() if c >= threshold and len(l) < 150}

        for p_data in pages_map.values():
            lines = p_data.get("corrected", "").split("\n")
            p_data["corrected"] = "\n".join([l for l in lines if l.strip() not in boilerplate])
        return self._apply_text_filters(pages_map)

    async def convert_to_markdown_simple(self, pdf_path, doc_name: str = ""):
        audit = self.inspect_pdf(pdf_path)
        engine_override = os.getenv("PDF_ENGINE", "auto").strip().lower()
        if engine_override not in {"auto", "fitz", "llamaparse", "hybrid"}:
            logger.warning(f"Unknown PDF_ENGINE={engine_override}; falling back to auto.")
            engine_override = "auto"

        use_hybrid = engine_override == "hybrid" or (engine_override == "auto" and self._is_mixed_text_document(pdf_path, doc_name, audit))
        is_scan = self.is_scanned_pdf(pdf_path, doc_name)
        if engine_override == "fitz":
            is_scan = False
            use_hybrid = False
        elif engine_override == "llamaparse":
            is_scan = True
            use_hybrid = False
        parser = None if use_hybrid else self._get_llamaparse() if is_scan else None
        engine_name = "hybrid" if use_hybrid else "llamaparse" if parser else "fitz"
        cache_path = pdf_path + f".{engine_name}.v4.json"
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        doc_base_name = os.path.basename(pdf_path).replace(".pdf", "")
        img_dir = os.path.join(project_root, "data/processed/images", doc_base_name)
        table_img_dir = os.path.join(project_root, "data/processed/table_imgs", doc_base_name)
        layout_dir = os.path.join(project_root, "data/interim", doc_base_name, "layouts")
        os.makedirs(img_dir, exist_ok=True); os.makedirs(table_img_dir, exist_ok=True); os.makedirs(layout_dir, exist_ok=True)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return self._strip_headers_footers(self._apply_text_filters(json.load(f)))
            except Exception as exc:
                logger.warning("Could not read PDF engine cache %s: %s", cache_path, exc)

        full_doc_map = {}
        if parser:
            logger.info(f" - [SCAN] LlamaParse: {os.path.basename(pdf_path)}")
            full_doc_map = await self._load_or_parse_llamaparse_map(pdf_path, audit)
        else:
            logger.info(f" - [{'HYBRID' if use_hybrid else 'DIGITAL'}] PyMuPDF: {os.path.basename(pdf_path)}")
            doc_fitz = fitz.open(pdf_path)
            for i in range(len(doc_fitz)):
                page = doc_fitz.load_page(i)
                text = page.get_text("text").strip()
                layout = self._clean_bytes(page.get_text("dict"))
                page_stat = audit["page_stats"][i] if i < len(audit["page_stats"]) else {}
                is_sign = (len(text) < 100 and page_stat.get("embedded_images", 0) > 200)
                l_file = f"page_{i:03d}.layout.json"
                with open(os.path.join(layout_dir, l_file), "w", encoding="utf-8") as f: json.dump(layout, f, ensure_ascii=False)
                txt = f"\n[INTERNAL_PAGE_MARKER_{i}]\n" + text
                norm = TextNormalizer.process(txt)
                full_doc_map[str(i)] = {
                    "raw": txt, "corrected": norm["normalized"], "raw_sha256": norm["raw_sha256"], 
                    "norm_sha256": norm["norm_sha256"], "img": "", "tables": [], 
                    "layout_path": os.path.join("data/interim", doc_base_name, "layouts", l_file),
                    "is_sign_page": is_sign, "page_meta": page_stat
                }
            doc_fitz.close()

            if use_hybrid:
                llama_map = await self._load_or_parse_llamaparse_map(pdf_path, audit)
                replaced = 0
                for i, page_stat in enumerate(audit.get("page_stats", [])):
                    key = str(i)
                    if key not in llama_map or key not in full_doc_map:
                        continue
                    fitz_text = full_doc_map[key].get("corrected", "")
                    llama_text = llama_map[key].get("corrected", "")
                    if page_stat.get("text_len", 0) < 250 and len(llama_text) > len(fitz_text) + 100:
                        img = full_doc_map[key].get("img", "")
                        tables = full_doc_map[key].get("tables", [])
                        full_doc_map[key] = llama_map[key]
                        full_doc_map[key]["img"] = img
                        full_doc_map[key]["tables"] = tables
                        full_doc_map[key]["extraction_engine"] = "hybrid:llamaparse"
                        replaced += 1
                    else:
                        full_doc_map[key]["extraction_engine"] = "hybrid:fitz"
                logger.info(f" - [HYBRID] Replaced {replaced} low-text pages with LlamaParse/OCR text.")

        full_doc_map = self._strip_headers_footers(full_doc_map)
        doc_fitz = fitz.open(pdf_path)
        for i in range(len(doc_fitz)):
            pix = doc_fitz.load_page(i).get_pixmap(matrix=fitz.Matrix(2, 2))
            img_name = f"page_{i}.png"
            pix.save(os.path.join(img_dir, img_name))
            if str(i) in full_doc_map: full_doc_map[str(i)]["img"] = os.path.join("data/processed/images", doc_base_name, img_name)
        doc_fitz.close()

        try:
            page_tables = self._extract_tables(pdf_path, table_img_dir)
            for i, tables in page_tables.items():
                if str(i) in full_doc_map: full_doc_map[str(i)]["tables"] = tables
        except Exception as e: logger.warning(f" - Table fail: {e}")

        for page in full_doc_map.values():
            page["extraction_engine"] = page.get("extraction_engine") or engine_name
            page["document_audit"] = {"low_text_ratio": audit.get("low_text_ratio"), "image_heavy_ratio": audit.get("image_heavy_ratio"), "total_embedded_images": audit.get("total_embedded_images")}

        with open(cache_path, "w", encoding="utf-8") as f: json.dump(full_doc_map, f, ensure_ascii=False, indent=2)
        return full_doc_map
