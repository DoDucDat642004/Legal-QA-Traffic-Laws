import re
import json
import os
import logging
from typing import List, Union
from pydantic import BaseModel
from src.data_pipeline.text_normalizer import TextNormalizer
from .base_parser import BaseParser
from ..schemas import Decree168List, Decree336List

logger = logging.getLogger("DecreeParser")

class DecreeParser(BaseParser):
    """
    State-machine parser with spatial table linking.
    Handles multi-page tracking, anchored legal regex, and deterministic ID generation.
    """
    # Conservative limit to avoid Gemini context saturation
    MAX_WORDS = 800 

    def extract_structured_data(self, chunk: dict, doc_name: str) -> Union[Decree168List, Decree336List, None]:
        """Sử dụng LLM để trích xuất dữ liệu có cấu trúc từ text chunk của Nghị định."""
        # Determine schema class for validation
        schema_class = Decree168List if "168" in doc_name else Decree336List
        
        system_prompt = f"""
# ROLE
You are a Legal Data Engineer specializing in Vietnamese administrative decrees.

# GOAL
Extract every single legal rule from the provided multimodal content (text + table images + figure images) with 100% precision. 
Do not miss any points (a, b, c...), fine amounts, or remedial measures.

# FEW-SHOT EXAMPLES (Logic only)
- SOURCE: "Phạt tiền từ 100k đến 200k đối với cá nhân, từ 200k đến 400k đối với tổ chức..."
- OUTPUT: {{ "individual_min_vnd": 100000, "individual_max_vnd": 200000, "organization_min_vnd": 200000, "organization_max_vnd": 400000 }}
- SOURCE: "Phạt tiền từ 2tr đến 3tr đối với người điều khiển..." (assume individual if not specified)
- OUTPUT: {{ "individual_min_vnd": 2000000, "individual_max_vnd": 3000000 }}

# CHAIN-OF-THOUGHT (CoT) PROCESS
1. ANALYZE: Identify the violation behavior, the target subject (Individual vs Organization), and all penalty components. Cross-check with table images if provided.
2. THINK: Reason about the hierarchy and how it maps to the structured output. Write this in 'thought_process' in Vietnamese.
3. EXTRACT: Transform the legal text into a strict JSON list of rules.

# OUTPUT FORMAT (MANDATORY)
Return ONLY a JSON block. All text values MUST be in Vietnamese.
JSON Structure:
{{
  "thought_process": "Phân tích logic về các điều khoản này, bao gồm cả việc xác định mức phạt cho Cá nhân vs Tổ chức...",
  "rules": [
    {{
      "id": "Unique_ID_based_on_Article_Clause_Point",
      "legal_reference": {{ 
        "document": "{doc_name}", 
        "article": "...", "clause": "...", "point": "...",
        "cross_references": ["Điều 5", "Khoản 2 Điều 10"] 
      }},
      "violation_content": "Mô tả đầy đủ hành vi vi phạm",
      "penalties": {{
        "main_penalty": {{
            "individual_min_vnd": 0, "individual_max_vnd": 0,
            "organization_min_vnd": 0, "organization_max_vnd": 0,
            "description": "Mô tả bằng chữ về mức phạt tiền"
        }},
        "additional_penalties": ["Tước GPLX từ 01-03 tháng", "..."],
        "remedial_measures": ["Buộc khôi phục tình trạng ban đầu", "..."]
      }},
      "qa_context": "Tóm tắt ngắn gọn để trả lời câu hỏi QA"
    }}
  ]
}}

# CRITICAL CONSTRAINTS
- **Legal Reference**: Do NOT use "Không xác định", "N/A", or "None". If a value is unknown, return an empty string "". Use the values provided in the Location field as your primary source.
- **Penalty Logic**: In Vietnam Decrees, if a penalty is stated without specifying "Individual" or "Organization", it is usually for the Individual. The Organization fine is typically DOUBLE. Extract all 4 VND fields if possible.
- **Cross-References**: Carefully extract any references to other Articles or Clauses mentioned in the text.
- **Verbatim**: Fine amounts must be 100% exact. Use numbers only for VND fields.
"""
        try:
            return self.extract_with_llm(chunk, schema_class, system_prompt)
        except Exception as e:
            logger.error(f"Error in LLM extraction: {e}")
            return None

    def parse(self, md_text: str, doc_name: str, doc_map: dict = None) -> list[dict]:
        # Nếu có doc_map với layout info, sử dụng layout-aware parsing
        if doc_map and any(p.get("layout") or p.get("layout_path") for p in doc_map.values()):
            return self._parse_with_layout(doc_map, doc_name)
        
        # Fallback to regex-based line parsing (existing logic)
        return self._parse_with_regex(md_text, doc_name, doc_map)

    def _parse_with_layout(self, doc_map: dict, doc_name: str) -> list[dict]:
        """Uses bbox and span information for layout-aware legal parsing and spatial table linking."""
        chunks = []
        state = {
            "chapter_num": "", "chapter_title": "", "article_num": "", "article_title": "",
            "clause_num": "", "point_key": "", "article_preamble": "", "clause_preamble": "",
            "buffer": [], "buffer_bboxes": [], "page_start": None, "page_end": None,
            "pending_header": "" # For header stitching
        }

        # Heading regexes must be anchored. Legal bodies often contain references such as
        # "điểm a khoản 2 Điều 11"; allowing arbitrary prefixes makes those references look
        # like real article headers and corrupts the article/clause/point state.
        chapter_re = re.compile(r'^\s*(?:#\s*)?(?:Chương|CHUONG)\s+([IVXLCDM\d]+)\b', re.IGNORECASE)
        article_re = re.compile(r'^\s*(?:#\s*)?(?:Điều|DIEU)\s+(\d+[a-z]*)\b', re.IGNORECASE)
        clause_re = re.compile(r'^\s*(?:Khoản\s+)?(\d+)[\.\)\-]\s+')
        # Keep this case-sensitive so roman headings like "# I. QUY ĐỊNH CHUNG" are not
        # misread as point "i".
        point_re = re.compile(r'^\s*([a-zđ])[\)\.]\s+')

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

        def flush():
            # Join text and normalize
            text_raw = "\n".join(state["buffer"]).strip()
            if not text_raw: return
            
            text = TextNormalizer.normalize_vietnamese(text_raw)
            kind = "point" if state["point_key"] else "clause" if state["clause_num"] else "article"
            
            # Context management
            if kind == "article": state["article_preamble"] = text
            elif kind == "clause": state["clause_preamble"] = text

            parent_hierarchy = []
            if state["chapter_num"]:
                parent_hierarchy.append({"kind": "chapter", "num": state["chapter_num"], "title": state.get("chapter_title")})
            if state["article_num"] and kind != "article":
                parent_hierarchy.append({"kind": "article", "num": state["article_num"], "title": state.get("article_title")})
            if state["clause_num"] and kind == "point":
                parent_hierarchy.append({"kind": "clause", "num": state["clause_num"]})

            import hashlib
            article_id = state['article_num'] or "0"
            clause_id = state['clause_num'] or "0"
            point_id = state['point_key'] or "0"
            short_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
            content_hash = f"{doc_name}_{article_id}_{clause_id}_{point_id}_{short_hash}".lower().replace(" ", "_").replace(".", "")
            
            tables = []
            if state["page_start"] is not None:
                chunk_meta = {
                    "text": text,
                    "page_start": state["page_start"],
                    "page_end": state["page_end"] or state["page_start"],
                    "buffer_bboxes": state["buffer_bboxes"]
                }
                tables = self._get_tables_for_chunk(chunk_meta, doc_map)

            chunk = self.make_chunk(
                doc_name=doc_name, source_file="", chunk_kind=kind, text=text,
                source_groups=[], body_source_lines=[],
                chapter_num=state["chapter_num"], article_num=state["article_num"],
                article_title=state["article_title"], clause_num=state["clause_num"], 
                point_key=state["point_key"], tables=tables
            )

            context_parts = []
            if state.get("article_title"):
                context_parts.append(f"Điều {state['article_num']}. {state['article_title']}")
            elif state.get("article_num"):
                context_parts.append(f"Điều {state['article_num']}")
            if kind == "point" and state.get("clause_preamble"):
                context_parts.append(state["clause_preamble"])
            elif kind in {"clause", "point"} and state.get("article_preamble"):
                context_parts.append(state["article_preamble"])
            
            page_data = doc_map.get(str(state["page_start"]), {})
            chunk.update({
                "source_chunk_id": content_hash, 
                "page_start": state["page_start"], 
                "page_end": state["page_end"],
                "source_body_exact": text,
                "semantic_context": "\n".join(p for p in context_parts if p and p != text),
                "parent_hierarchy": parent_hierarchy,
                "image_path": page_data.get("img", ""),
                "is_sign_page": page_data.get("is_sign_page", False)
            })
            chunks.append(chunk)
            state["buffer"].clear()
            state["buffer_bboxes"].clear()

        # Iterate through pages and blocks in natural order
        for p_idx_str, page_data in sorted(doc_map.items(), key=lambda x: int(x[0])):
            p_idx = int(p_idx_str)
            layout = page_data.get("layout")
            
            if not layout and page_data.get("layout_path"):
                l_path = os.path.join(project_root, page_data["layout_path"])
                if os.path.exists(l_path):
                    try:
                        with open(l_path, "r", encoding="utf-8") as f:
                            layout = json.load(f)
                    except Exception as exc:
                        logger.debug("Could not load layout file %s: %s", l_path, exc)

            if not layout: 
                lines = page_data.get("corrected", "").split("\n")
                for line in lines:
                    clean_line = line.strip()
                    # NOISE FILTER: Ignore single-number lines (likely page numbers)
                    if re.match(r'^\d{1,3}$', clean_line): continue
                    self._process_line(clean_line, None, p_idx, state, flush, chapter_re, article_re, clause_re, point_re)
                continue

            for block in layout.get("blocks", []):
                if block.get("type") != 0: continue
                for line in block.get("lines", []):
                    line_text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if not line_text: continue
                    # NOISE FILTER
                    if re.match(r'^\d{1,3}$', line_text): continue
                    bbox = line.get("bbox")
                    self._process_line(line_text, bbox, p_idx, state, flush, chapter_re, article_re, clause_re, point_re)
        
        flush()
        return chunks

    def _get_tables_for_chunk(self, chunk: dict, doc_map: dict) -> list:
        """Lấy bảng thực sự thuộc chunk dựa trên page và text overlap (ít nhất 1 cell khớp).
        Bổ sung kiểm tra tỉ lệ để tránh bảng 'nuốt' sạch văn bản chính."""
        tables = []
        page_start = chunk.get("page_start", 0)
        page_end = chunk.get("page_end") or page_start
        
        chunk_text_raw = chunk.get("text", "")
        chunk_text_lower = chunk_text_raw.lower()
        chunk_len = len(chunk_text_raw)
        
        for p_idx in range(page_start, page_end + 1):
            page_data = doc_map.get(str(p_idx), {})
            page_tables = page_data.get("tables", [])
            
            for tbl in page_tables:
                tbl_text = tbl.get("text", "")
                tbl_text_lower = tbl_text.lower()
                if not tbl_text: continue
                
                # Tránh trường hợp bảng chứa quá nhiều văn bản so với chính chunk đó
                if chunk_len > 0 and len(tbl_text) > chunk_len * 0.7:
                    continue

                first_cell = tbl_text_lower.split("|")[0].strip()[:30]
                if first_cell and first_cell in chunk_text_lower:
                    tables.append(tbl)
                    continue
                    
                if chunk.get("buffer_bboxes"):
                    y0_min = min(b[1] for b in chunk["buffer_bboxes"])
                    y1_max = max(b[3] for b in chunk["buffer_bboxes"])
                    t_bbox = tbl.get("bbox")
                    if t_bbox:
                        if t_bbox[1] >= y0_min - 10 and t_bbox[1] <= y1_max + 150:
                            tables.append(tbl)
        return tables

    def _is_legal_clause(self, line: str) -> bool:
        """Xác nhận dòng là khoản luật."""
        clean = line.strip()
        if re.match(r'^\d+[\.\)]\s+', clean):
            return True
        return False

    def _process_line(self, line_text, bbox, p_idx, state, flush_func, chapter_re, article_re, clause_re, point_re):
        """Line-by-line state machine processing with anchored legal checks."""
        match_text = line_text.strip()
        
        # HEADER STITCHING: If line is just "Điều" or "Khoản" without a number, wait for next line
        if match_text.lower() in ["điều", "khoản", "chương"] and not state["pending_header"]:
            state["pending_header"] = match_text
            return
        
        if state["pending_header"]:
            match_text = f"{state['pending_header']} {match_text}"
            state["pending_header"] = ""

        def article_sequence_is_plausible(article_num: str) -> bool:
            if not article_num.isdigit():
                return True
            current = state.get("article_num") or ""
            if not str(current).isdigit():
                return True
            new_num = int(article_num)
            current_num = int(current)
            # OCR/table rows in amendment articles often contain references such as
            # "Điều 82", "Điều 95" etc. They are legal content, not top-level headings.
            if new_num > current_num + 5:
                return False
            if new_num + 2 < current_num:
                return False
            return True

        if chapter_re.match(match_text):
            flush_func()
            m = chapter_re.match(match_text)
            state.update({"chapter_num": m.group(1), "article_num": "", "clause_num": "", "point_key": "", "page_start": p_idx})
        elif article_re.match(match_text) and article_sequence_is_plausible(article_re.match(match_text).group(1)):
            flush_func()
            m = article_re.match(match_text)
            state.update({"article_num": m.group(1), "clause_num": "", "point_key": "", "page_start": p_idx})
        elif clause_re.match(match_text) and self._is_legal_clause(match_text):
            flush_func()
            m = clause_re.match(match_text)
            state.update({"clause_num": m.group(1), "point_key": "", "page_start": p_idx})
        elif point_re.match(match_text):
            flush_func()
            m = point_re.match(match_text)
            state.update({"point_key": m.group(1), "page_start": p_idx})
        
        # SAFEGUARD: Buffer too long
        if len(state["buffer"]) > 150: # Increased slightly for stitched blocks
            flush_func()
            state["page_start"] = p_idx

        if state["page_start"] is None: state["page_start"] = p_idx
        state["page_end"] = p_idx
        state["buffer"].append(line_text)
        if bbox: state["buffer_bboxes"].append(bbox)

    def _parse_with_regex(self, md_text: str, doc_name: str, doc_map: dict = None) -> list[dict]:
        logger.warning(f" - [FALLBACK] Missing layout info for {doc_name}. Using regex-based line parsing.")
        return self._parse_with_layout(doc_map or {"0": {"raw": md_text, "corrected": md_text, "layout": None}}, doc_name)
