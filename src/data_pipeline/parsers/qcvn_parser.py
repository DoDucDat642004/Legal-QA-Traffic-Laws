import re
import json
import os
import logging
from typing import Optional
from src.data_pipeline.text_normalizer import TextNormalizer
from .decree_parser import DecreeParser
from ..schemas import QCVNRuleList

logger = logging.getLogger("QcvnParser")

class QcvnParser(DecreeParser):
    """
    Hybrid parser for Technical Regulations (QCVN).
    Handles administrative clauses and technical multilevel indices (e.g., 61.3.1).
    """
    
    def extract_structured_data(self, chunk: dict, doc_name: str) -> Optional[QCVNRuleList]:
        """Sử dụng LLM để trích xuất dữ liệu có cấu trúc từ text chunk của QCVN."""
        schema = QCVNRuleList
        
        system_prompt = f"""
# ROLE
You are a High-Level Technical Auditor for Vietnamese Road Marking and Traffic Sign Regulations (QCVN).

# GOAL
Extract detailed technical rules and sign specifications from the provided multimodal content (text + table images + figure images) with absolute accuracy.

# CHAIN-OF-THOUGHT (CoT)
1. ANALYZE: Identify if this text describes a sign, a road marking, or a technical standard. Cross-reference with the [ẢNH BẢNG] or [HÌNH BIỂN] if provided.
2. VALIDATE: Ensure the dimensions, colors, and codes match the visuals.
3. REASON: Interpret colors, shapes, placement rules, and meanings. Explain this in 'thought_process' in Vietnamese.

# OUTPUT FORMAT (MANDATORY JSON)
Return ONLY a JSON block. Use VIETNAMESE for all text values.
{{
  "thought_process": "Phân tích kỹ thuật về quy chuẩn biển báo/vạch kẻ đường này...",
  "rules": [
    {{
      "id": "Unique_ID_based_on_SignCode",
      "legal_reference": {{ 
        "document": "{doc_name}", "article": "...", "section": "...",
        "appendix": "...", "cross_references": ["QCVN 41:2019", "Điều..."]
      }},
      "sign_info": {{ 
        "sign_code": "P.101", "sign_name": "Tên biển báo", "sign_type": "Biển báo cấm/Hiệu lệnh/..." 
      }},
      "meaning_and_usage": "Ý nghĩa và cách sử dụng chi tiết bằng tiếng Việt",
      "technical_specs": {{ 
        "shape": "Hình tròn/Hình chữ nhật...", 
        "colors": "Nền trắng, viền đỏ...", 
        "dimensions": "Đường kính 70cm...",
        "placement_rules": "Cắm ở vị trí..." 
      }},
      "qa_context": "Câu trả lời ngắn gọn phục vụ QA"
    }}
  ]
}}

# CRITICAL RULES
- **Legal Reference**: Do NOT use "Không xác định", "N/A", or "None". If a value is unknown, return an empty string "". Use the coordinates provided in the Location or document fields.
- **Sign Identity**: The 'sign_code' (e.g., P.101, W.201, Vạch 1.1) is the MOST important field. 
- **Multimodal**: If the text describes a shape/color, cross-verify with any images provided in the context.
- **Precision**: Dimensions must be exact. Do not generalize.
- **Cross-References**: Identify if this sign replaces or relates to signs in previous regulations.
"""
        try:
            return self.extract_with_llm(chunk, schema, system_prompt)
        except Exception as e:
            logger.error(f"Error in QCVN optimized extraction: {e}")
            return None

    def parse(self, md_text: str, doc_name: str, doc_map: dict = None) -> list[dict]:
        """
        Ghi đè hoàn toàn logic parse cho QCVN.
        Sử dụng máy trạng thái (State Machine) để cắt chunk dựa trên các mục kỹ thuật (1.1, 1.2.1, Phụ lục).
        """
        lines = md_text.split('\n')
        chunks = []
        
        # Regex cho các cấp độ mục của QCVN
        tech_re = re.compile(r'^(\d+(?:\.\d+){0,3})\.?\s+(.*)$')
        section_re = re.compile(r'^(PHẦN|CHƯƠNG)\s+\d+', re.IGNORECASE)
        appendix_re = re.compile(r'^Phụ lục\s+([A-Z])', re.IGNORECASE)
        page_marker_re = re.compile(r'^\[INTERNAL_PAGE_MARKER_(\d+)\]$')

        state = {
            "level1_num": "", "level1_title": "",
            "level2_num": "", "level2_title": "",
            "level3_num": "", "level3_title": "",
            "appendix_id": "", "appendix_title": "",
            "content_lines": [],
            "page_start": None, "page_end": None,
            "parent_hierarchy": []
        }

        def flush():
            if not state["content_lines"]: return
            
            content_raw = "\n".join(state["content_lines"]).strip()
            if not content_raw: return
            
            content = TextNormalizer.normalize_vietnamese(content_raw)
            if len(content.split()) < 3: return # Bỏ qua các dòng rác

            page_idx = str(state["page_start"]) if state["page_start"] is not None else "0"
            tables = []
            if doc_map and page_idx in doc_map:
                # Tìm bảng phù hợp cho chunk kỹ thuật này
                p_data = doc_map[page_idx]
                tables = p_data.get("tables", [])
                # Lọc bảng dựa trên text overlap đơn giản
                tables = [t for t in tables if (t.get("text") or "").lower().split("|")[0].strip()[:20] in content.lower()]

            import hashlib
            # Tạo ID định danh dựa trên mục kỹ thuật hoặc phụ lục
            ref_id = state["appendix_id"] or state["level3_num"] or state["level2_num"] or state["level1_num"] or "intro"
            short_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:6]
            source_chunk_id = f"TECH_{doc_name}_{ref_id}_{short_hash}".lower().replace(" ", "_").replace(".", "_")

            chunk = self.make_chunk(
                doc_name=doc_name, source_file="", chunk_kind="technical",
                text=content, source_groups=[], body_source_lines=[],
                chapter_num=state["level1_num"],
                article_num=ref_id,
                article_title=state["level3_title"] or state["level2_title"] or state["level1_title"] or state["appendix_title"],
                clause_num="", point_key="", tables=tables
            )
            
            page_data = doc_map.get(str(state["page_start"]), {}) if doc_map else {}
            
            chunk.update({
                "page_start": state["page_start"],
                "page_end": state["page_end"],
                "source_chunk_id": source_chunk_id,
                "source_body_exact": content,
                "semantic_context": "\n".join(
                    f"{item.get('kind', '')} {item.get('num', '')}: {item.get('title', '')}".strip()
                    for item in state["parent_hierarchy"]
                    if item.get("num") or item.get("title")
                ),
                "parent_hierarchy": list(state["parent_hierarchy"]),
                "image_path": page_data.get("img", "")
            })
            chunks.append(chunk)
            state["content_lines"].clear()

        for line in lines:
            line_s = line.strip()
            if not line_s: continue
            
            # Theo dõi trang
            pm = page_marker_re.match(line_s)
            if pm:
                p_num = int(pm.group(1))
                if state["page_start"] is None: state["page_start"] = p_num
                state["page_end"] = p_num
                continue

            # Nhận diện Phụ lục
            ap = appendix_re.match(line_s)
            if ap:
                flush()
                state.update({
                    "appendix_id": ap.group(0), "appendix_title": line_s,
                    "level1_num": "", "level2_num": "", "level3_num": "",
                    "parent_hierarchy": [{"kind": "appendix", "num": ap.group(1), "title": line_s}]
                })
                state["page_start"] = state["page_end"]
                continue

            # Nhận diện Phần/Chương
            se = section_re.match(line_s)
            if se:
                flush()
                state.update({
                    "level1_num": line_s, "level1_title": line_s,
                    "level2_num": "", "level3_num": "", "appendix_id": "",
                    "parent_hierarchy": [{"kind": "section", "num": line_s, "title": ""}]
                })
                state["page_start"] = state["page_end"]
                continue

            # Nhận diện Mục kỹ thuật (1.1, 1.2.1...)
            m = tech_re.match(line_s)
            if m:
                num_parts = m.group(1).strip('.').split('.')
                title = m.group(2).strip()
                
                # Chỉ flush nếu là đầu mục mới (tránh flush ở mỗi dòng bắt đầu bằng số)
                if len(num_parts) <= 3:
                    flush()
                    if len(num_parts) == 1: 
                        state.update({"level1_num": m.group(1), "level1_title": title, "level2_num": "", "level3_num": ""})
                        state["parent_hierarchy"] = [{"kind": "mục", "num": m.group(1), "title": title}]
                    elif len(num_parts) == 2: 
                        state.update({"level2_num": m.group(1), "level2_title": title, "level3_num": ""})
                        # Giữ lại cấp 1 trong hierarchy nếu có
                        p_h = [item for item in state["parent_hierarchy"] if item["kind"] in ["section", "appendix"]]
                        if state["level1_num"]: p_h.append({"kind": "mục", "num": state["level1_num"], "title": state["level1_title"]})
                        state["parent_hierarchy"] = p_h
                    elif len(num_parts) == 3: 
                        state.update({"level3_num": m.group(1), "level3_title": title})
                    
                    state["page_start"] = state["page_end"]
                    continue
            
            # Safeguard: Nếu một mục quá dài (> 80 dòng) mà không có ngắt nghỉ, tự động ngắt
            if len(state["content_lines"]) > 80:
                flush()
                state["page_start"] = state["page_end"]

            state["content_lines"].append(line)

        flush()
        return chunks
