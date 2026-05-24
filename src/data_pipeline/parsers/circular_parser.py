import re
import json
import os
import logging
from typing import Optional
from .decree_parser import DecreeParser
from ..schemas import CircularRuleList

logger = logging.getLogger("CircularParser")

class CircularParser(DecreeParser):
    """Specific parser for Circulars (Thông tư) which frequently contain dense tables and forms."""

    def extract_structured_data(self, chunk: dict, doc_name: str) -> Optional[CircularRuleList]:
        """Sử dụng LLM để trích xuất dữ liệu có cấu trúc từ text chunk của Thông tư."""
        schema = CircularRuleList
        system_prompt = f"""
# ROLE
You are a High-Level Legal Auditor for Vietnamese Administrative Circulars (Thông tư).

# GOAL
Extract every regulatory provision, procedure, and quantitative requirement from the provided multimodal content.

# CHAIN-OF-THOUGHT (CoT)
1. ANALYZE: Identify procedures, dossier requirements, processing times, and quantitative specs (hours, km, etc.).
2. VALIDATE: Cross-check dense tables or forms in the images with the extracted text.
3. EXTRACT: Write the 'thought_process' and then transform into strict JSON.

# OUTPUT FORMAT (MANDATORY JSON)
Return ONLY a JSON block. Use VIETNAMESE for all values.
{{
  "thought_process": "Phân tích mục đích của quy định/thủ tục này...",
  "rules": [
    {{
      "id": "Unique_ID",
      "legal_reference": {{ "document": "{doc_name}", "article": "...", "clause": "..." }},
      "original_text": "Trích xuất nguyên văn hoặc tóm tắt đầy đủ SOURCE_BODY_EXACT",
      "quantitative_data": {{ "processing_time_days": 10, "theory_hours": 40, ... }},
      "qa_context": "Detailed explanation for QA in Vietnamese"
    }}
  ]
}}

# CRITICAL CONSTRAINTS
- **Legal Reference**: Do NOT use "Không xác định", "N/A", or "None". If a value is unknown, return an empty string "".
- **Quantitative integers**: For numeric fields such as theory_hours, practice_hours, processing_time_days, total_training_hours, required_distance_km, return an integer only when the source states one exact number. If missing or unclear, return null, never "".
- Accuracy: All numbers, time periods, and dossier items must be 100% accurate.
- Prioritize images if text is garbled.
"""
        try:
            return self.extract_with_llm(chunk, schema, system_prompt)
        except Exception as e:
            logger.error(f"Error in Circular optimized extraction: {e}")
            return None

    def parse(self, md_text: str, doc_name: str, doc_map: dict = None) -> list[dict]:
        return super().parse(md_text, doc_name, doc_map)
