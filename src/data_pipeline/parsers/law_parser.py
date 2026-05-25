import re
import json
import os
import logging
from typing import Optional
from .decree_parser import DecreeParser
from ..schemas import LawRuleList

logger = logging.getLogger("LawParser")

class LawParser(DecreeParser):
    """Specialized parser for Laws (Luật) supporting non-enumerated preamble paragraphs."""

    def extract_structured_data(self, chunk: dict, doc_name: str) -> Optional[LawRuleList]:
        """Sử dụng LLM để trích xuất dữ liệu có cấu trúc từ text chunk của Luật."""
        schema = LawRuleList
        article_num = chunk.get('article_num', 'N/A')
        clause_num = chunk.get('clause_num', 'N/A')
        
        system_prompt = f"""
# ROLE
You are a High-Level Legal Auditor for Vietnamese Traffic Laws.

# GOAL
Extract every single legal provision from the provided multimodal content (text + table images + figure images) with 100% precision.

# COORDINATES (MANDATORY)
- Current Article: {article_num}
- Current Clause: {clause_num}
- Document: {doc_name}
You MUST use these coordinates in your JSON output. Do NOT guess or say "Unknown".

# CHAIN-OF-THOUGHT (CoT)
1. ANALYZE: Identify the type of rule (Definition, Prohibition, Mandatory rule, etc.).
2. VALIDATE: Cross-reference text with images if provided.
3. EXTRACT: Write the 'thought_process' and then transform into strict JSON.

# OUTPUT FORMAT (MANDATORY JSON)
Return ONLY a JSON block. Use VIETNAMESE for all values.
{{
  "thought_process": "Phân tích mục đích của quy định pháp luật này...",
  "rules": [
    {{
      "id": "Unique_ID",
      "legal_reference": {{ 
          "document": "{doc_name}", 
          "article": "{article_num}", 
          "clause": "{clause_num}",
          "point": "..." 
      }},
      "original_text": "Trích xuất nguyên văn, chính xác từng dấu câu",
      "qa_context": "Detailed explanation for QA in Vietnamese"
    }}
  ]
}}

# CRITICAL CONSTRAINTS
- **Legal Reference**: Do NOT use "Không xác định", "N/A", or "None". If a value is unknown, return an empty string "". Use the coordinates provided in the system prompt.
- NO hallucination. If text is missing, check images.
- Language: Instructions in English, DATA in VIETNAMESE.
"""
        try:
            return self.extract_with_llm(chunk, schema, system_prompt)
        except Exception as e:
            logger.error(f"Error in Law LLM extraction: {e}")
            return None

    def parse(self, md_text: str, doc_name: str, doc_map: dict = None) -> list[dict]:
        return super().parse(md_text, doc_name, doc_map)
