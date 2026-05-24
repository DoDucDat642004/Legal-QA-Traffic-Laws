import hashlib
import os
import time
import json
import re
import logging
from pydantic import BaseModel
import PIL.Image
try:
    from google import genai
except ImportError:
    genai = None

from google.genai import types
from src.data_pipeline.schemas import optional_int_from_llm
from src.rag.model_policy import generate_content_with_fallback, model_candidates

logger = logging.getLogger("BaseParser")


INT_LIKE_FIELDS = {
    "page_start",
    "page_end",
    "min_amount_vnd",
    "max_amount_vnd",
    "point_deduction",
    "total_training_hours",
    "theory_hours",
    "practice_hours",
    "required_distance_km",
    "processing_time_days",
}
LIST_FIELDS = {
    "remedial_measures",
    "additional_penalties",
    "vehicle_type",
    "keyword_tags",
    "subject_target",
    "rules",
    "qa_pairs",
    "asset_ids",
    "other_metrics",
    "target_audience",
    "traffic_participant",
}


def sanitize_llm_json(obj):
    """Repair common LLM JSON shape issues before Pydantic validation."""
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key in LIST_FIELDS and isinstance(value, str):
                obj[key] = [] if not value.strip() else [value.strip()]
                continue
            if key in INT_LIKE_FIELDS:
                obj[key] = optional_int_from_llm(value)
                continue
            sanitize_llm_json(value)
    elif isinstance(obj, list):
        for item in obj:
            sanitize_llm_json(item)
    return obj

class BaseParser:
    """Abstract base for legal document parsing with support for Hybrid Vision-Text Flow."""
    
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
        if genai is None or not api_key:
            self.client = None
        else:
            self.client = genai.Client(api_key=api_key)
            
    def extract_with_llm(self, chunk: dict, schema: type[BaseModel], system_prompt: str) -> BaseModel:
        """Hybrid Model Flow: Vision by Gemini, Extraction by Gemma."""
        if not self.client:
            raise RuntimeError("google-genai is not installed or configured.")
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        
        vision_repaired_context = ""
        has_visuals = bool(chunk.get("tables") or chunk.get("figures") or chunk.get("is_sign_page"))
        
        models_to_try = model_candidates("EXTRACTION_PRIMARY_MODEL", "EXTRACTION_MODEL")
        
        if has_visuals:
            logger.info(f" - [VISION REPAIR] Processing visuals for {chunk.get('source_chunk_id')}...")
            
            vision_contents = [
                "Bạn là chuyên gia về báo hiệu đường bộ. Dưới đây là hình ảnh từ Quy chuẩn QCVN 41 hoặc văn bản luật. "
                "Hãy liệt kê chính xác các biển báo/vạch kẻ đường xuất hiện trong ảnh (Mã biển, Tên biển, Ý nghĩa). "
                "Nếu là bảng biểu, hãy trích xuất lại nội dung bảng dạng Markdown. Trả về văn bản rõ ràng, trung thực."
            ]
            
            if chunk.get("is_sign_page") and chunk.get("image_path"):
                full_img_path = os.path.join(project_root, chunk["image_path"])
                if os.path.exists(full_img_path):
                    vision_contents.append("\n[ẢNH TRANG PHỤ LỤC]:")
                    vision_contents.append(PIL.Image.open(full_img_path))

            for tbl in chunk.get("tables", []):
                if tbl.get("image_path"):
                    tbl_path = os.path.join(project_root, tbl["image_path"])
                    if os.path.exists(tbl_path):
                        vision_contents.append(f"\n[ẢNH CROP BẢNG {tbl['id']}]:")
                        vision_contents.append(PIL.Image.open(tbl_path))
            
            for fig in chunk.get("figures", []):
                if fig.get("image_path"):
                    fig_path = os.path.join(project_root, fig["image_path"])
                    if os.path.exists(fig_path):
                        vision_contents.append(f"\n[ẢNH CROP BIỂN {fig.get('code')}]:")
                        vision_contents.append(PIL.Image.open(fig_path))

            try:
                vision_res, _model = generate_content_with_fallback(
                    self.client,
                    contents=vision_contents,
                    config=types.GenerateContentConfig(temperature=0.0),
                    env_names=("EXTRACTION_VISION_MODEL",),
                    vision=True,
                    logger=logger,
                    label="Vision repair",
                )
                if vision_res.text:
                    vision_repaired_context = f"\n[NỘI DUNG ĐÃ ĐƯỢC AI PHỤC HỒI TỪ ẢNH]:\n{vision_res.text}\n"
            except Exception as e:
                logger.warning("Vision repair failed across allowed Gemini VLM models: %s", e)

        extraction_prompt = f"""
                {system_prompt}

                # INPUT DATA
                [ORIGINAL TEXT]:
                {chunk.get('text', '')}

                {vision_repaired_context}

                # TASK
                Perform structured legal extraction into JSON format using the above data.
                Return ONLY the JSON block.
            """
        for attempt in range(10): 
            model_name = models_to_try[attempt % len(models_to_try)]
            try:
                # 13s wait for free tier RPM limits
                time.sleep(13)
                
                response, used_model = generate_content_with_fallback(
                    self.client,
                    contents=[extraction_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=8192,
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    ),
                    env_names=("EXTRACTION_PRIMARY_MODEL", "EXTRACTION_MODEL"),
                    logger=logger,
                    label="Structured extraction",
                )
                model_name = used_model
                
                if not response.text:
                    raise ValueError("API returned empty text")
                
                text = response.text.strip()
                json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                json_str = json_match.group(1) if json_match else text
                
                # Cleanup potential start/end markers
                start = json_str.find('{')
                end = json_str.rfind('}')
                if start != -1 and end != -1:
                    json_str = json_str[start:end+1]
                
                # Robust LLM JSON repair before schema validation.
                try:
                    data = json.loads(json_str)
                    sanitize_llm_json(data)
                    json_str = json.dumps(data, ensure_ascii=False)
                except Exception:
                    pass

                return schema.model_validate_json(json_str)
            except Exception as e:
                logger.error(f"Retry {attempt+1}/10 extraction due to error: {e}")
                time.sleep(min(60, 5 + 2 ** attempt))
        return None

    def clean_text(self, text_lines: list[str]) -> str:
        return "\n".join(text_lines).strip()

    def source_text(self, line_refs: list[tuple[int, str]]) -> str:
        return "\n".join(line for _, line in line_refs).strip()

    def merge_line_refs(self, *groups: list[tuple[int, str]] | None) -> list[tuple[int, str]]:
        merged = []
        for g in groups:
            if g: merged.extend(g)
        return sorted(list(set(merged)), key=lambda x: x[0])

    def make_chunk(self, doc_name: str, source_file: str, chunk_kind: str, text: str, 
                   source_groups: list = None, body_source_lines: list = None, 
                   chapter_num: str = "", article_num: str = "", 
                   article_title: str = "", clause_num: str = "", 
                   point_key: str = "", tables: list = None) -> dict:
        return {
            "doc_name": doc_name,
            "source_file": source_file,
            "kind": chunk_kind,
            "text": text,
            "chapter_num": chapter_num,
            "article_num": article_num,
            "article_title": article_title,
            "clause_num": clause_num,
            "point_key": point_key,
            "tables": tables or []
        }
