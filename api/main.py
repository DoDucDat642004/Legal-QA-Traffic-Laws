import os
import io
import re
import json
import logging
from functools import lru_cache
from google import genai
from google.genai import types
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.rag.legal_graph_rag import LegalGraphRAG
import uvicorn
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("traffic_law_api")
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


app = FastAPI(title="Luật Giao Thông AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
processed_dir = os.path.join(project_root, "data", "processed")
if os.path.exists(processed_dir):
    app.mount("/processed", StaticFiles(directory=processed_dir), name="processed")


@lru_cache(maxsize=1)
def get_rag() -> LegalGraphRAG:
    enable_reranker = os.getenv("RAG_ENABLE_RERANKER", "").lower() in {"1", "true", "yes", "on"}
    return LegalGraphRAG(
        "data/processed",
        graph_path="data/graph/legal_graph.json",
        use_reranker=enable_reranker,
    )


def _public_asset_url(path: str) -> str:
    if not path:
        return ""
    normalized = path.replace("\\", "/")
    if normalized.startswith("data/processed/"):
        return "/processed/" + normalized[len("data/processed/"):]
    return "/processed/" + normalized.split("data/processed/")[-1]


def _traffic_sign_query_hints(description: str, user_query: str = "") -> str:
    text = f"{description} {user_query}".lower()
    hints = ["QCVN 41:2024", "Thông tư 51/2024", "biển báo cấm", "Phụ lục B"]

    sign_hints = []
    if re.search(r"\bP\s*\.?\s*\d{2,3}[a-zđ]?\b", text, re.IGNORECASE):
        sign_hints.extend(re.findall(r"\bP\s*\.?\s*\d{2,3}[a-zđ]?\b", f"{description} {user_query}", re.IGNORECASE))
    if any(k in text for k in ["ngược chiều", "no entry", "thanh ngang", "gạch ngang", "vạch ngang", "dấu trừ", "một chiều cấm vào"]):
        sign_hints.extend(["P.102", "Cấm đi ngược chiều"])
    if ("đường cấm" in text) or ("hình tròn" in text and "viền đỏ" in text and "nền trắng" in text and not any(k in text for k in ["thanh ngang", "gạch ngang", "ô tô", "xe máy", "người đi bộ", "rẽ", "quay đầu"])):
        sign_hints.extend(["P.101", "Đường cấm"])
    if "ô tô" in text or "xe hơi" in text or "car" in text:
        sign_hints.extend(["P.103a", "Cấm xe ô tô"])
    if "xe máy" in text or "mô tô" in text or "motorcycle" in text:
        sign_hints.extend(["P.104", "Cấm xe máy"])
    if "xe tải" in text or "truck" in text:
        sign_hints.extend(["P.106a", "Cấm xe ô tô tải"])
    if "người đi bộ" in text or "pedestrian" in text:
        sign_hints.extend(["P.112", "Cấm người đi bộ"])
    if "rẽ trái" in text:
        sign_hints.extend(["P.123a", "Cấm rẽ trái"])
    if "rẽ phải" in text:
        sign_hints.extend(["P.123b", "Cấm rẽ phải"])
    if "quay đầu" in text or "u-turn" in text:
        sign_hints.extend(["P.124a", "Cấm quay đầu xe"])
    if "vượt" in text:
        sign_hints.extend(["P.125", "Cấm vượt"])
    if "tốc độ" in text or re.search(r"\b\d{2,3}\s*(km/h|kmh)\b", text):
        sign_hints.extend(["P.127", "Tốc độ tối đa cho phép"])
    if "dừng" in text or "đỗ" in text or "parking" in text:
        sign_hints.extend(["P.130", "Cấm dừng xe và đỗ xe", "P.131", "Cấm đỗ xe"])

    if not sign_hints and ("biển cấm" in text or "viền đỏ" in text or "màu đỏ" in text):
        sign_hints.extend(["nhóm biển báo cấm", "P.101 đến P.140"])

    deduped = list(dict.fromkeys(x.strip() for x in [*hints, *sign_hints] if x and x.strip()))
    return ". ".join(deduped)


def _looks_like_table_query(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in ["bảng", "dòng", "cột", "ô bảng", "tra bảng", "table"])


def _looks_like_sign_query(query: str) -> bool:
    q = (query or "").lower()
    return bool(
        re.search(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", query or "", re.IGNORECASE)
        or "biển báo" in q
        or "biển cấm" in q
        or ("biển" in q and any(k in q for k in ["viền đỏ", "nền đỏ", "nền trắng", "cấm"]))
    )


def _parse_vision_json(text: str) -> dict:
    if not text:
        return {}
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        codes = re.findall(r"\b(?:DP|IE|P|W|R|I|S|E)\s*\.?\s*\d{2,3}[a-zđ]?\b", text, re.IGNORECASE)
        return {
            "raw_description": text,
            "candidate_codes": list(dict.fromkeys(codes)),
            "confidence": 0.45 if codes else 0.0,
            "is_traffic_sign": bool(codes or "biển" in text.lower()),
        }


def _references(docs: list[dict]) -> list[dict]:
    return [
        {
            "source_chunk_id": d.get("source_chunk_id"),
            "modality": d.get("rag_modality"),
            "legal_reference": d.get("legal_reference"),
            "image": _public_asset_url(d.get("image_path")),
            "retrieval_reasons": d.get("retrieval_reasons", []),
            "retrieval_score": d.get("retrieval_score"),
            "matched_table_rows": d.get("matched_table_rows", []),
        }
        for d in docs
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "rag_loaded": get_rag.cache_info().currsize > 0}


@app.post('/chat/text')
async def chat_text(query: str = Form(...)):
    try:
        rag = get_rag()
        if _looks_like_table_query(query):
            docs = rag.retrieve_table(query)
        elif _looks_like_sign_query(query):
            docs = rag.retrieve_sign(query)
        else:
            docs = rag.retrieve(query)
        ans = rag.generate_answer(query, docs)

        images = [_public_asset_url(d.get("image_path")) for d in docs if d.get("image_path")]

        return {
            "answer": ans,
            "images": sorted(set(x for x in images if x)),
            "references": _references(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/chat/sign')
async def chat_sign(query: str = Form(...)):
    try:
        rag = get_rag()
        docs = rag.retrieve_sign(query, top_k=8)
        ans = rag.generate_answer(query, docs)
        images = [_public_asset_url(d.get("image_path")) for d in docs if d.get("image_path")]
        return {
            "answer": ans,
            "reference_images": sorted(set(x for x in images if x)),
            "references": _references(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/chat/table')
async def chat_table(query: str = Form(...)):
    try:
        rag = get_rag()
        docs = rag.retrieve_table(query, top_k=8)
        ans = rag.generate_answer(query, docs)
        images = [_public_asset_url(d.get("image_path")) for d in docs if d.get("image_path")]
        return {
            "answer": ans,
            "table_images": sorted(set(x for x in images if x)),
            "references": _references(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/image')
async def chat_image(image: UploadFile = File(...), query: str = Form("")):
    try:
        if client is None:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY/GENAI_API_KEY is required for image understanding.")
        image_bytes = await image.read()
        user_img = Image.open(io.BytesIO(image_bytes))
        
        vision_prompt = (
            "Bạn đang nhận diện biển báo giao thông Việt Nam theo QCVN 41:2024/BGTVT. "
            "Chỉ trả về JSON hợp lệ, không markdown, với schema: "
            "{\"is_traffic_sign\": boolean, \"candidate_codes\": [\"P.102\"], "
            "\"confidence\": number, \"shape\": string, \"colors\": [string], "
            "\"symbol\": string, \"text_on_sign\": string, \"likely_meaning\": string, "
            "\"raw_description\": string}. "
            "Nếu không chắc mã, candidate_codes chứa 2-3 mã ứng viên phù hợp nhất và confidence thấp hơn 0.6."
        )
        
        vision_models = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-flash-latest"]
        res = None
        for v_model in vision_models:
            try:
                res = client.models.generate_content(
                    model=v_model,
                    contents=[vision_prompt, user_img],
                    config=genai.types.GenerateContentConfig(
                        temperature=0.0,
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    )
                )
                if res.text: break
            except Exception as e:
                logger.warning("Vision model %s failed: %s", v_model, e)
                continue
        
        if not res: raise HTTPException(status_code=500, detail="All vision models failed")
        vision = _parse_vision_json(res.text or "")
        visual_description = " ".join(
            str(x)
            for x in [
                vision.get("raw_description") or "",
                vision.get("shape") or "",
                " ".join(vision.get("colors") or []) if isinstance(vision.get("colors"), list) else vision.get("colors") or "",
                vision.get("symbol") or "",
                vision.get("text_on_sign") or "",
                vision.get("likely_meaning") or "",
            ]
            if x
        ).strip() or (res.text or "")
        candidate_codes = [str(code) for code in vision.get("candidate_codes") or [] if str(code).strip()]
        confidence = float(vision.get("confidence") or 0.0)

        if not vision.get("is_traffic_sign", True) and confidence < 0.4 and not candidate_codes:
            return {
                "description": visual_description,
                "vision": vision,
                "answer": "Ảnh chưa đủ dấu hiệu để xác định đây là biển báo giao thông Việt Nam trong QCVN 41:2024.",
                "reference_images": [],
                "references": [],
            }

        sign_hints = _traffic_sign_query_hints(" ".join([visual_description, *candidate_codes]), query)
        final_query = f"{sign_hints}. {' '.join(candidate_codes)}. {visual_description}. {query if query else ''}"
        rag = get_rag()
        docs = rag.retrieve_sign(final_query, top_k=8)
        ans = rag.generate_answer(final_query, docs)

        images = [_public_asset_url(d.get("image_path")) for d in docs if d.get("image_path")]

        return {
            "description": visual_description,
            "vision": vision,
            "answer": ans,
            "reference_images": sorted(set(x for x in images if x)),
            "references": _references(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
