"""
Traffic Law AI API Server.

This module provides a FastAPI-based web server for legal traffic law Q&A.
It supports text-based queries, traffic sign identification, and image-based sign recognition.
"""

import io
import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from google import genai
from PIL import Image

from src.rag.legal_graph_rag import LegalGraphRAG
from src.rag.legal_utils import public_asset_path, record_image_paths
from src.rag.model_policy import generate_content_with_fallback

# --- Configuration & Initialization ---
load_dotenv(override=False)
logger = logging.getLogger("traffic_law_api")

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# --- FastAPI App Setup ---
app = FastAPI(
    title="Luật Giao Thông AI",
    description="Hệ thống hỏi đáp pháp luật giao thông đường bộ Việt Nam.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for document images and sign assets
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
processed_dir = os.path.join(project_root, "data", "processed")
if os.path.exists(processed_dir):
    app.mount("/processed", StaticFiles(directory=processed_dir), name="processed")

# --- Dependency Providers ---
@lru_cache(maxsize=1)
def get_rag() -> LegalGraphRAG:
    """Provides a singleton instance of the RAG engine."""
    enable_reranker = os.getenv("RAG_ENABLE_RERANKER", "false").lower() in {"1", "true", "yes", "on"}
    return LegalGraphRAG(
        "data/processed",
        graph_path="data/graph/legal_graph.json",
        use_reranker=enable_reranker,
    )

# --- Helper Functions ---
def _traffic_sign_query_hints(description: str, user_query: str = "") -> str:
    """Generates visual and legal hints for sign-related queries."""
    text = f"{description} {user_query}".lower()
    hints = ["QCVN 41:2024", "Thông tư 51/2024", "biển báo cấm", "Phụ lục B"]
    
    sign_hints = []
    if re.search(r"\bP\s*\.?\s*\d{2,3}[a-zđ]?\b", text, re.IGNORECASE):
        sign_hints.extend(re.findall(r"\bP\s*\.?\s*\d{2,3}[a-zđ]?\b", f"{description} {user_query}", re.IGNORECASE))
    
    mappings = {
        "ngược chiều": "P.102", "no entry": "P.102", "thanh ngang": "P.102",
        "đường cấm": "P.101", "ô tô": "P.103a", "xe máy": "P.104",
        "xe tải": "P.106a", "người đi bộ": "P.112", "rẽ trái": "P.123a",
        "rẽ phải": "P.123b", "quay đầu": "P.124a", "vượt": "P.125",
        "tốc độ": "P.127", "dừng": "P.130", "đỗ": "P.131",
    }
    for phrase, code in mappings.items():
        if phrase in text: sign_hints.append(code)
        
    deduped = list(dict.fromkeys(x.strip() for x in [*hints, *sign_hints] if x and x.strip()))
    return ". ".join(deduped)

def _parse_vision_json(text: str) -> Dict[str, Any]:
    """Extracts structured JSON from Vision model output."""
    if not text: return {}
    cleaned = re.sub(r"```(?:json)?", "", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback regex extraction
        codes = re.findall(r"\b(?:P|W|R|I|S|IE)\.?\d{2,3}[a-zđ]?\b", text, re.IGNORECASE)
        return {"candidate_codes": list(dict.fromkeys(codes)), "is_traffic_sign": bool(codes)}

def _references(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Formats source records for API metadata."""
    def images_for_doc(doc: Dict[str, Any]) -> List[str]:
        return [public_asset_path(path) for path in record_image_paths(doc)]

    references = []
    for d in docs:
        images = images_for_doc(d)
        references.append({
            "source_chunk_id": d.get("source_chunk_id"),
            "modality": d.get("rag_modality"),
            "legal_reference": d.get("legal_reference"),
            "image": images[0] if images else "",
            "images": images,
            "retrieval_reasons": d.get("retrieval_reasons", []),
            "retrieval_score": d.get("retrieval_score"),
        })
    return references

def _context_images(docs: List[Dict[str, Any]]) -> List[str]:
    images: List[str] = []
    seen = set()
    for doc in docs:
        for path in record_image_paths(doc):
            public = public_asset_path(path)
            if public and public not in seen:
                seen.add(public)
                images.append(public)
    return images

# --- API Endpoints ---
@app.get("/health")
async def health():
    return {"status": "ok", "rag_loaded": get_rag.cache_info().currsize > 0}

@app.post("/chat/analyze")
async def chat_analyze(query: str = Form(...), history: str = Form("[]")):
    """Analyzes query complexity and identified slots."""
    try:
        rag = get_rag()
        try:
            chat_history = json.loads(history)
        except Exception:
            chat_history = []
            
        search_query = query
        if chat_history and rag.client is not None:
            search_query = _condense_query(rag.client, query, chat_history)
            
        analysis = rag.analyze_query(search_query)
        return {
            "query": query,
            "condensed_query": search_query if search_query != query else None,
            "analysis": analysis
        }
    except Exception as e:
        logger.exception("Error in /chat/analyze")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/text')
async def chat_text(query: str = Form(...), history: str = Form("[]")):
    try:
        rag = get_rag()
        try:
            chat_history = json.loads(history)
        except Exception:
            chat_history = []
            
        search_query = query
        if chat_history and rag.client is not None:
            search_query = _condense_query(rag.client, query, chat_history)
            logger.info("Search Query: %s", search_query)

        result = rag.query_adaptive(search_query)
        ans, docs = result["answer"], result["contexts"]
        analysis = result.get("query_analysis") or rag.analyze_query(search_query)
        extra = result.get("metadata") or {}

        images = _context_images(docs)
        return {
            "answer": ans,
            "condensed_query": search_query if search_query != query else None,
            "query_analysis": analysis,
            "images": images,
            "reference_images": images,
            "references": _references(docs),
            "metadata": extra
        }
    except Exception as e:
        logger.exception("Error in /chat/text")
        raise HTTPException(status_code=500, detail=str(e))

def _condense_query(client: Any, current_query: str, chat_history: List[Dict]) -> str:
    """Rewrite query based on history."""
    if not chat_history: return current_query
    history_text = "\n".join([f"{m.get('role', 'user').upper()}: {m.get('content', '')[:300]}" for m in chat_history[-3:]])
    prompt = (
        "Bạn là hệ thống tóm tắt ngữ cảnh cho Luật Giao Thông.\n"
        "Viết lại câu hỏi mới nhất thành một câu hỏi ĐỘC LẬP, ĐẦY ĐỦ NGỮ CẢNH.\n"
        f"Lịch sử:\n{history_text}\n"
        f"Câu hỏi mới: {current_query}\n"
        "Viết lại:"
    )
    try:
        res, _model = generate_content_with_fallback(
            client,
            contents=[prompt],
            env_names=("RAG_ANSWER_MODEL",),
            logger=logger,
            label="Condense query",
        )
        return (res.text or current_query).strip()
    except Exception: return current_query

@app.post('/chat/sign')
async def chat_sign(query: str = Form(...)):
    try:
        rag = get_rag()
        docs = rag.retrieve_sign(query, top_k=8)
        ans = rag.generate_answer(query, docs)
        images = _context_images(docs)
        return {
            "answer": ans,
            "reference_images": images,
            "references": _references(docs),
        }
    except Exception as e:
        logger.exception("Error in /chat/sign")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/table')
async def chat_table(query: str = Form(...)):
    try:
        rag = get_rag()
        docs = rag.retrieve_table(query, top_k=8)
        ans = rag.generate_answer(query, docs)
        images = _context_images(docs)
        return {
            "answer": ans,
            "table_images": images,
            "references": _references(docs),
        }
    except Exception as e:
        logger.exception("Error in /chat/table")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/chat/image')
async def chat_image(image: UploadFile = File(...), query: str = Form("")):
    """Multimodal sign identification."""
    try:
        if not client: raise HTTPException(status_code=500, detail="Gemini client not initialized.")
        image_bytes = await image.read()
        user_img = Image.open(io.BytesIO(image_bytes))
        
        vision_prompt = (
            "Nhận diện biển báo giao thông Việt Nam theo QCVN 41:2024. "
            "Trả về JSON: {\"is_traffic_sign\": bool, \"candidate_codes\": [string], \"confidence\": float, \"raw_description\": string}."
        )
        
        res, _model = generate_content_with_fallback(
            client,
            contents=[vision_prompt, user_img],
            env_names=("RAG_VISION_MODEL",),
            vision=True,
            logger=logger,
            label="Vision sign recognition",
        )
        vision = _parse_vision_json(res.text or "")
        
        visual_desc = vision.get("raw_description", "")
        codes = vision.get("candidate_codes", [])
        
        if not vision.get("is_traffic_sign", True) and float(vision.get("confidence", 0)) < 0.4:
            return {"answer": "Không nhận diện được biển báo giao thông trong ảnh.", "references": []}

        sign_hints = _traffic_sign_query_hints(" ".join([visual_desc, *[str(c) for c in codes]]), query)
        final_query = f"{sign_hints}. {' '.join(str(c) for c in codes)}. {visual_desc}. {query}"
        rag = get_rag()
        result = rag.query_adaptive(final_query)
        ans, docs = result["answer"], result["contexts"]
        images = _context_images(docs)
        return {
            "answer": ans,
            "vision": vision,
            "query_analysis": result.get("query_analysis"),
            "metadata": result.get("metadata"),
            "reference_images": images,
            "references": _references(docs),
        }
    except Exception as e:
        logger.exception("Error in /chat/image")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
