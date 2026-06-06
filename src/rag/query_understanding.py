from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from src.rag.legal_utils import ascii_lower
from src.rag.model_policy import generate_content_with_fallback


ALLOWED_UNDERSTANDING_FACETS = {
    "general",
    "rule",
    "penalty",
    "procedure",
    "sign",
    "table",
    "definition",
    "priority",
    "scenario",
    "source_image",
    "document_overview",
    "legal_detail",
    "aggregation",
    "out_of_scope",
}


def looks_like_colloquial_traffic_query(query: str) -> bool:
    """Detects everyday phrasing where LLM understanding is useful before rule planning."""

    qa = ascii_lower(query)
    if not qa:
        return False

    everyday_markers = [
        "co on",
        "co sao",
        "co bi sao",
        "co duoc khong",
        "duoc khong",
        "duoc ko",
        "co duoc",
        "thi sao",
        "ra sao",
        "the nao",
        "bi gi",
        "bi phat khong",
        "thoat phat",
        "co thoat",
        "lo ",
        "loi ",
        "toi ",
        "minh ",
        "hey ",
        "alo ",
        "ban oi",
        "ad oi",
    ]
    traffic_terms = [
        "xe",
        "o to",
        "xe tai",
        "xe khach",
        "xe may",
        "mo to",
        "gplx",
        "bang lai",
        "bien bao",
        "vach",
        "den do",
        "toc do",
        "nong do con",
        "ruou bia",
        "mu bao hiem",
        "nguoc chieu",
        "via he",
        "he pho",
        "long duong",
        "dung xe",
        "do xe",
        "nhuong duong",
        "tai nan",
        "giao thong",
    ]
    return any(marker in qa for marker in everyday_markers) and any(term in qa for term in traffic_terms)


def understand_query_with_llm(client: Any, query: str, fallback_summary: Dict[str, Any], *, logger: Any = None) -> Dict[str, Any]:
    """Uses an LLM as a bounded query-understanding layer; it must not answer legal questions."""

    prompt = _understanding_prompt(query, fallback_summary)
    config = None
    try:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=1800)
    except Exception:
        pass
    response, model = generate_content_with_fallback(
        client,
        contents=[prompt],
        config=config,
        env_names=("RAG_QUERY_UNDERSTANDING_MODEL", "RAG_PLANNER_MODEL", "RAG_AI_PLANNER_MODEL"),
        task="query_understanding",
        logger=logger,
        label="LLM query understanding",
    )
    data = _parse_json_object(response.text or "")
    if data:
        data["_model"] = model
    return data


def retrieval_queries_to_subquestions(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or item.get("search_query") or "").strip()
        if len(query) < 5:
            continue
        facet = ascii_lower(str(item.get("facet") or "general")).replace(" ", "_")
        if facet not in ALLOWED_UNDERSTANDING_FACETS:
            facet = "general"
        out.append(
            {
                "facet": facet,
                "query": query[:700],
                "priority": _bounded_int(item.get("priority"), default=idx + 1, min_value=1, max_value=20),
                "reason": str(item.get("reason") or "LLM hiểu truy vấn đời thường và sinh nhánh truy xuất.").strip()[:300],
                "must_answer": bool(item.get("must_answer", True)),
            }
        )
    return out


def sanitized_facets(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        facet = ascii_lower(str(item or "")).replace(" ", "_")
        if facet in ALLOWED_UNDERSTANDING_FACETS and facet not in out:
            out.append(facet)
    return out


def sanitized_sign_codes(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        code = re.sub(r"[\s.]+", "", str(item or "")).upper()
        if code and re.match(r"^(?:P|W|R|I|S|DP|IE)\d{2,3}[A-ZĐ]?$", code) and code not in out:
            out.append(code)
    return out[:8]


def _understanding_prompt(query: str, fallback_summary: Dict[str, Any]) -> str:
    fallback_json = json.dumps(fallback_summary or {}, ensure_ascii=False)
    return (
        "Bạn là lớp hiểu truy vấn cho hệ thống RAG pháp luật giao thông đường bộ Việt Nam.\n"
        "Nhiệm vụ của bạn KHÔNG phải trả lời pháp lý. Chỉ chuyển câu hỏi đời thường thành JSON để hệ thống truy xuất đúng nguồn.\n"
        "Hãy nhận diện intent, thực thể, dữ kiện thiếu và tạo các truy vấn truy xuất ngắn, chính xác.\n\n"
        "Phải hiểu các cách hỏi tự nhiên như: 'có ổn không', 'có sao không', 'được không', 'thì sao', "
        "'bị gì không', 'thoát phạt không', lời chào đứng đầu câu như 'Hey/Alo'.\n"
        "Nếu câu hỏi hỏi 'có ổn không/được không/có sao không' với một hành vi giao thông, thường cần cả rule và penalty.\n"
        "Nếu câu hỏi có xử phạt nhưng thiếu loại phương tiện, ghi missing_facts và tạo truy vấn bao phủ nhóm phương tiện phù hợp.\n"
        "Nếu rõ ngoài phạm vi luật giao thông đường bộ, đặt in_scope=false và retrieval_queries rỗng.\n"
        "Không bịa căn cứ, số tiền, điều khoản. Không xuất chain-of-thought.\n\n"
        "Schema JSON bắt buộc:\n"
        "{\n"
        '  "in_scope": true,\n'
        '  "intent": "general|penalty|procedure|sign|table|definition|priority|scenario|document_overview|legal_detail|aggregation|out_of_scope",\n'
        '  "confidence": 0.0,\n'
        '  "difficulty_hint": "easy|medium|hard",\n'
        '  "user_tone": "colloquial|formal|mixed",\n'
        '  "facets": ["rule", "penalty"],\n'
        '  "entities": {\n'
        '    "vehicle": "xe tải", "action": "đỗ xe", "location": "vỉa hè",\n'
        '    "sign_codes": [], "asks_legality": true, "asks_penalty": true,\n'
        '    "missing_facts": []\n'
        "  },\n"
        '  "retrieval_queries": [\n'
        '    {"facet": "rule", "query": "quy định dừng đỗ xe tải trên vỉa hè, hè phố, lòng đường", "priority": 1, "reason": "xác định hành vi có được phép không", "must_answer": true},\n'
        '    {"facet": "penalty", "query": "mức phạt xe tải đỗ trên vỉa hè theo Nghị định 168/2024/NĐ-CP", "priority": 2, "reason": "người dùng hỏi có sao không nên cần chế tài", "must_answer": true}\n'
        "  ],\n"
        '  "notes": ["ngắn gọn"]\n'
        "}\n\n"
        "Ví dụ:\n"
        "Q: Hey tôi đỗ xe tải ở vỉa hè có ổn không?\n"
        "A facets: rule, penalty; vehicle=xe tải; action=đỗ xe; location=vỉa hè; asks_legality=true; asks_penalty=true.\n"
        "Q: Lỡ chạy xe máy qua đèn đỏ thì sao?\n"
        "A facets: rule, penalty; vehicle=xe máy; action=vượt đèn đỏ.\n"
        "Q: Bạn có thể giúp gì?\n"
        "A in_scope=false, intent=out_of_scope, retrieval_queries=[] vì đây là câu hỏi sản phẩm, không phải truy vấn pháp luật.\n\n"
        f"Fallback rule plan hiện tại để tham khảo, không bắt buộc theo nếu thiếu ngữ cảnh đời thường:\n{fallback_json}\n\n"
        f"Câu hỏi người dùng: {query}"
    )


def _parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?", "", text or "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def _bounded_int(value: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(parsed, max_value))
