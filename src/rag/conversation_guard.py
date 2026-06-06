from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.rag.legal_utils import ascii_lower


SUPPORTED_SOURCE_NAMES = [
    "Nghị định 168/2024/NĐ-CP",
    "Luật Đường bộ 2024",
    "Luật Trật tự an toàn giao thông đường bộ 2024",
    "QCVN 41:2024 và Thông tư 51/2024/TT-BGTVT",
    "Thông tư 35/2024/TT-BGTVT",
    "Nghị định 336/2025/NĐ-CP",
]


@dataclass(frozen=True)
class ConversationalResponse:
    intent: str
    answer: str
    reason: str
    facets: List[str] = field(default_factory=list)

    def query_analysis(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "facets": self.facets or ["system_interaction"],
            "difficulty": "easy",
            "difficulty_label": "Dễ",
            "difficulty_score": 0,
            "difficulty_reason": self.reason,
            "max_wait_seconds": 0,
            "retrieval_budget": {},
            "evidence_slots": [],
            "decomposition_source": "conversation_guard",
            "plan": {
                "intent": self.intent,
                "confidence": 1.0,
                "sign_codes": [],
                "expected_modalities": [],
                "subquestions": [],
                "plan_source": "conversation_guard",
                "difficulty_hint": "easy",
                "analysis_notes": [self.reason],
            },
        }

    def metadata(self) -> Dict[str, Any]:
        return {
            "route": "conversation_guard",
            "conversation_intent": self.intent,
            "reason": self.reason,
            "retrieval_skipped": True,
            "scope": "vietnam_road_traffic_law",
        }


def route_conversational_query(query: str) -> ConversationalResponse | None:
    """Routes product-level chat, greetings, and clear off-topic turns before RAG retrieval."""

    raw = (query or "").strip()
    if not raw:
        return ConversationalResponse(
            intent="scope_invite",
            answer=_scope_invite_answer(),
            reason="empty_or_missing_query",
        )

    qa = ascii_lower(raw)
    compact = _compact(qa)

    if _looks_like_capability_question(compact):
        return ConversationalResponse(
            intent="capability_intro",
            answer=_capability_answer(),
            reason="user_asked_about_system_capabilities_or_sources",
        )

    if _has_traffic_law_signal(compact):
        return None

    if _looks_like_thanks(compact):
        return ConversationalResponse(
            intent="thanks",
            answer=(
                "Không có gì. Khi cần tra cứu pháp luật giao thông đường bộ, bạn cứ gửi tình huống, "
                "loại xe, hành vi, biển báo hoặc ảnh biển báo; mình sẽ đối chiếu căn cứ phù hợp."
            ),
            reason="short_thanks",
        )

    if _looks_like_greeting(compact):
        return ConversationalResponse(
            intent="greeting",
            answer=_greeting_answer(),
            reason="short_greeting",
        )

    if _looks_like_help_invite(compact):
        return ConversationalResponse(
            intent="scope_invite",
            answer=_scope_invite_answer(),
            reason="user_requested_help_without_traffic_law_facts",
        )

    if _clearly_out_of_scope(compact):
        return ConversationalResponse(
            intent="out_of_scope",
            answer=_out_of_scope_answer(),
            reason="outside_vietnam_road_traffic_law_scope",
            facets=["out_of_scope"],
        )

    return None


def _capability_answer() -> str:
    sources = "; ".join(SUPPORTED_SOURCE_NAMES)
    return (
        "Mình là trợ lý tra cứu pháp luật giao thông đường bộ Việt Nam. Mình hỗ trợ bạn tìm và giải thích căn cứ từ "
        f"các nguồn dữ liệu đã trích xuất như: {sources}.\n\n"
        "Bạn có thể hỏi về mức phạt, trừ điểm GPLX, tước GPLX, quy tắc đi đường, quyền ưu tiên, biển báo/vạch kẻ, "
        "thủ tục GPLX, dữ liệu bảng/phụ lục, ảnh trang gốc hoặc tình huống vi phạm thực tế. "
        "Khi trả lời, mình sẽ cố gắng nêu kết luận kèm Điều/Khoản/Điểm, tên văn bản và dữ kiện còn thiếu nếu cần.\n\n"
        "Mình chỉ tập trung vào pháp luật giao thông đường bộ, không trò chuyện lan man sang chủ đề khác và không thay thế tư vấn pháp lý chính thức."
    )


def _greeting_answer() -> str:
    return (
        "Chào bạn. Mình có thể giúp tra cứu pháp luật giao thông đường bộ Việt Nam: mức phạt, trừ điểm GPLX, "
        "biển báo, quy tắc đi đường, thủ tục GPLX hoặc tình huống vi phạm cụ thể.\n\n"
        "Bạn chỉ cần mô tả loại xe, hành vi, địa điểm/biển báo, tốc độ hoặc nồng độ nếu có; mình sẽ tìm căn cứ phù hợp trong dữ liệu pháp luật giao thông."
    )


def _scope_invite_answer() -> str:
    return (
        "Bạn cứ mô tả tình huống giao thông đường bộ cần tra cứu. Để trả lời chính xác hơn, hãy nêu loại xe, "
        "hành vi, biển báo/vạch kẻ, tốc độ, nồng độ cồn hoặc văn bản/điều khoản nếu bạn đã biết."
    )


def _out_of_scope_answer() -> str:
    return (
        "Mình chỉ hỗ trợ tư vấn và tìm kiếm thông tin về pháp luật giao thông đường bộ Việt Nam, nên mình không trả lời chủ đề này.\n\n"
        "Bạn có thể hỏi về mức phạt, quy tắc đi đường, GPLX, biển báo/vạch kẻ, thủ tục hoặc tình huống vi phạm giao thông đường bộ."
    )


def _compact(value: str) -> str:
    value = re.sub(r"[_`~!@#$%^&*()+={}\[\]|\\:;\"'<>,?]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _looks_like_capability_question(qa: str) -> bool:
    capability_terms = [
        "ban co the giup gi",
        "co the giup gi",
        "giup gi cho toi",
        "ban giup duoc gi",
        "ban lam duoc gi",
        "ban co the lam gi",
        "he thong nay lam gi",
        "he thong lam gi",
        "tro ly nay lam gi",
        "ban ho tro gi",
        "ho tro nguoi dung nhu the nao",
        "gioi thieu ve ban",
        "ban la ai",
        "pham vi ho tro",
        "hoi gi duoc",
        "co the hoi gi",
    ]
    source_terms = [
        "nguon du lieu",
        "du lieu tu dau",
        "truy xuat tu dau",
        "lay thong tin tu dau",
        "lay nguon tu dau",
        "tim kiem tu dau",
        "tra cuu tu dau",
        "can cu tu dau",
        "co nhung nguon nao",
    ]
    return any(term in qa for term in [*capability_terms, *source_terms])


def _looks_like_greeting(qa: str) -> bool:
    if len(qa.split()) > 5:
        return False
    greetings = {
        "hi",
        "hello",
        "hey",
        "alo",
        "chao",
        "xin chao",
        "chao ban",
        "chao ad",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if qa in greetings:
        return True
    return bool(re.fullmatch(r"(xin )?chao( ban| anh| chi| em)?", qa))


def _looks_like_thanks(qa: str) -> bool:
    if len(qa.split()) > 6:
        return False
    thanks_terms = [
        "cam on",
        "cam on ban",
        "thanks",
        "thank you",
        "ok cam on",
        "ok thanks",
        "tks",
    ]
    return qa in thanks_terms or any(qa.startswith(f"{term} ") for term in thanks_terms)


def _looks_like_help_invite(qa: str) -> bool:
    if len(qa.split()) > 8:
        return False
    invite_terms = [
        "toi can hoi",
        "cho toi hoi",
        "minh can hoi",
        "giup toi voi",
        "tu van giup toi",
        "ban oi",
        "ad oi",
        "hoi chut",
    ]
    return any(term == qa or qa.startswith(f"{term} ") for term in invite_terms)


def _has_traffic_law_signal(qa: str) -> bool:
    traffic_terms = [
        "giao thong",
        "duong bo",
        "atgt",
        "trat tu an toan giao thong",
        "xe may",
        "mo to",
        "gan may",
        "o to",
        "xe hoi",
        "xe con",
        "xe tai",
        "xe khach",
        "xe dap",
        "xe tho so",
        "phuong tien",
        "gplx",
        "giay phep lai xe",
        "bang lai",
        "hang a1",
        "hang a2",
        "bien bao",
        "bien cam",
        "vach ke",
        "den do",
        "den vang",
        "den tin hieu",
        "toc do",
        "qua toc",
        "vuot toc",
        "nong do con",
        "ruou bia",
        "mu bao hiem",
        "nguoc chieu",
        "lan duong",
        "cao toc",
        "tai nan giao thong",
        "xe uu tien",
        "nhuong duong",
        "nghi dinh 168",
        "168/2024",
        "168-2024",
        "nghi dinh 336",
        "336/2025",
        "336-2025",
        "qcvn 41",
        "thong tu 35",
        "35/2024",
        "thong tu 51",
        "51/2024",
        "luat duong bo",
        "luat trat tu",
    ]
    if any(term in qa for term in traffic_terms):
        return True
    return bool(re.search(r"\b(?:p|w|r|i|s|dp|ie)\s*\.?\s*\d{2,3}[a-zd]?\b", qa))


def _clearly_out_of_scope(qa: str) -> bool:
    if len(qa.split()) <= 3:
        return False
    non_traffic_legal_terms = [
        "luat doanh nghiep",
        "luat dat dai",
        "dat dai",
        "hon nhan",
        "ly hon",
        "thua ke",
        "hop dong lao dong",
        "bao hiem xa hoi",
        "thue thu nhap",
        "hinh su",
        "dan su",
        "so huu tri tue",
    ]
    unrelated_terms = [
        "nau an",
        "mon an",
        "thoi tiet",
        "gia vang",
        "chung khoan",
        "co phieu",
        "crypto",
        "lap trinh",
        "python",
        "javascript",
        "bong da",
        "lich thi dau",
        "du lich",
        "khach san",
        "y te",
        "thuoc",
        "benh",
        "suc khoe",
        "ke chuyen",
        "chuyen cuoi",
        "viet tho",
        "am nhac",
        "phim",
        "game",
        "marketing",
        "tieng anh",
    ]
    if any(term in qa for term in [*non_traffic_legal_terms, *unrelated_terms]):
        return True

    legal_but_unscoped = any(
        term in qa
        for term in [
            "phat",
            "xu phat",
            "bi phat",
            "vi pham",
            "xu ly",
            "bi gi",
            "bao nhieu tien",
            "tru diem",
            "tuoc",
            "quy dinh",
            "thu tuc",
            "ho so",
            "dieu luat",
        ]
    )
    if legal_but_unscoped:
        return False

    return not any(term in qa for term in ["luat", "nghi dinh", "thong tu", "quy chuan", "can cu", "dieu "])
