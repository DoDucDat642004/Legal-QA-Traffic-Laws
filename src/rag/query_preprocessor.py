from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from src.rag.legal_utils import SIGN_CODE_RE, ascii_lower
from src.rag.model_policy import generate_content_with_fallback

logger = logging.getLogger("query_preprocessor")


@dataclass
class PreparedQuery:
    """Normalized query passed to planner/retriever."""

    original_query: str
    effective_query: str
    reason: str = "unchanged"
    used_llm: bool = False
    history_summary: str = ""
    missing_data_hints: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    @property
    def was_preprocessed(self) -> bool:
        return self.effective_query.strip() != self.original_query.strip()

    def public_payload(self) -> Dict[str, Any]:
        return {
            "was_preprocessed": self.was_preprocessed,
            "reason": self.reason,
            "used_llm": self.used_llm,
            "effective_query": self.effective_query,
            "history_summary": self.history_summary,
            "missing_data_hints": self.missing_data_hints,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def prepare_chat_query(client: Any, current_query: str, chat_history: Sequence[Dict[str, Any]] | None = None) -> PreparedQuery:
    """Builds a standalone, bounded query from user input and chat history."""

    original_query = _clean_text(current_query)
    history_items = _normalize_history(chat_history or [])
    history_text = _format_history(history_items, max_chars=_env_int("RAG_HISTORY_FORMAT_MAX_CHARS", 6000, 1000, 30000))
    stats = {
        "original_chars": len(original_query),
        "original_words": _word_count(original_query),
        "history_messages": len(history_items),
        "history_chars": len(history_text),
        "estimated_input_tokens": _rough_tokens(f"{history_text}\n{original_query}"),
    }
    base_hints = missing_data_hints(original_query)

    reason = _preprocess_reason(original_query, history_text)
    if not reason:
        return PreparedQuery(
            original_query=original_query,
            effective_query=original_query,
            missing_data_hints=base_hints,
            stats=stats,
        )

    fallback_query = _deterministic_prepare(original_query, history_items)
    if client is None or not _env_bool("RAG_ENABLE_QUERY_PREPROCESSOR_LLM", True):
        return PreparedQuery(
            original_query=original_query,
            effective_query=fallback_query,
            reason=reason,
            used_llm=False,
            history_summary=_history_summary(history_items),
            missing_data_hints=_dedupe([*base_hints, *missing_data_hints(fallback_query)]),
            warnings=["llm_preprocessor_unavailable"],
            stats={**stats, "effective_chars": len(fallback_query)},
        )

    try:
        llm_payload = _llm_prepare(client, original_query, history_items, fallback_query)
        effective_query = _bounded_effective_query(llm_payload.get("standalone_query") or fallback_query)
        history_summary = _clean_text(llm_payload.get("history_summary") or _history_summary(history_items))
        llm_hints = _string_list(llm_payload.get("missing_data_hints"))
        warnings = _string_list(llm_payload.get("warnings"))
        return PreparedQuery(
            original_query=original_query,
            effective_query=effective_query,
            reason=reason,
            used_llm=True,
            history_summary=history_summary,
            missing_data_hints=_dedupe([*base_hints, *missing_data_hints(effective_query), *llm_hints]),
            warnings=warnings,
            stats={**stats, "effective_chars": len(effective_query)},
        )
    except Exception as exc:
        logger.warning("Query preprocessing LLM failed; using deterministic fallback: %s", exc)
        return PreparedQuery(
            original_query=original_query,
            effective_query=fallback_query,
            reason=reason,
            used_llm=False,
            history_summary=_history_summary(history_items),
            missing_data_hints=_dedupe([*base_hints, *missing_data_hints(fallback_query)]),
            warnings=["llm_preprocessor_failed"],
            stats={**stats, "effective_chars": len(fallback_query)},
        )


def missing_data_hints(query: str) -> List[str]:
    """Returns deterministic hints for missing facts that affect legal precision."""

    qa = ascii_lower(query)
    hints: List[str] = []
    vehicle_terms = [
        "o to",
        "xe hoi",
        "xe con",
        "xe tai",
        "xe khach",
        "xe may",
        "mo to",
        "gan may",
        "may chuyen dung",
        "xe dap",
        "tho so",
    ]
    penalty_like = any(term in qa for term in ["phat", "xu phat", "muc phat", "vi pham", "bi gi", "xu ly", "tru diem", "tuoc"])
    vehicle_like = any(term in qa for term in ["xe", "phuong tien", "chay", "toc do", "bien", "p127", "p.127", "tat ca", "toan bo"])
    if penalty_like and vehicle_like and not any(term in qa for term in vehicle_terms):
        hints.append("Loại phương tiện chưa rõ: ô tô, mô tô/xe gắn máy, xe máy chuyên dùng, xe đạp/xe thô sơ.")

    if any(term in qa for term in ["toc do", "qua toc", "vuot toc", "p127", "p.127"]) and not re.search(r"\d+(?:[.,]\d+)?\s*km/?h", qa):
        hints.append("Thiếu ngưỡng tốc độ: tốc độ thực tế, tốc độ cho phép/ghi trên biển, loại đường hoặc khu vực.")

    if any(term in qa for term in ["nong do con", "hoi con", "ruou bia", "say xin", "co con"]) and not re.search(r"\d+(?:[.,]\d+)?", qa):
        hints.append("Thiếu ngưỡng nồng độ cồn trong máu hoặc hơi thở nếu cần chốt đúng khung xử phạt.")

    if any(term in qa for term in ["tai nan", "gay tai nan"]):
        hints.append("Thiếu hậu quả tai nạn: thương tích, thiệt hại tài sản, có rời hiện trường hoặc cứu giúp người bị nạn không.")

    broad_violation = penalty_like and any(term in qa for term in ["chay xe vi pham", "vi pham giao thong", "di sai", "lam sai", "bi phat sao"])
    known_behavior = any(
        term in qa
        for term in [
            "den do",
            "toc do",
            "nong do con",
            "mu bao hiem",
            "nguoc chieu",
            "dien thoai",
            "ma tuy",
            "tai nan",
            "dung xe",
            "do xe",
            "vuot",
        ]
    )
    if broad_violation and not known_behavior:
        hints.append("Hành vi vi phạm còn quá rộng; hệ thống sẽ bao phủ các khả năng liên quan nhưng cần hành vi cụ thể để chốt mức áp dụng.")

    if ("bien" in qa or "vach" in qa) and not SIGN_CODE_RE.search(query or "") and "anh" not in qa and "hinh" not in qa:
        hints.append("Thiếu mã biển/vạch hoặc ảnh rõ màu sắc, hình dạng, ký hiệu để xác định chính xác.")

    if any(term in qa for term in ["hay vi pham", "pho bien", "thuong gap"]):
        hints.append("Hệ thống chỉ có dữ liệu văn bản pháp luật; không có dataset số vụ vi phạm thực tế nếu người dùng hỏi tần suất ngoài đời.")

    return _dedupe(hints)


def _llm_prepare(client: Any, current_query: str, history_items: Sequence[Dict[str, str]], fallback_query: str) -> Dict[str, Any]:
    query_input = _salient_text(
        current_query,
        max_chars=_env_int("RAG_QUERY_PREP_INPUT_MAX_CHARS", 12000, 2000, 60000),
    )
    history_text = _format_history(
        history_items,
        max_chars=_env_int("RAG_QUERY_PREP_HISTORY_MAX_CHARS", 7000, 1000, 30000),
    )
    prompt = (
        "Bạn là lớp chuẩn hóa truy vấn cho hệ thống RAG pháp luật giao thông Việt Nam.\n"
        "Nhiệm vụ: viết lại câu hỏi mới nhất thành một truy vấn độc lập, ngắn gọn nhưng đủ dữ kiện để planner phân tích.\n"
        "Không trả lời pháp lý. Không tự điền dữ kiện thiếu. Nếu thiếu dữ kiện, ghi vào missing_data_hints.\n"
        "Phải giữ: chủ thể, loại phương tiện, hành vi, diễn biến theo thời gian, địa điểm/loại đường, biển báo/mã biển, "
        "tốc độ/nồng độ/số liệu, hậu quả, văn bản/điều/khoản/điểm, và chính xác yêu cầu người dùng muốn tra cứu.\n"
        "Nếu có nhiều hành vi hoặc nhiều nhánh, viết thành các mệnh đề rõ ràng, phân tách bằng dấu chấm phẩy.\n"
        "Nếu lịch sử mâu thuẫn câu hỏi mới, ưu tiên câu hỏi mới và chỉ dùng lịch sử để giải tham chiếu như 'trường hợp đó'.\n"
        "Chỉ trả về JSON object, không markdown, schema:\n"
        "{\n"
        '  "standalone_query": "truy vấn độc lập <= 2600 ký tự",\n'
        '  "history_summary": "ngữ cảnh lịch sử dùng được, <= 700 ký tự",\n'
        '  "missing_data_hints": ["dữ kiện thiếu quan trọng"],\n'
        '  "warnings": ["cảnh báo nếu phải lược bớt dữ kiện"]\n'
        "}\n"
        f"Lịch sử hội thoại đã rút gọn:\n{history_text or '(không có)'}\n\n"
        f"Câu hỏi mới nhất đã rút tín hiệu pháp lý:\n{query_input}\n\n"
        f"Fallback deterministic để tham khảo nếu input quá dài:\n{fallback_query}"
    )
    config = None
    try:
        from google.genai import types

        config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=1400)
    except Exception:
        pass
    response, _model = generate_content_with_fallback(
        client,
        contents=[prompt],
        config=config,
        env_names=("RAG_QUERY_PREP_MODEL", "RAG_CONDENSE_MODEL", "RAG_PLANNER_MODEL"),
        task="condense",
        logger=logger,
        label="Prepare chat query",
    )
    parsed = _parse_json_object(response.text or "")
    if parsed:
        return parsed
    text = _clean_text(response.text or "")
    return {"standalone_query": text} if text else {}


def _preprocess_reason(query: str, history_text: str) -> str:
    query_chars = len(query)
    query_words = _word_count(query)
    combined_tokens = _rough_tokens(f"{history_text}\n{query}")
    if history_text:
        if query_chars >= _env_int("RAG_LONG_QUERY_CHARS", 3200, 800, 50000) or query_words >= _env_int("RAG_LONG_QUERY_WORDS", 550, 120, 10000):
            return "history_and_long_query"
        return "history_context"
    if query_chars >= _env_int("RAG_LONG_QUERY_CHARS", 3200, 800, 50000):
        return "long_query"
    if query_words >= _env_int("RAG_LONG_QUERY_WORDS", 550, 120, 10000):
        return "long_query"
    if combined_tokens >= _env_int("RAG_PREPROCESS_COMBINED_TOKEN_THRESHOLD", 1400, 300, 30000):
        return "large_input"
    return ""


def _deterministic_prepare(query: str, history_items: Sequence[Dict[str, str]]) -> str:
    max_chars = _env_int("RAG_PREPARED_QUERY_MAX_CHARS", 2600, 600, 12000)
    history_budget = min(900, max(0, max_chars // 3))
    query_budget = max(500, max_chars - history_budget - 120)
    compact_query = _salient_text(query, max_chars=query_budget)
    history_summary = _history_summary(history_items, max_chars=history_budget)
    if history_summary:
        combined = f"Ngữ cảnh hội thoại liên quan: {history_summary}\nCâu hỏi hiện tại cần trả lời: {compact_query}"
    else:
        combined = compact_query
    return _bounded_effective_query(combined)


def _bounded_effective_query(value: str) -> str:
    max_chars = _env_int("RAG_PREPARED_QUERY_MAX_CHARS", 2600, 600, 12000)
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return _salient_text(text, max_chars=max_chars)


def _salient_text(text: str, *, max_chars: int) -> str:
    text = _clean_text(text)
    if len(text) <= max_chars:
        return text
    pieces = _sentence_pieces(text)
    if not pieces:
        return text[:max_chars].strip()
    scored = []
    for idx, piece in enumerate(pieces):
        score = _salience_score(piece)
        if idx == 0:
            score += 3
        if idx == len(pieces) - 1:
            score += 3
        scored.append((score, idx, piece))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1]))
    selected_ranked = ranked[: _env_int("RAG_SALIENT_SENTENCE_LIMIT", 24, 6, 80)]
    min_score = 2 if len(text) > max_chars * 2 else 0
    keep_indices = {idx for score, idx, _piece in selected_ranked if score >= min_score}
    if len(keep_indices) < 2:
        keep_indices = {idx for _score, idx, _piece in selected_ranked[:6]}
    selected = [piece for idx, piece in enumerate(pieces) if idx in keep_indices]
    out: List[str] = []
    total = 0
    for piece in selected:
        add_len = len(piece) + (2 if out else 0)
        if total + add_len > max_chars:
            remaining = max_chars - total - (2 if out else 0)
            if remaining > 80:
                out.append(piece[:remaining].rstrip())
            break
        out.append(piece)
        total += add_len
    return " ".join(out).strip() or text[:max_chars].strip()


def _sentence_pieces(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    pieces: List[str] = []
    for item in raw:
        item = item.strip(" \t\r\n-•")
        if not item:
            continue
        words = item.split()
        if len(words) <= 85:
            pieces.append(item)
            continue
        for start in range(0, len(words), 60):
            pieces.append(" ".join(words[start : start + 60]))
    return pieces


def _salience_score(text: str) -> int:
    qa = ascii_lower(text)
    score = 0
    if SIGN_CODE_RE.search(text):
        score += 5
    if re.search(r"\b(?:dieu|khoan|diem)\s+\w+", qa) or re.search(r"\d+/\d{4}|qcvn|nghi dinh|thong tu|luat", qa):
        score += 5
    if re.search(r"\d+(?:[.,]\d+)?\s*(?:km/?h|mg|ml|nam|tuoi|dong|trieu|%)?", qa):
        score += 3
    for terms, value in [
        (["o to", "xe may", "mo to", "gan may", "xe dap", "may chuyen dung", "phuong tien"], 3),
        (["phat", "xu phat", "tru diem", "tuoc", "tam giu", "vi pham"], 4),
        (["den do", "toc do", "nong do con", "mu bao hiem", "nguoc chieu", "dien thoai", "tai nan"], 4),
        (["bien bao", "vach", "p127", "p.127", "qcvn"], 3),
        (["sau do", "dong thoi", "cung luc", "tinh huong", "truong hop", "nga tu", "giao nhau"], 3),
        (["hoi", "can tra cuu", "bao nhieu", "co duoc", "the nao", "phai lam gi"], 2),
    ]:
        if any(term in qa for term in terms):
            score += value
    return score


def _normalize_history(history: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    max_messages = _env_int("RAG_HISTORY_MAX_MESSAGES", 8, 0, 30)
    normalized: List[Dict[str, str]] = []
    for item in list(history)[-max_messages:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user").strip().lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = _clean_text(str(item.get("content") or ""))
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _format_history(history_items: Sequence[Dict[str, str]], *, max_chars: int) -> str:
    if not history_items:
        return ""
    per_message = _env_int("RAG_HISTORY_MESSAGE_MAX_CHARS", 900, 120, 4000)
    lines = []
    for item in history_items:
        role = item.get("role", "user").upper()
        content = _salient_text(item.get("content", ""), max_chars=per_message)
        lines.append(f"{role}: {content}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return _salient_text(text, max_chars=max_chars)


def _history_summary(history_items: Sequence[Dict[str, str]], *, max_chars: int = 700) -> str:
    if not history_items:
        return ""
    user_turns = [item["content"] for item in history_items if item.get("role") == "user"]
    assistant_turns = [item["content"] for item in history_items if item.get("role") == "assistant"]
    bits = []
    if user_turns:
        bits.append("Câu hỏi trước: " + _salient_text(user_turns[-1], max_chars=max_chars // 2))
    if assistant_turns:
        bits.append("Trả lời trước: " + _salient_text(assistant_turns[-1], max_chars=max_chars // 2))
    return _salient_text(" ".join(bits), max_chars=max_chars)


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


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item or "").strip()][:12]


def _dedupe(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = _clean_text(value)
        key = ascii_lower(item)
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _word_count(value: str) -> int:
    return len(re.findall(r"\S+", value or ""))


def _rough_tokens(value: str) -> int:
    text = value or ""
    return max(_word_count(text), len(text) // 4)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default
    return max(minimum, min(value, maximum))
