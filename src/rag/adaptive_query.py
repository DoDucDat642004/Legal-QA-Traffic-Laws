"""
Adaptive Query Analyzer for Vietnamese Traffic Law.

This module analyzes user queries to determine their legal intent, complexity, 
and required retrieval budget. It decomposes complex questions into 
independent 'evidence slots' for multi-stage retrieval.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.rag.legal_utils import SIGN_CODE_RE, ascii_lower, extract_explicit_legal_refs

# --- Logging Configuration ---
logger = logging.getLogger("AdaptiveQuestionAnalyzer")


@dataclass
class AdaptiveQueryProfile:
    """
    Holds the diagnostic profile of a user query.
    """
    raw_query: str
    normalized_query: str
    intent: str = "general"
    facets: List[str] = field(default_factory=list)
    difficulty: str = "easy"
    difficulty_label: str = "Dễ"
    difficulty_score: int = 0
    difficulty_reason: str = ""
    max_wait_seconds: int = 15
    evidence_slots: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_budget: Dict[str, int] = field(default_factory=dict)
    decomposition_source: str = "rule"

    def public_summary(self) -> Dict[str, Any]:
        """Provides a safe dictionary representation for API responses."""
        return {
            "intent": self.intent,
            "facets": self.facets,
            "difficulty": self.difficulty,
            "difficulty_label": self.difficulty_label,
            "difficulty_score": self.difficulty_score,
            "difficulty_reason": self.difficulty_reason,
            "max_wait_seconds": self.max_wait_seconds,
            "retrieval_budget": self.retrieval_budget,
            "evidence_slots": self.evidence_slots,
            "decomposition_source": self.decomposition_source
        }


class AdaptiveQuestionAnalyzer:
    """
    Analyzes legal queries to optimize retrieval strategies.
    Supports Laws, Decrees, Circulars, and Technical Regulations (QCVN).
    """

    # Keyword mappings for broad document identification
    DOC_KEYWORDS = {
        "Nghị định 168/2024/NĐ-CP": ["phat", "xu phat", "muc phat", "tru diem", "tuoc bang"],
        "Luật Trật tự ATGT 2024": ["quy tac", "tin hieu", "lan duong", "bien bao", "giay phep", "bang lai"],
        "QCVN 41:2024 (Thông tư 51/2024)": ["hinh dang", "mau sac", "y nghia bien", "vach ke duong", "dao an toan"],
        "Thông tư 35/2024/TT-BGTVT": ["dao tao", "sat hach", "cap bang", "ly thuyet", "thuc hanh"],
        "Nghị định 336/2025/NĐ-CP": ["kinh doanh van tai", "ben xe", "taxi", "thu phi"],
    }

    def analyze(self, query: str, plan: Any) -> AdaptiveQueryProfile:
        """
        Performs a multi-faceted analysis of the query.
        """
        q_norm = query.strip()
        qa = ascii_lower(q_norm)

        slots = self._decompose(q_norm, qa, plan)
        facets = self._facets_from_slots(slots)
        score, reasons = self._calculate_complexity(q_norm, qa, plan, facets, len(slots))
        
        difficulty, label, wait, budget = self._determine_budget(score)
        intent = getattr(getattr(plan, "intent", None), "value", None) or self._intent_from_facets(facets)
        source = getattr(plan, "plan_source", "rule") if plan else "rule"
        
        return AdaptiveQueryProfile(
            raw_query=query,
            normalized_query=q_norm,
            intent=intent,
            facets=facets,
            difficulty=difficulty,
            difficulty_label=label,
            difficulty_score=score,
            difficulty_reason="; ".join(reasons) or "Câu hỏi rõ ràng.",
            max_wait_seconds=wait,
            evidence_slots=slots,
            retrieval_budget=budget,
            decomposition_source=source,
        )

    def _calculate_complexity(
        self,
        q: str,
        qa: str,
        plan: Any,
        facets: List[str],
        slot_count: int,
    ) -> Tuple[int, List[str]]:
        """Heuristic complexity scoring based on length, entities, and keywords."""
        score = 0
        reasons: List[str] = []
        
        word_count = len(q.split())
        if word_count > 35:
            score += 4
            reasons.append("Câu hỏi rất dài, nhiều dữ kiện")
        elif word_count > 22:
            score += 3
            reasons.append("Câu hỏi dài, nhiều chi tiết")
        elif word_count > 12:
            score += 1

        meaningful_facets = [facet for facet in facets if facet != "general"]
        if len(meaningful_facets) >= 3:
            score += 4
            reasons.append("Cần kết hợp nhiều loại căn cứ pháp lý")
        elif len(meaningful_facets) == 2:
            score += 2
            reasons.append("Câu hỏi có nhiều nhánh pháp lý")

        if slot_count >= 4:
            score += 3
            reasons.append("Planner đã tách thành nhiều câu hỏi con")
        elif slot_count >= 2:
            score += 2
            reasons.append("Cần truy vấn tuần tự nhiều câu hỏi con")

        if SIGN_CODE_RE.search(q):
            score += 2
            reasons.append("Chứa mã hiệu biển báo")

        if extract_explicit_legal_refs(q):
            score += 2
            reasons.append("Có tham chiếu điều/khoản pháp luật cụ thể")

        if len(re.findall(r"\b\d+(?:[.,]\d+)?\b", qa)) >= 2:
            score += 1
            reasons.append("Có nhiều dữ kiện số cần đối chiếu")

        if any(k in qa for k in ["tong hop", "tong cong", "tong hau qua", "cong don", "vua", "dong thoi", "cung luc"]):
            score += 3
            reasons.append("Yêu cầu tổng hợp nhiều hành vi")

        if any(k in qa for k in ["neu", "truong hop", "so sanh", "khac nhau", "ngoai ra", "trong khi"]):
            score += 2
            reasons.append("Có điều kiện hoặc yêu cầu so sánh")

        if "scenario" in facets:
            score += 2
            reasons.append("Có tình huống thực tế cần bóc tách theo diễn biến")
        if "priority" in facets:
            score += 2
            reasons.append("Có yếu tố quyền ưu tiên/nhường đường")
        if "source_image" in facets:
            score += 1
            reasons.append("Cần truy xuất ảnh căn cứ từ văn bản gốc")

        difficulty_hint = str(getattr(plan, "difficulty_hint", "") or "").lower()
        if difficulty_hint == "hard":
            score = max(score, 6)
            reasons.append("AI planner đánh giá là câu hỏi khó")
        elif difficulty_hint == "medium":
            score = max(score, 3)
            reasons.append("AI planner đánh giá là câu hỏi trung bình")

        return min(score, 12), list(dict.fromkeys(reasons))

    def _determine_budget(self, score: int) -> Tuple[str, str, int, Dict[str, int]]:
        """Maps complexity score to retrieval resource allocation."""
        if score >= 5:
            return "hard", "Khó", 90, {"top_k": 40, "expand_depth": 4, "evidence_slot_top_k": 10, "max_contexts": 45, "max_images": 30}
        if score >= 3:
            return "medium", "Trung bình", 55, {"top_k": 24, "expand_depth": 3, "evidence_slot_top_k": 7, "max_contexts": 28, "max_images": 18}
        return "easy", "Dễ", 30, {"top_k": 14, "expand_depth": 2, "evidence_slot_top_k": 5, "max_contexts": 16, "max_images": 10}

    def _decompose(self, q: str, qa: str, plan: Any) -> List[Dict[str, Any]]:
        """Builds evidence slots from the planner, with rule fallback."""
        plan_slots = self._sanitize_slots(getattr(plan, "subquestions", None), fallback_query=q)
        if plan_slots:
            return plan_slots
        return self._rule_decompose(q, qa)

    def _rule_decompose(self, q: str, qa: str) -> List[Dict[str, Any]]:
        """Rules-based decomposition into evidence slots."""
        slots: List[Dict[str, Any]] = []

        if "bien" in qa and any(k in qa for k in ["hinh", "mau", "nhu the nao", "y nghia", "qcvn"]):
            slots.append({
                "facet": "sign",
                "query": f"Xác định biển báo/vạch kẻ, hình dạng, ý nghĩa và căn cứ áp dụng trong câu hỏi: {q}",
                "priority": 1,
                "reason": "Yêu cầu mô tả hình ảnh biển báo"
            })

        if "bang" in qa or any(k in qa for k in ["kich thuoc", "toc do", "hang gplx", "hang giay phep"]):
            slots.append({
                "facet": "table",
                "query": f"Tra cứu bảng/phụ lục/thông số kỹ thuật liên quan đến câu hỏi: {q}",
                "priority": 2,
                "reason": "Yêu cầu dữ liệu dạng bảng hoặc thông số"
            })

        if any(k in qa for k in ["anh", "hinh anh", "trang goc", "van ban goc", "can cu goc", "file goc", "scan", "crop"]):
            slots.append({
                "facet": "source_image",
                "query": f"Tìm ảnh trang gốc, hình, bảng hoặc phụ lục trong văn bản làm căn cứ trực quan cho câu hỏi: {q}",
                "priority": 6,
                "reason": "Yêu cầu ảnh căn cứ từ văn bản gốc"
            })

        if any(k in qa for k in ["xe uu tien", "quyen uu tien", "uu tien", "nhuong duong", "cuu thuong", "chua chay", "cong an", "quan su", "doan xe"]):
            slots.append({
                "facet": "priority",
                "query": f"Tra cứu quy định về quyền ưu tiên, thứ tự nhường đường và điều kiện áp dụng cho câu hỏi: {q}",
                "priority": 2,
                "reason": "Yêu cầu xác định trường hợp ưu tiên"
            })

        if any(k in qa for k in ["tinh huong", "truong hop", "neu", "khi", "dang", "sau do", "dong thoi", "cung luc", "nga tu", "giao nhau", "vong xuyen"]):
            slots.append({
                "facet": "scenario",
                "query": f"Tra cứu căn cứ xử lý tình huống thực tế nhiều bước, gồm chủ thể, hành vi, thời điểm và điều kiện trong câu hỏi: {q}",
                "priority": 1,
                "reason": "Cần bóc tách diễn biến thực tế trước khi kết luận"
            })

        if any(k in qa for k in ["co duoc", "duoc phep", "quy dinh", "lan duong", "den do", "nguoc chieu"]):
            slots.append({
                "facet": "rule",
                "query": f"Tra cứu quy tắc pháp lý và điều kiện áp dụng cho câu hỏi: {q}",
                "priority": 3,
                "reason": "Yêu cầu xác định quy định nền"
            })

        if any(k in qa for k in ["phat", "bao nhieu", "bi gi", "xu ly", "vi pham"]):
            behavior_slots = self._compound_penalty_slots(q, qa, start_priority=4)
            slots.extend(behavior_slots or [{
                "facet": "penalty",
                "query": f"Tra cứu mức phạt tiền, trừ điểm, tước giấy phép và biện pháp bổ sung cho hành vi trong câu hỏi: {q}",
                "priority": 4,
                "reason": "Yêu cầu tra cứu mức xử phạt",
            }])

        if any(k in qa for k in ["thu tuc", "ho so", "cap doi", "cap lai", "sat hach", "thoi han"]):
            slots.append({
                "facet": "procedure",
                "query": f"Tra cứu thủ tục, hồ sơ, thời hạn và cơ quan xử lý liên quan đến câu hỏi: {q}",
                "priority": 5,
                "reason": "Yêu cầu thủ tục hành chính"
            })

        if not slots:
            slots.append({"facet": "general", "query": q, "priority": 1, "reason": "Truy xuất tổng hợp"})

        return sorted(slots, key=lambda item: int(item.get("priority") or 1))

    def _sanitize_slots(self, value: Any, *, fallback_query: str) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        allowed_facets = {"general", "rule", "penalty", "procedure", "sign", "table", "definition", "priority", "scenario", "source_image"}
        slots: List[Dict[str, Any]] = []
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or fallback_query).strip()
            if not query:
                continue
            facet = ascii_lower(str(item.get("facet") or "general")).replace(" ", "_")
            if facet not in allowed_facets:
                facet = "general"
            slots.append({
                "facet": facet,
                "query": query[:700],
                "priority": self._safe_int(item.get("priority"), idx + 1),
                "reason": str(item.get("reason") or "Nhánh truy vấn").strip()[:300],
                "must_answer": bool(item.get("must_answer", True)),
            })
        return sorted(slots, key=lambda item: int(item.get("priority") or 1))[:10]

    def _compound_penalty_slots(self, q: str, qa: str, *, start_priority: int) -> List[Dict[str, Any]]:
        if not any(marker in qa for marker in [" dong thoi ", " cung luc ", " vua ", " va "]):
            return []
        parts = [
            part.strip(" ,.;")
            for part in re.split(r"\s+(?:và|đồng thời|cùng lúc|kèm theo|rồi)\s+", q, flags=re.IGNORECASE)
            if part.strip(" ,.;")
        ]
        parts = [part for part in parts if self._looks_like_penalty_behavior(part)]
        if not (2 <= len(parts) <= 4):
            return []
        out = []
        for idx, part in enumerate(parts, start=start_priority):
            if len(part.split()) < 3:
                continue
            out.append({
                "facet": "penalty",
                "query": f"Tra cứu mức phạt, trừ điểm và biện pháp bổ sung cho riêng hành vi: {part}. Ngữ cảnh câu hỏi gốc: {q}",
                "priority": idx,
                "reason": "Tách riêng hành vi để tránh bỏ sót mức xử phạt",
                "must_answer": True,
            })
        return out[:4]

    def _looks_like_penalty_behavior(self, text: str) -> bool:
        qa = ascii_lower(text)
        phrase_terms = [
            "di vao",
            "khong doi",
            "khong chap hanh",
            "doi mu",
            "nong do",
            "su dung",
            "nhuong duong",
        ]
        if any(term in qa for term in phrase_terms):
            return True
        return bool(re.search(r"\b(?:di|vuot|dung|do|re|quay|chay|uong)\b", qa))

    def _facets_from_slots(self, slots: List[Dict[str, Any]]) -> List[str]:
        facets = [str(slot.get("facet") or "general") for slot in slots]
        return list(dict.fromkeys(facets)) or ["general"]

    def _intent_from_facets(self, facets: List[str]) -> str:
        for facet in ["penalty", "priority", "scenario", "procedure", "sign", "table", "definition"]:
            if facet in facets:
                return facet
        return "general"

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(1, parsed)
