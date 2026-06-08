import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from src.rag.legal_utils import (
    SIGN_CODE_RE,
    ascii_lower,
    extract_explicit_legal_refs,
    looks_like_statutory_fine_cap_query,
)

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
        has_vehicle_scope = any(term in qa for term in vehicle_terms)
        penalty_like = "penalty" in facets or any(term in qa for term in ["phat", "xu phat", "muc phat", "vi pham", "bi gi", "xu ly"])
        if (
            penalty_like
            and "aggregation" not in facets
            and not has_vehicle_scope
            and any(term in qa for term in ["xe", "phuong tien", "chay", "toc do", "bien", "p127", "p.127", "tat ca", "toan bo"])
        ):
            score += 4
            reasons.append("Câu hỏi xử phạt chưa nêu loại phương tiện nên phải bao phủ nhiều nhóm xe")

        if any(term in qa for term in ["toc do", "qua toc", "p127", "p.127"]) and not re.search(r"\d+(?:[.,]\d+)?\s*km/?h", qa):
            score += 2
            reasons.append("Câu hỏi tốc độ chưa nêu ngưỡng km/h cụ thể")

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
        if "aggregation" in facets:
            score += 3
            reasons.append("Cần thống kê/tổng hợp trên nhiều bản ghi thay vì trả lời một đoạn luật")
            if self._looks_like_sanction_catalog(qa):
                score += 2
                reasons.append("Cần quét danh mục chế tài trên toàn bộ bản ghi phù hợp")
        if "out_of_scope" in facets:
            score = max(score, 1)
            reasons.append("Câu hỏi nằm ngoài phạm vi luật giao thông đường bộ")
        if "priority" in facets:
            score += 2
            reasons.append("Có yếu tố quyền ưu tiên/nhường đường")
        if "source_image" in facets:
            score += 1
            reasons.append("Cần truy xuất ảnh căn cứ từ văn bản gốc")
        if "document_overview" in facets:
            score += 2
            reasons.append("Cần thống kê cấu trúc văn bản thay vì tìm kiếm ngữ nghĩa thông thường")
        if "legal_detail" in facets:
            score += 4
            reasons.append("Cần gom đầy đủ toàn bộ cấu trúc điều/khoản/điểm")

        if self._is_simple_vehicle_penalty_query(q, qa, facets, slot_count):
            score = min(score, 2)
            reasons = [
                reason for reason in reasons
                if reason not in {
                    "Câu hỏi có nhiều nhánh pháp lý",
                    "Cần truy vấn tuần tự nhiều câu hỏi con",
                }
            ]
            reasons.append("Câu hỏi xử phạt một hành vi, đã rõ nhóm phương tiện")

        if looks_like_statutory_fine_cap_query(qa):
            score = min(score, 2)
            reasons = ["Câu hỏi tra cứu trực tiếp trần phạt tiền được văn bản quy định"]

        difficulty_hint = str(getattr(plan, "difficulty_hint", "") or "").lower()
        if difficulty_hint == "hard":
            score = max(score, 6)
            reasons.append("AI planner đánh giá là câu hỏi khó")
        elif difficulty_hint == "medium":
            score = max(score, 3)
            reasons.append("AI planner đánh giá là câu hỏi trung bình")

        return min(score, 12), list(dict.fromkeys(reasons))

    def _is_simple_vehicle_penalty_query(self, q: str, qa: str, facets: List[str], slot_count: int) -> bool:
        meaningful = {facet for facet in facets if facet != "general"}
        if not meaningful or not meaningful.issubset({"rule", "penalty"}):
            return False
        if not self._vehicle_scope(q):
            return False
        if slot_count > 2:
            return False
        behavior_count = len([
            code for code, terms, _description, facet in self._known_behavior_patterns()
            if facet == "penalty" and any(term in qa for term in terms)
        ])
        return behavior_count <= 1

    def _determine_budget(self, score: int) -> Tuple[str, str, int, Dict[str, int]]:
        """Maps complexity score to retrieval resource allocation."""
        profile = os.getenv("RAG_PROFILE", "balanced").strip().lower()
        if profile in {"fast", "speed", "lite"}:
            if score >= 5:
                return "hard", "Khó", 75, {"top_k": 28, "expand_depth": 2, "evidence_slot_top_k": 8, "max_contexts": 34, "max_images": 12}
            if score >= 3:
                return "medium", "Trung bình", 45, {"top_k": 18, "expand_depth": 1, "evidence_slot_top_k": 6, "max_contexts": 22, "max_images": 8}
            return "easy", "Dễ", 25, {"top_k": 12, "expand_depth": 1, "evidence_slot_top_k": 4, "max_contexts": 12, "max_images": 5}
        if profile in {"deep", "accurate", "accuracy"}:
            if score >= 5:
                return "hard", "Khó", 90, {"top_k": 64, "expand_depth": 5, "evidence_slot_top_k": 14, "max_contexts": 90, "max_images": 40}
            if score >= 3:
                return "medium", "Trung bình", 55, {"top_k": 24, "expand_depth": 3, "evidence_slot_top_k": 7, "max_contexts": 28, "max_images": 18}
            return "easy", "Dễ", 30, {"top_k": 14, "expand_depth": 2, "evidence_slot_top_k": 5, "max_contexts": 16, "max_images": 10}

        if score >= 5:
            return "hard", "Khó", 105, {"top_k": 44, "expand_depth": 3, "evidence_slot_top_k": 11, "max_contexts": 60, "max_images": 24}
        if score >= 3:
            return "medium", "Trung bình", 60, {"top_k": 24, "expand_depth": 2, "evidence_slot_top_k": 7, "max_contexts": 30, "max_images": 12}
        return "easy", "Dễ", 30, {"top_k": 14, "expand_depth": 1, "evidence_slot_top_k": 5, "max_contexts": 16, "max_images": 6}

    def _decompose(self, q: str, qa: str, plan: Any) -> List[Dict[str, Any]]:
        """Builds evidence slots from the planner, with rule fallback."""
        if looks_like_statutory_fine_cap_query(qa):
            return [{
                "facet": "definition",
                "query": q,
                "priority": 1,
                "reason": "Tra cứu trực tiếp điều khoản quy định trần phạt tiền.",
                "must_answer": True,
            }]
        plan_slots = self._sanitize_slots(getattr(plan, "subquestions", None), fallback_query=q)
        rule_slots = self._rule_decompose(q, qa)
        if plan_slots:
            return self._merge_slots(plan_slots, rule_slots, q=q, qa=qa)
        return rule_slots

    def _merge_slots(
        self,
        primary: List[Dict[str, Any]],
        secondary: List[Dict[str, Any]],
        *,
        q: str,
        qa: str,
    ) -> List[Dict[str, Any]]:
        """Keeps planner slots but adds high-recall rule slots for missed facets."""
        merged: List[Dict[str, Any]] = []
        for slot in [*primary, *secondary]:
            facet = str(slot.get("facet") or "general")
            if facet == "general" and any(str(existing.get("facet") or "") != "general" for existing in merged):
                continue
            if self._has_similar_slot(slot, merged):
                continue
            item = dict(slot)
            item["priority"] = self._safe_int(item.get("priority"), len(merged) + 1)
            merged.append(item)

        merged.sort(key=lambda item: int(item.get("priority") or 1))
        for idx, slot in enumerate(merged, start=1):
            slot["priority"] = idx
        return self._compact_simple_slots(merged, q=q, qa=qa)[:16]

    def _compact_simple_slots(self, slots: List[Dict[str, Any]], *, q: str, qa: str) -> List[Dict[str, Any]]:
        facets = {str(slot.get("facet") or "general") for slot in slots}
        complex_facets = {"scenario", "aggregation", "legal_detail", "document_overview", "table", "sign", "source_image", "priority", "procedure", "definition"}
        if facets & complex_facets:
            return slots
        if not self._vehicle_scope(q):
            return slots
        behavior_count = len([
            code for code, terms, _description, facet in self._known_behavior_patterns()
            if facet == "penalty" and any(term in qa for term in terms)
        ])
        if behavior_count > 1:
            return slots

        compacted: List[Dict[str, Any]] = []
        kept_rule = False
        kept_penalty = False
        for slot in slots:
            facet = str(slot.get("facet") or "general")
            if facet == "rule":
                if kept_rule:
                    continue
                kept_rule = True
            elif facet == "penalty":
                if kept_penalty:
                    continue
                kept_penalty = True
            compacted.append(slot)
        for idx, slot in enumerate(compacted, start=1):
            slot["priority"] = idx
        return compacted

    def _has_similar_slot(self, candidate: Dict[str, Any], slots: List[Dict[str, Any]]) -> bool:
        candidate_facet = str(candidate.get("facet") or "general")
        candidate_query = str(candidate.get("query") or "")
        candidate_terms = set(re.findall(r"[a-z0-9đ]+", ascii_lower(candidate_query)))
        candidate_vehicle = self._slot_vehicle_marker(candidate_query)
        for slot in slots:
            if str(slot.get("facet") or "general") != candidate_facet:
                continue
            slot_query = str(slot.get("query") or "")
            slot_vehicle = self._slot_vehicle_marker(slot_query)
            if (candidate_vehicle or slot_vehicle) and candidate_vehicle != slot_vehicle:
                continue
            slot_terms = set(re.findall(r"[a-z0-9đ]+", ascii_lower(slot_query)))
            if not candidate_terms or not slot_terms:
                continue
            overlap = len(candidate_terms & slot_terms) / max(1, len(candidate_terms | slot_terms))
            if overlap >= 0.62:
                return True
        return False

    def _slot_vehicle_marker(self, query: str) -> str:
        qa = ascii_lower(query)
        if "may chuyen dung" in qa:
            return "specialized"
        if any(term in qa for term in ["mo to", "xe mo to", "xe may", "gan may"]):
            return "motorbike"
        if any(term in qa for term in ["o to", "xe hoi", "xe con", "xe tai", "xe khach", "container"]):
            return "car"
        if any(term in qa for term in ["xe dap", "tho so"]):
            return "bicycle"
        return ""

    def _rule_decompose(self, q: str, qa: str) -> List[Dict[str, Any]]:
        """Rules-based decomposition into evidence slots."""
        slots: List[Dict[str, Any]] = []
        statutory_fine_cap = looks_like_statutory_fine_cap_query(qa)
        aggregation_like = self._looks_like_aggregation(qa)
        has_specific_behavior = any(
            term in qa
            for _code, terms, _description, _facet in self._known_behavior_patterns()
            for term in terms
        )
        has_penalty_behavior = any(
            term in qa
            for _code, terms, _description, facet in self._known_behavior_patterns()
            if facet == "penalty"
            for term in terms
        )

        if not SIGN_CODE_RE.search(q or "") and self._looks_like_out_of_scope(qa):
            return [{
                "facet": "out_of_scope",
                "query": f"Câu hỏi ngoài phạm vi luật giao thông đường bộ: {q}",
                "priority": 1,
                "reason": "Cần từ chối đúng phạm vi thay vì truy xuất nhầm nguồn.",
                "must_answer": True,
            }]

        if self._looks_like_document_overview(qa):
            slots.append({
                "facet": "document_overview",
                "query": f"Tra cứu cấu trúc văn bản, số điều, danh sách điều/chương và tiêu đề điều trong câu hỏi: {q}",
                "priority": 1,
                "reason": "Cần trả lời bằng thống kê cấu trúc văn bản, không chỉ tìm đoạn ngữ nghĩa.",
                "must_answer": True,
            })

        if aggregation_like:
            slots.append({
                "facet": "aggregation",
                "query": f"Thống kê/tổng hợp dữ liệu pháp lý liên quan đến câu hỏi: {q}",
                "priority": 1,
                "reason": "Cần tính toán trên nhiều bản ghi có cấu trúc, không ước lượng bằng ngôn ngữ.",
                "must_answer": True,
            })

        if self._looks_like_legal_detail(qa):
            slots.append({
                "facet": "legal_detail",
                "query": f"Gom đầy đủ nội dung điều/khoản/điểm được hỏi, bao gồm tiêu đề, khoản, điểm, mức tiền, ngưỡng và hình thức bổ sung nếu có trong câu hỏi: {q}",
                "priority": 1,
                "reason": "Cần lấy toàn bộ cấu trúc điều luật thay vì vài đoạn rời.",
                "must_answer": True,
            })

        if SIGN_CODE_RE.search(q or "") or ("bien" in qa and any(k in qa for k in ["hinh", "mau", "nhu the nao", "y nghia", "qcvn"])):
            slots.append({
                "facet": "sign",
                "query": f"Xác định biển báo/vạch kẻ, hình dạng, ý nghĩa và căn cứ áp dụng trong câu hỏi: {q}",
                "priority": 1,
                "reason": "Cần hiểu đúng biển báo/vạch kẻ trước khi kết luận."
            })

        if self._looks_like_table_query(qa):
            slots.append({
                "facet": "table",
                "query": f"Tra cứu bảng/phụ lục/thông số kỹ thuật liên quan đến câu hỏi: {q}",
                "priority": 2,
                "reason": "Yêu cầu dữ liệu dạng bảng hoặc thông số"
            })

        if self._looks_like_source_image(qa):
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

        if any(
            k in qa
            for k in [
                "co duoc",
                "duoc phep",
                "quy dinh",
                "lan duong",
                "sai lan",
                "khong dung lan",
                "phan duong",
                "sai phan duong",
                "khong dung phan duong",
                "lan xe cam",
                "lan cam",
                "cam xe may",
                "long le duong",
                "den do",
                "nguoc chieu",
            ]
        ):
            slots.append({
                "facet": "rule",
                "query": f"Tra cứu quy tắc pháp lý và điều kiện áp dụng cho câu hỏi: {q}",
                "priority": 3,
                "reason": "Yêu cầu xác định quy định nền"
            })

        if (
            statutory_fine_cap
            or any(k in qa for k in ["la gi", "khai niem", "dinh nghia", "nghia la gi"])
            or self._looks_like_license_points_fact(qa)
        ):
            slots.append({
                "facet": "definition",
                "query": f"Tra cứu định nghĩa, quy định nền và căn cứ trực tiếp liên quan đến câu hỏi: {q}",
                "priority": 3,
                "reason": "Yêu cầu một quy định/khái niệm pháp lý tổng quát.",
                "must_answer": True,
            })

        penalty_like = has_penalty_behavior or any(
            k in qa
            for k in ["phat", "xu phat", "muc phat", "bao nhieu tien", "bi gi", "xu ly", "vi pham", "tru diem", "tuoc"]
        )
        if penalty_like and not statutory_fine_cap and not (aggregation_like and not has_specific_behavior):
            behavior_slots = self._known_behavior_slots(q, qa, start_priority=4)
            if not behavior_slots:
                behavior_slots = self._compound_penalty_slots(q, qa, start_priority=4)
            penalty_slots = behavior_slots or [{
                "facet": "penalty",
                "query": f"Tra cứu mức phạt tiền, trừ điểm, tước giấy phép và biện pháp bổ sung cho hành vi trong câu hỏi: {q}",
                "priority": 4,
                "reason": "Yêu cầu tra cứu mức xử phạt",
                "must_answer": True,
            }]
            vehicle_slots = self._ambiguous_vehicle_penalty_slots(
                q,
                qa,
                start_priority=4 + len(penalty_slots),
            )
            slots.extend([*penalty_slots, *vehicle_slots])

        if self._looks_like_procedure_query(qa):
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
        allowed_facets = {
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
        return sorted(slots, key=lambda item: int(item.get("priority") or 1))[:12]

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

    def _known_behavior_slots(self, q: str, qa: str, *, start_priority: int) -> List[Dict[str, Any]]:
        slots: List[Dict[str, Any]] = []
        for code, terms, description, facet in self._known_behavior_patterns():
            if not any(term in qa for term in terms):
                continue
            if facet == "rule":
                subquery = (
                    "Tra cứu điều kiện pháp lý và căn cứ áp dụng cho riêng vấn đề: "
                    f"{description}. Ngữ cảnh câu hỏi gốc: {q}"
                )
                reason = "Tách riêng điều kiện nền để không trộn với phần xử phạt."
            elif code == "accident":
                subquery = (
                    "Tra cứu trách nhiệm, nghĩa vụ tại hiện trường, mức xử phạt tăng nặng và hậu quả pháp lý "
                    f"cho riêng vấn đề: {description}. Ngữ cảnh câu hỏi gốc: {q}"
                )
                reason = "Tách riêng hậu quả tai nạn vì có thể kéo theo nhiều nhánh trách nhiệm."
            else:
                subquery = (
                    "Tra cứu mức phạt tiền, trừ điểm giấy phép lái xe, tước giấy phép và biện pháp bổ sung "
                    f"theo Nghị định 168/2024/NĐ-CP cho riêng hành vi: {description}. "
                    f"Ngữ cảnh câu hỏi gốc: {q}"
                )
                reason = "Tách riêng từng hành vi để không bỏ sót hoặc cộng sai mức xử phạt."
            slots.append({
                "facet": facet,
                "query": subquery,
                "priority": start_priority + len(slots),
                "reason": reason,
                "must_answer": True,
            })
        return slots[:8]

    def _known_behavior_patterns(self) -> List[Tuple[str, List[str], str, str]]:
        return [
            (
                "underage_license",
                ["chua du tuoi", "khong du tuoi", "chua du 18", "chua du 18 tuoi", "duoi 18", "17 tuoi", "nguoi 17", "gplx hang a", "giay phep lai xe hang a"],
                "độ tuổi, điều kiện giấy phép lái xe và xe mô tô/xe gắn máy phân khối lớn",
                "rule",
            ),
            (
                "red_light",
                ["vuot den do", "khong chap hanh tin hieu den", "den tin hieu giao thong", "den do"],
                "vượt đèn đỏ hoặc không chấp hành tín hiệu đèn giao thông",
                "penalty",
            ),
            (
                "alcohol",
                ["say xin", "xay xin", "hoi con", "nong do con", "ruou bia", "uong ruou", "co con cao"],
                "điều khiển xe khi trong máu hoặc hơi thở có nồng độ cồn",
                "penalty",
            ),
            (
                "helmet",
                ["khong doi mu", "mu bao hiem"],
                "không đội mũ bảo hiểm khi tham gia giao thông bằng xe mô tô/xe máy",
                "penalty",
            ),
            (
                "wrong_way",
                ["nguoc chieu", "duong nguoc chieu", "duong cam", "cam di nguoc chieu", "p102", "p.102"],
                "đi vào đường ngược chiều, đường một chiều hoặc đường cấm",
                "penalty",
            ),
            (
                "wrong_lane_or_prohibited_lane",
                [
                    "di sai lan",
                    "chay sai lan",
                    "sai lan",
                    "khong dung lan",
                    "di khong dung lan",
                    "chay khong dung lan",
                    "di sai phan duong",
                    "chay sai phan duong",
                    "sai phan duong",
                    "khong dung phan duong",
                    "di khong dung phan duong",
                    "chay khong dung phan duong",
                    "lan xe cam",
                    "lan cam",
                    "lan cam xe",
                    "lan xe cam xe may",
                    "lan cam xe may",
                    "cam xe may",
                    "cam mo to",
                    "cam xe gan may",
                    "di vao lan cam",
                    "chay vao lan cam",
                    "long le duong",
                    "giua long le duong",
                    "chay giua long le duong",
                ],
                "đi sai làn đường, sai phần đường hoặc đi vào làn/đường cấm theo loại phương tiện",
                "penalty",
            ),
            (
                "speed",
                ["vi pham toc do", "qua toc do", "qua toc", "chay qua toc", "chay qua toc do", "vuot toc", "gioi han 40", "40km/h", "40 km/h", "p127", "p.127", "toc do toi da"],
                "chạy quá tốc độ quy định hoặc vượt trị số tốc độ tối đa ghi trên biển P.127",
                "penalty",
            ),
            (
                "phone",
                ["dien thoai", "thiet bi am thanh", "dung tay cam va su dung dien thoai"],
                "sử dụng điện thoại hoặc thiết bị âm thanh khi điều khiển phương tiện",
                "penalty",
            ),
            (
                "parking_sidewalk_obstruction",
                [
                    "do xe",
                    "dung xe",
                    "dung do",
                    "do xe tai",
                    "do o via he",
                    "do tren via he",
                    "do xe o via he",
                    "do xe tren via he",
                    "dung o via he",
                    "dung tren via he",
                    "do o he pho",
                    "do tren he pho",
                    "do long duong",
                    "do tren long duong",
                ],
                "dừng, đỗ xe ở vỉa hè, hè phố, lòng đường hoặc vị trí có nguy cơ cản trở giao thông",
                "penalty",
            ),
            (
                "drug",
                ["ma tuy", "chat ma tuy", "chat kich thich"],
                "điều khiển xe khi trong cơ thể có chất ma túy hoặc chất kích thích bị cấm",
                "penalty",
            ),
            (
                "accident",
                ["gay tai nan", "tai nan cho nguoi khac", "tai nan giao thong"],
                "gây tai nạn giao thông cho người khác",
                "penalty",
            ),
        ]

    def _ambiguous_vehicle_penalty_slots(self, q: str, qa: str, *, start_priority: int) -> List[Dict[str, Any]]:
        if self._vehicle_scope(q):
            return []
        if not any(term in qa for term in ["phat", "xu phat", "muc phat", "bi gi", "xu ly", "vi pham", "tru diem", "tuoc"]):
            return []
        if not any(term in qa for term in ["xe", "phuong tien", "chay", "toc do", "bien", "p127", "p.127", "tat ca", "toan bo"]):
            return []

        speed_query = self._looks_like_speed_query(qa)
        slots: List[Dict[str, Any]] = []
        for vehicle in self._vehicle_groups_for_query(q):
            label = self._vehicle_label(vehicle)
            article_hint = self._vehicle_article_hint(vehicle)
            if speed_query:
                query = (
                    f"Tra cứu đầy đủ mọi mức xử phạt chạy quá tốc độ/P.127 cho {label} "
                    f"theo {article_hint} Nghị định 168/2024/NĐ-CP: mốc km/h, điểm/khoản, "
                    "phạt tiền bằng số, trừ điểm, tước GPLX/tạm giữ nếu có. "
                    f"Câu hỏi gốc: {q}"
                )
                reason = "Câu hỏi tốc độ chưa nêu loại phương tiện nên phải tách riêng từng nhóm xe."
            else:
                query = (
                    f"Tra cứu khả năng xử phạt cho {label} theo {article_hint} Nghị định 168/2024/NĐ-CP. "
                    "Nếu câu hỏi gốc mơ hồ, lấy đủ hành vi liên quan cùng mức tiền, trừ điểm, "
                    "tước GPLX/tạm giữ nếu có; không chỉ ghi 'theo quy định'. "
                    f"Câu hỏi gốc: {q}"
                )
                reason = "Câu hỏi xử phạt chưa nêu loại phương tiện nên phải bao phủ từng nhóm xe."
            slots.append({
                "facet": "penalty",
                "query": query,
                "priority": start_priority + len(slots),
                "reason": reason,
                "must_answer": True,
            })
        return slots[:6]

    def _vehicle_scope(self, query: str) -> str:
        qa = ascii_lower(query)
        if "may chuyen dung" in qa:
            return "specialized"
        if any(term in qa for term in ["xe may", "mo to", "gan may"]):
            return "motorbike"
        if any(term in qa for term in ["o to", "xe hoi", "xe con", "xe tai", "xe khach"]):
            return "car"
        if "xe dap" in qa or "tho so" in qa:
            return "bicycle"
        return ""

    def _vehicle_groups_for_query(self, query: str) -> List[str]:
        qa = ascii_lower(query)
        groups = ["car", "motorbike", "specialized"]
        if not self._looks_like_speed_query(qa) or any(term in qa for term in ["tat ca", "toan bo", "phuong tien", "chay xe"]):
            groups.append("bicycle")
        return groups

    def _vehicle_label(self, vehicle: str) -> str:
        return {
            "car": "ô tô, xe chở hàng bốn bánh có gắn động cơ và các loại xe tương tự ô tô",
            "motorbike": "mô tô, xe gắn máy và các loại xe tương tự xe mô tô/xe gắn máy",
            "specialized": "xe máy chuyên dùng",
            "bicycle": "xe đạp, xe đạp máy và xe thô sơ khác",
        }.get(vehicle, vehicle)

    def _vehicle_article_hint(self, vehicle: str) -> str:
        return {
            "car": "Điều 6 và Điều 13",
            "motorbike": "Điều 7 và Điều 14",
            "specialized": "Điều 8 và Điều 15",
            "bicycle": "Điều 9",
        }.get(vehicle, "")

    def _looks_like_speed_query(self, qa: str) -> bool:
        return any(term in qa for term in ["toc do", "qua toc", "vuot toc", "p127", "p.127"])

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

    def _looks_like_table_query(self, qa: str) -> bool:
        if any(term in qa for term in ["phu luc", "bieu mau", "v85", "he so", "kich thuoc", "tong so gio", "tong thoi gian dao tao"]):
            return True
        has_table_word = bool(re.search(r"\bbang\b", qa)) and not any(
            term in qa
            for term in [
                "bang lai",
                "bang a1",
                "bang a2",
                "bang b",
                "bang c",
                "bang d",
                "bang e",
                "tuoc bang",
                "bang gplx",
                "bang giay phep lai xe",
            ]
        )
        if has_table_word:
            return True
        if "toc do toi da" in qa and any(term in qa for term in ["cao toc", "ngoai khu dong dan cu", "trong khu dong dan cu", "bang"]):
            return True
        if any(term in qa for term in ["hang gplx", "hang giay phep", "hang xe"]) and any(
            term in qa for term in ["bang", "phu luc", "tong so gio", "chuong trinh", "so sanh", "danh sach"]
        ):
            return True
        return False

    def _looks_like_procedure_query(self, qa: str) -> bool:
        return any(
            term in qa
            for term in [
                "thu tuc",
                "ho so",
                "cap doi",
                "cap lai",
                "sat hach",
                "thoi han",
                "nang hang",
                "dao tao",
                "hoc vien",
                "du sat hach",
                "lai xe an toan",
                "bao cao",
                "luu tru",
                "cap chung chi",
                "giay phep lai xe qua han",
                "gplx qua han",
            ]
        )

    def _facets_from_slots(self, slots: List[Dict[str, Any]]) -> List[str]:
        facets = [str(slot.get("facet") or "general") for slot in slots]
        return list(dict.fromkeys(facets)) or ["general"]

    def _intent_from_facets(self, facets: List[str]) -> str:
        for facet in ["out_of_scope", "document_overview", "legal_detail", "aggregation", "penalty", "priority", "scenario", "procedure", "sign", "table", "definition"]:
            if facet in facets:
                return facet
        return "general"

    def _looks_like_aggregation(self, qa: str) -> bool:
        if looks_like_statutory_fine_cap_query(qa):
            return False
        if bool(re.search(r"\btop\s*[-_ ]?\s*k\b", qa)) or any(
            term in qa
            for term in ["truy xuat", "he thong chi co top", "chi co top", "top k nho", "top-k nho"]
        ):
            return False
        ranking_terms = [
            "cao nhat",
            "thap nhat",
            "nang nhat",
            "nhe nhat",
            "lon nhat",
            "nho nhat",
            "toi da",
            "toi thieu",
            "top",
            "xep hang",
            "thong ke",
            "tong hop",
            "nhieu nhat",
            "it nhat",
            "hay vi pham",
            "pho bien nhat",
            "thuong gap",
        ]
        target_terms = [
            "muc phat",
            "phat tien",
            "tru diem",
            "tuoc",
            "dieu luat",
            "dieu nao",
            "hanh vi",
            "vi pham",
            "bien bao",
        ]
        ranking_query = any(term in qa for term in ranking_terms) and any(term in qa for term in target_terms)
        catalog_terms = [
            "cac hanh vi",
            "nhung hanh vi",
            "hanh vi nao",
            "danh sach hanh vi",
            "toan bo hanh vi",
            "liet ke hanh vi",
        ]
        sanction_terms = [
            "tuoc",
            "tru diem",
            "tich thu",
            "tam giu",
            "hinh phat bo sung",
            "hinh thuc xu phat bo sung",
        ]
        catalog_query = any(term in qa for term in catalog_terms) and any(term in qa for term in sanction_terms)
        return ranking_query or catalog_query

    def _looks_like_source_image(self, qa: str) -> bool:
        if re.search(r"\banh\b", qa):
            return True
        return any(
            term in qa
            for term in ["hinh anh", "trang goc", "van ban goc", "can cu goc", "file goc", "scan", "crop"]
        )

    def _looks_like_sanction_catalog(self, qa: str) -> bool:
        catalog_terms = ["cac hanh vi", "nhung hanh vi", "hanh vi nao", "danh sach hanh vi", "toan bo hanh vi"]
        sanction_terms = ["tuoc", "tru diem", "tich thu", "tam giu", "hinh phat bo sung"]
        return any(term in qa for term in catalog_terms) and any(term in qa for term in sanction_terms)

    def _looks_like_license_points_fact(self, qa: str) -> bool:
        license_terms = ["giay phep lai xe", "gplx", "diem lai xe"]
        point_terms = ["bao nhieu diem", "so diem", "tong diem", "co may diem"]
        return any(term in qa for term in license_terms) and any(term in qa for term in point_terms)

    def _looks_like_out_of_scope(self, qa: str) -> bool:
        traffic_terms = [
            "giao thong",
            "duong bo",
            "xe",
            "o to",
            "mo to",
            "xe may",
            "bien bao",
            "den do",
            "toc do",
            "gplx",
            "giay phep lai xe",
            "nong do con",
            "tai nan",
            "nghi dinh 168",
            "nghi dinh 336",
            "qcvn",
            "luat trat tu",
            "luat duong bo",
            "thong tu 35",
            "thong tu 51",
        ]
        if any(term in qa for term in traffic_terms):
            return False
        unrelated_terms = [
            "nau an",
            "nau pho",
            "mon an",
            "thoi tiet",
            "gia vang",
            "chung khoan",
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
            "hon nhan",
            "ly hon",
            "dat dai",
            "hop dong lao dong",
        ]
        if any(term in qa for term in unrelated_terms):
            return True
        legalish_terms = ["phat", "xu phat", "vi pham", "quy dinh", "thu tuc", "ho so", "dieu luat"]
        if any(term in qa for term in legalish_terms):
            return False
        if len(qa.split()) <= 3:
            return False
        return not any(term in qa for term in ["luat", "nghi dinh", "thong tu", "quy chuan", "can cu"])

    def _looks_like_document_overview(self, qa: str) -> bool:
        has_document = any(
            term in qa
            for term in [
                "nghi dinh",
                "nd ",
                "luat",
                "thong tu",
                "qcvn",
                "168/2024",
                "168-2024",
                "336/2025",
                "336-2025",
                "35/2024",
                "36/2024",
                "51/2024",
            ]
        )
        has_overview = any(
            term in qa
            for term in [
                "bao nhieu dieu",
                "may dieu",
                "so dieu",
                "tong so dieu",
                "danh sach dieu",
                "co nhung dieu",
                "gom nhung dieu",
                "bao nhieu chuong",
                "may chuong",
                "cau truc van ban",
            ]
        )
        return has_document and has_overview

    def _looks_like_legal_detail(self, qa: str) -> bool:
        if not re.search(r"\bdieu\s+\d+[a-z]?\b", qa):
            return False
        has_document = any(
            term in qa
            for term in [
                "nghi dinh",
                "nd ",
                "luat",
                "thong tu",
                "qcvn",
                "168/2024",
                "168-2024",
                "336/2025",
                "336-2025",
                "35/2024",
                "36/2024",
                "51/2024",
            ]
        )
        has_detail_word = any(
            term in qa
            for term in [
                "chi tiet",
                "toan van",
                "day du",
                "noi dung",
                "quy dinh gi",
                "noi gi",
                "phan tich dieu",
                "tom tat dieu",
            ]
        )
        return has_document and (has_detail_word or len(qa.split()) <= 12)

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(1, parsed)
