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
        if penalty_like and not has_vehicle_scope and any(term in qa for term in ["xe", "phuong tien", "chay", "toc do", "bien", "p127", "p.127", "tat ca", "toan bo"]):
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
            return "hard", "Khó", 90, {"top_k": 64, "expand_depth": 5, "evidence_slot_top_k": 14, "max_contexts": 90, "max_images": 40}
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

        if self._looks_like_aggregation(qa):
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

        if "bien" in qa and any(k in qa for k in ["hinh", "mau", "nhu the nao", "y nghia", "qcvn"]):
            slots.append({
                "facet": "sign",
                "query": f"Xác định biển báo/vạch kẻ, hình dạng, ý nghĩa và căn cứ áp dụng trong câu hỏi: {q}",
                "priority": 1,
                "reason": "Yêu cầu mô tả hình ảnh biển báo"
            })

        if self._looks_like_table_query(qa):
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
        if any(term in qa for term in ["bang", "phu luc", "bieu mau", "v85", "he so", "kich thuoc", "tong so gio", "tong thoi gian dao tao"]):
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
        return any(term in qa for term in ranking_terms) and any(term in qa for term in target_terms)

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
