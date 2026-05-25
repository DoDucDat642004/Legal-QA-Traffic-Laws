"""
Legal Query Planner for Multi-Document RAG.

This module determines the intent of a user query and generates optimized 
search variations to cover different legal terminologies and document types.
"""

import logging
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.rag.legal_utils import SIGN_CODE_RE, ascii_lower, normalize_sign_code
from src.rag.model_policy import generate_content_with_fallback

# --- Logging Configuration ---
logger = logging.getLogger("LegalQueryPlanner")


class QueryIntent(Enum):
    """Enumerates possible legal question intents."""
    GENERAL = "general"
    PENALTY = "penalty"
    PROCEDURE = "procedure"
    SIGN = "sign"
    TABLE = "table"
    DEFINITION = "definition"
    PRIORITY = "priority"
    SCENARIO = "scenario"
    DOCUMENT_OVERVIEW = "document_overview"
    LEGAL_DETAIL = "legal_detail"
    AGGREGATION = "aggregation"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class QueryPlan:
    """
    Holds the plan for executing a retrieval task.
    """
    intent: QueryIntent = QueryIntent.GENERAL
    confidence: float = 0.5
    sign_codes: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    expected_modalities: List[str] = field(default_factory=list)
    subquestions: List[Dict[str, Any]] = field(default_factory=list)
    plan_source: str = "rule"
    difficulty_hint: Optional[str] = None
    analysis_notes: List[str] = field(default_factory=list)

    def search_queries(self) -> List[str]:
        """Returns specific queries generated for this plan."""
        queries = self.filters.get("search_queries", [])
        if queries:
            return queries
        return [slot.get("query", "") for slot in self.subquestions if slot.get("query")]

    def public_summary(self) -> Dict[str, Any]:
        """Returns client-safe plan diagnostics."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "sign_codes": self.sign_codes,
            "expected_modalities": self.expected_modalities,
            "subquestions": self.subquestions,
            "plan_source": self.plan_source,
            "difficulty_hint": self.difficulty_hint,
            "analysis_notes": self.analysis_notes,
        }


class LegalQueryPlanner:
    """
    Strategic component that plans the retrieval path based on query intent.
    """

    DEFAULT_DOCUMENTS = [
        "Nghị định 168/2024/NĐ-CP",
        "Luật Đường bộ 2024",
        "Luật Trật tự ATGT 2024",
        "Luật Trật tự ATGT 2024 (Tiếp)",
        "QCVN 41:2024 (Thông tư 51/2024)",
        "Thông tư 35/2024/TT-BGTVT",
        "Nghị định 336/2025/NĐ-CP",
    ]

    def __init__(self, documents: Optional[List[str]] = None):
        self.documents = documents or self.DEFAULT_DOCUMENTS

    def plan(self, query: str, client: Any = None) -> QueryPlan:
        """Determines intent and generates a structured plan."""
        fallback = self.rule_plan(query)
        enabled = os.getenv("RAG_ENABLE_AI_PLANNER", "true").lower() in {"1", "true", "yes", "on"}
        if not client or not enabled:
            return fallback
        return self.ai_plan(query, client, fallback=fallback) or fallback

    def rule_plan(self, query: str) -> QueryPlan:
        """Applies heuristic rules to categorize the query and build a plan."""
        qa = ascii_lower(query)
        sign_codes = [normalize_sign_code(m.group(0)) for m in SIGN_CODE_RE.finditer(query)]
        facets = self._detect_facets(qa, has_sign_code=bool(sign_codes))
        if ("sign" in facets or "table" in facets) and "source_image" not in facets:
            facets.append("source_image")
        intent = self._primary_intent(facets)
        confidence = 0.9 if facets != ["general"] else 0.65
        subquestions = self._rule_subquestions(query, facets)
        expected_modalities = self._modalities_for_facets(facets)
        documents = self._documents_for_facets(facets)

        return QueryPlan(
            intent=intent,
            confidence=confidence,
            sign_codes=sign_codes,
            filters={
                "documents": documents,
                "search_queries": [slot["query"] for slot in subquestions],
            },
            expected_modalities=expected_modalities,
            subquestions=subquestions,
            plan_source="rule",
            analysis_notes=["rule_based_classification"],
        )

    def ai_plan(self, query: str, client: Any, *, fallback: QueryPlan) -> Optional[QueryPlan]:
        """
        Uses an LLM to split a complex legal question into sequential retrieval units.

        The rule plan remains the safety net: AI output is accepted only when it
        returns valid JSON with actionable subquestions.
        """
        prompt = (
            "Bạn là query planner cho hệ thống RAG pháp luật giao thông Việt Nam.\n"
            "Hãy phân tích câu hỏi người dùng thành các câu hỏi con để truy vấn TUẦN TỰ, "
            "không truy vấn song song. Mỗi câu hỏi con phải đủ ngữ cảnh và có facet phù hợp.\n"
            "Chỉ trả về JSON object, không markdown.\n"
            "Schema:\n"
            "{\n"
            '  "intent": "general|penalty|procedure|sign|table|definition|priority|scenario|document_overview|legal_detail|aggregation|out_of_scope",\n'
            '  "confidence": 0.0,\n'
            '  "difficulty_hint": "easy|medium|hard",\n'
            '  "expected_modalities": ["text"],\n'
            '  "sign_codes": ["P.102"],\n'
            '  "subquestions": [\n'
            '    {"facet": "sign|rule|penalty|procedure|table|definition|priority|scenario|source_image|document_overview|legal_detail|aggregation|out_of_scope|general", '
            '"query": "câu hỏi con đầy đủ ngữ cảnh", "priority": 1, '
            '"reason": "vì sao cần truy vấn nhánh này", "must_answer": true}\n'
            "  ]\n"
            "}\n"
            "Quy tắc zero-shot:\n"
            "- Luôn suy luận nội bộ trước, nhưng KHÔNG xuất chain-of-thought; chỉ xuất JSON.\n"
            "- Câu dễ vẫn phải có ít nhất 1 nhánh truy vấn đủ ngữ cảnh.\n"
            "- Nếu hỏi vừa ý nghĩa/quy tắc vừa mức phạt thì tách thành các nhánh riêng.\n"
            "- Nếu hỏi biển báo, ưu tiên nhánh sign trước rule/penalty; nếu cần ảnh/căn cứ gốc thêm source_image.\n"
            "- Nếu hỏi P.127/tốc độ và xử phạt, tách nhánh ý nghĩa biển, quy tắc tốc độ, và mức phạt quá tốc độ theo từng nhóm phương tiện.\n"
            "- Nếu hỏi bảng/phụ lục/kích thước/tốc độ tối đa hoặc dòng dữ liệu GPLX trong bảng thì dùng table và giữ mã bảng/phụ lục.\n"
            "- Nếu hỏi định nghĩa/khái niệm thì dùng definition.\n"
            "- Nếu hỏi văn bản có bao nhiêu điều/chương hoặc danh sách điều thì dùng document_overview.\n"
            "- Nếu hỏi chi tiết/toàn văn/nội dung một Điều/Khoản/Điểm cụ thể thì dùng legal_detail.\n"
            "- Nếu hỏi cao nhất/thấp nhất/top/thống kê mức phạt hoặc điều có nhiều chế tài thì dùng aggregation.\n"
            "- Nếu hỏi 'điều nào hay vi phạm nhất' phải dùng aggregation và ghi rõ chỉ thống kê được theo dữ liệu văn bản, không có dữ liệu vi phạm thực tế nếu nguồn không cung cấp.\n"
            "- Nếu câu hỏi rõ ràng ngoài luật giao thông đường bộ thì dùng out_of_scope, không cố truy xuất văn bản luật giao thông.\n"
            "- Nếu hỏi xử phạt nhưng không nêu loại phương tiện, không được đoán; truy vấn lần lượt ô tô, mô tô/xe gắn máy, xe máy chuyên dùng, và xe thô sơ nếu hành vi có thể liên quan.\n"
            "- Nếu câu hỏi rất mơ hồ kiểu 'chạy xe vi phạm', tạo nhánh bao quát theo nhóm phương tiện và yêu cầu lấy mức tiền/trừ điểm/tước GPLX cụ thể, không chỉ lấy tên điều khoản.\n"
            "- Nếu hỏi xe ưu tiên/quyền ưu tiên/nhường đường/giao nhau thì dùng priority.\n"
            "- Nếu mô tả diễn biến thực tế nhiều bước thì dùng scenario rồi rule/penalty theo từng hành vi.\n"
            "Few-shot:\n"
            "Q: 'Xe cứu thương bật còi ở ngã tư, xe máy không nhường thì sao?'\n"
            "A facets: priority -> rule -> penalty.\n"
            "Q: 'Bảng tốc độ tối đa trên cao tốc là bao nhiêu?'\n"
            "A facets: table -> rule -> source_image.\n"
            "Q: 'Biển P.102 có hình gì và đi vào bị phạt không?'\n"
            "A facets: sign -> rule -> penalty -> source_image.\n"
            f"Câu hỏi: {query}"
        )
        try:
            config = None
            try:
                from google.genai import types

                config = types.GenerateContentConfig(temperature=0.0, max_output_tokens=2048)
            except Exception:
                pass
            response, _model = generate_content_with_fallback(
                client,
                contents=[prompt],
                config=config,
                env_names=("RAG_PLANNER_MODEL", "RAG_AI_PLANNER_MODEL", "RAG_ANSWER_MODEL"),
                logger=logger,
                label="AI query planning",
            )
            data = self._parse_json_object(response.text or "")
            return self._plan_from_ai_payload(data, fallback=fallback)
        except Exception as exc:
            logger.warning("AI query planning failed; falling back to rules: %s", exc)
            return None

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
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

    def _plan_from_ai_payload(self, data: Dict[str, Any], *, fallback: QueryPlan) -> Optional[QueryPlan]:
        if not data:
            return None
        subquestions = self._sanitize_subquestions(data.get("subquestions"))
        if not subquestions:
            return None

        for slot in fallback.subquestions:
            facet = slot.get("facet")
            if facet and facet != "general" and not self._has_similar_slot(slot, subquestions):
                subquestions.append(dict(slot))

        subquestions = sorted(subquestions, key=lambda item: int(item.get("priority") or 1))[:12]
        intent = self._intent_from_value(data.get("intent")) or fallback.intent
        confidence = self._bounded_float(data.get("confidence"), default=fallback.confidence)
        difficulty_hint = str(data.get("difficulty_hint") or "").lower()
        if difficulty_hint not in {"easy", "medium", "hard"}:
            difficulty_hint = fallback.difficulty_hint or None

        sign_codes = [
            normalize_sign_code(str(code))
            for code in (data.get("sign_codes") or [])
            if str(code or "").strip()
        ]
        sign_codes = list(dict.fromkeys([*fallback.sign_codes, *sign_codes]))
        modalities = self._sanitize_modalities(data.get("expected_modalities")) or fallback.expected_modalities
        documents = self._documents_for_facets([slot.get("facet", "general") for slot in subquestions])

        return QueryPlan(
            intent=intent,
            confidence=max(confidence, fallback.confidence),
            sign_codes=sign_codes,
            filters={
                "documents": documents,
                "search_queries": [slot["query"] for slot in subquestions],
            },
            expected_modalities=modalities,
            subquestions=subquestions,
            plan_source="ai",
            difficulty_hint=difficulty_hint,
            analysis_notes=["ai_decomposition", *fallback.analysis_notes],
        )

    def _has_similar_slot(self, candidate: Dict[str, Any], slots: List[Dict[str, Any]]) -> bool:
        candidate_facet = candidate.get("facet")
        candidate_terms = set(re.findall(r"\w+", ascii_lower(candidate.get("query") or "")))
        if not candidate_terms:
            return False
        for slot in slots:
            if slot.get("facet") != candidate_facet:
                continue
            slot_terms = set(re.findall(r"\w+", ascii_lower(slot.get("query") or "")))
            if not slot_terms:
                continue
            overlap = len(candidate_terms & slot_terms) / max(1, min(len(candidate_terms), len(slot_terms)))
            if overlap >= 0.72:
                return True
        return False

    def _sanitize_subquestions(self, value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        allowed_facets = self._allowed_facets()
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(value):
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if len(query) < 5:
                continue
            facet = ascii_lower(str(item.get("facet") or "general")).replace(" ", "_")
            if facet not in allowed_facets:
                facet = "general"
            out.append(
                {
                    "facet": facet,
                    "query": query[:700],
                    "priority": self._bounded_int(item.get("priority"), default=idx + 1, min_value=1, max_value=20),
                    "reason": str(item.get("reason") or "Nhánh truy vấn do AI planner tạo.").strip()[:300],
                    "must_answer": bool(item.get("must_answer", True)),
                }
            )
        return out

    def _detect_facets(self, qa: str, *, has_sign_code: bool) -> List[str]:
        facets: List[str] = []
        if not has_sign_code and self._looks_like_out_of_scope(qa):
            return ["out_of_scope"]
        behavior_hits = self._known_behavior_hits(qa)
        penalty_behavior_hits = self._known_behavior_hits(qa, facets={"penalty"})
        if self._looks_like_aggregation(qa):
            facets.append("aggregation")
        if has_sign_code or any(
            key in qa
            for key in [
                "bien bao",
                "bien cam",
                "bien hieu lenh",
                "bien chi dan",
                "vach ke duong",
                "hinh dang",
                "mau sac",
                "qcvn 41",
            ]
        ):
            facets.append("sign")
        if self._looks_like_table_query(qa):
            facets.append("table")
        if self._looks_like_document_overview(qa):
            facets.append("document_overview")
        if self._looks_like_legal_detail(qa):
            facets.append("legal_detail")
        if any(key in qa for key in ["anh", "hinh anh", "trang goc", "van ban goc", "can cu goc", "file goc", "scan", "crop"]):
            facets.append("source_image")
        if any(key in qa for key in ["xe uu tien", "quyen uu tien", "uu tien", "nhuong duong", "cuu thuong", "chua chay", "cong an", "quan su", "doan xe"]):
            facets.append("priority")
        scenario_markers = [
            "tinh huong",
            "truong hop",
            "neu",
            "khi",
            "dang",
            "tham gia giao thong",
            "sau do",
            "dong thoi",
            "cung luc",
            "nga tu",
            "giao nhau",
            "vong xuyen",
        ]
        scenario_like = len(behavior_hits) >= 2 or any(key in qa for key in scenario_markers)
        if scenario_like and not ("sign" in facets and len(behavior_hits) < 2):
            facets.append("scenario")
        if any(
            key in qa
            for key in [
                "phat",
                "xu phat",
                "muc phat",
                "bao nhieu tien",
                "tru diem",
                "tuoc",
                "tam giu",
                "bi gi",
                "xu ly",
                "vi pham",
            ]
        ) or penalty_behavior_hits:
            facets.append("penalty")
        if self._looks_like_procedure_query(qa):
            facets.append("procedure")
        if any(key in qa for key in ["la gi", "khai niem", "dinh nghia", "nghia la gi"]):
            facets.append("definition")
        if any(
            key in qa
            for key in [
                "co duoc",
                "duoc phep",
                "quy dinh",
                "trach nhiem",
                "di vao",
                "lan duong",
                "den do",
                "nguoc chieu",
                "dung xe",
                "do xe",
                "vuot",
                "chua du tuoi",
                "khong du tuoi",
                "giay phep lai xe",
                "gplx",
                "phan khoi lon",
                "nong do con",
                "hoi con",
                "say xin",
                "mu bao hiem",
                "tai nan",
            ]
        ) or behavior_hits:
            facets.append("rule")
        if len(behavior_hits) >= 2 and "source_image" not in facets:
            facets.append("source_image")
        return list(dict.fromkeys(facets)) or ["general"]

    def _rule_subquestions(self, query: str, facets: List[str]) -> List[Dict[str, Any]]:
        templates = {
            "out_of_scope": (
                "Câu hỏi ngoài phạm vi luật giao thông đường bộ: {query}",
                "Cần từ chối đúng phạm vi thay vì truy xuất nhầm nguồn.",
            ),
            "aggregation": (
                "Thống kê/tổng hợp dữ liệu pháp lý liên quan đến câu hỏi gốc, giữ nguyên tiêu chí xếp hạng người dùng nêu: {query}",
                "Cần tính toán từ toàn bộ nguồn phù hợp, không để LLM ước lượng.",
            ),
            "sign": (
                "Xác định biển báo, vạch kẻ hoặc mã biển liên quan; mô tả hình dạng, ý nghĩa và phạm vi áp dụng trong câu hỏi: {query}",
                "Cần hiểu đúng tín hiệu/biển báo trước khi kết luận.",
            ),
            "rule": (
                "Tra cứu quy tắc pháp lý, điều kiện áp dụng và hành vi cần đánh giá trong câu hỏi: {query}",
                "Cần xác định quy định nền trước khi trả lời.",
            ),
            "penalty": (
                "Tra cứu mức phạt tiền, trừ điểm giấy phép lái xe, tước giấy phép và biện pháp bổ sung cho hành vi trong câu hỏi: {query}",
                "Cần căn cứ xử phạt tương ứng với hành vi.",
            ),
            "procedure": (
                "Tra cứu thủ tục, hồ sơ, thời hạn và cơ quan xử lý liên quan đến câu hỏi: {query}",
                "Cần căn cứ thủ tục hành chính.",
            ),
            "table": (
                "Tra cứu bảng, phụ lục, thông số kỹ thuật hoặc dòng dữ liệu liên quan đến câu hỏi: {query}",
                "Cần dữ liệu dạng bảng/phụ lục.",
            ),
            "definition": (
                "Tra cứu định nghĩa, khái niệm và phạm vi hiểu đúng của thuật ngữ trong câu hỏi: {query}",
                "Cần căn cứ định nghĩa.",
            ),
            "priority": (
                "Tra cứu quy định về quyền ưu tiên, xe ưu tiên, thứ tự nhường đường và điều kiện áp dụng trong câu hỏi: {query}",
                "Cần xác định trường hợp ưu tiên trước khi kết luận.",
            ),
            "scenario": (
                "Tra cứu căn cứ xử lý tình huống thực tế nhiều bước, gồm chủ thể, hành vi, thời điểm và điều kiện trong câu hỏi: {query}",
                "Cần bóc tách diễn biến thực tế để đối chiếu từng căn cứ.",
            ),
            "source_image": (
                "Tìm ảnh trang gốc, hình, bảng hoặc phụ lục trong văn bản làm căn cứ trực quan cho câu hỏi: {query}",
                "Người dùng cần căn cứ hình ảnh từ văn bản gốc.",
            ),
            "document_overview": (
                "Tra cứu cấu trúc văn bản, số điều, danh sách điều/chương và tiêu đề điều trong câu hỏi: {query}",
                "Cần trả lời bằng thống kê cấu trúc văn bản, không chỉ tìm đoạn ngữ nghĩa.",
            ),
            "legal_detail": (
                "Gom đầy đủ nội dung điều/khoản/điểm được hỏi, bao gồm tiêu đề, khoản, điểm, mức tiền, ngưỡng và hình thức bổ sung nếu có trong câu hỏi: {query}",
                "Cần lấy toàn bộ cấu trúc điều luật thay vì vài đoạn rời.",
            ),
            "general": (
                "Tra cứu căn cứ pháp luật trực tiếp để trả lời câu hỏi: {query}",
                "Truy vấn tổng hợp.",
            ),
        }
        priority_order = [
            "out_of_scope",
            "document_overview",
            "legal_detail",
            "aggregation",
            "scenario",
            "sign",
            "definition",
            "priority",
            "table",
            "penalty",
            "rule",
            "procedure",
            "source_image",
            "general",
        ]
        ordered_facets = [facet for facet in priority_order if facet in facets]
        slots = []
        priority = 1
        for facet in ordered_facets:
            if facet == "penalty":
                compound_slots = self._compound_penalty_subquestions(query, start_priority=priority)
                compound_slots.extend(self._ambiguous_vehicle_penalty_subquestions(
                    query,
                    start_priority=priority + len(compound_slots),
                ))
                if compound_slots:
                    slots.extend(compound_slots)
                    priority += len(compound_slots)
                    continue
            template, reason = templates[facet]
            slots.append(
                {
                    "facet": facet,
                    "query": template.format(query=query),
                    "priority": priority,
                    "reason": reason,
                    "must_answer": True,
                }
            )
            priority += 1
        return slots

    def _compound_penalty_subquestions(self, query: str, *, start_priority: int) -> List[Dict[str, Any]]:
        known_slots = self._known_behavior_subquestions(query, start_priority=start_priority)
        if known_slots:
            return known_slots

        qa = ascii_lower(query)
        if not any(marker in qa for marker in [" dong thoi ", " cung luc ", " vua ", " va "]):
            return []
        parts = [
            part.strip(" ,.;")
            for part in re.split(r"\s+(?:và|đồng thời|cùng lúc|kèm theo|rồi)\s+", query, flags=re.IGNORECASE)
            if part.strip(" ,.;")
        ]
        parts = [part for part in parts if self._looks_like_penalty_behavior(part)]
        if not (2 <= len(parts) <= 4):
            return []
        slots: List[Dict[str, Any]] = []
        for offset, part in enumerate(parts):
            if len(part.split()) < 3:
                continue
            slots.append(
                {
                    "facet": "penalty",
                    "query": (
                        "Tra cứu mức phạt tiền, trừ điểm giấy phép lái xe, tước giấy phép "
                        f"và biện pháp bổ sung cho riêng hành vi: {part}. "
                        f"Ngữ cảnh câu hỏi gốc: {query}"
                    ),
                    "priority": start_priority + offset,
                    "reason": "Tách riêng từng hành vi để không bỏ sót hoặc cộng sai mức xử phạt.",
                    "must_answer": True,
                }
            )
        return slots[:8]

    def _ambiguous_vehicle_penalty_subquestions(self, query: str, *, start_priority: int) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        if self._vehicle_scope(query):
            return []
        if not any(term in qa for term in ["phat", "xu phat", "muc phat", "bi gi", "xu ly", "vi pham"]):
            return []
        if not any(term in qa for term in ["xe", "phuong tien", "chay", "toc do", "bien", "p127", "p.127", "tat ca", "toan bo"]):
            return []

        speed_query = self._looks_like_speed_query(qa)
        slots: List[Dict[str, Any]] = []
        for vehicle in self._vehicle_groups_for_query(query):
            label = self._vehicle_label(vehicle)
            article_hint = self._vehicle_article_hint(vehicle)
            if speed_query:
                subquery = (
                    f"Tra cứu đầy đủ mọi mức xử phạt chạy quá tốc độ/P.127 cho {label} "
                    f"theo {article_hint} Nghị định 168/2024/NĐ-CP: mốc km/h, điểm/khoản, "
                    "phạt tiền bằng số, trừ điểm, tước GPLX/tạm giữ nếu có. "
                    f"Câu hỏi gốc: {query}"
                )
                reason = "Câu hỏi tốc độ chưa nêu loại phương tiện nên phải tách riêng từng nhóm xe."
            else:
                subquery = (
                    f"Tra cứu khả năng xử phạt cho {label} theo {article_hint} Nghị định 168/2024/NĐ-CP. "
                    "Nếu câu hỏi gốc mơ hồ, liệt kê toàn bộ hành vi liên quan tìm thấy cùng mức tiền, "
                    "trừ điểm, tước GPLX/tạm giữ nếu có. "
                    f"Câu hỏi gốc: {query}"
                )
                reason = "Câu hỏi xử phạt chưa nêu loại phương tiện nên phải bao phủ từng nhóm xe."
            slots.append({
                "facet": "penalty",
                "query": subquery,
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

    def _looks_like_speed_query(self, qa: str) -> bool:
        return any(term in qa for term in ["toc do", "qua toc", "p127", "p.127"])

    def _looks_like_table_query(self, qa: str) -> bool:
        table_terms = [
            "bang",
            "phu luc",
            "bieu mau",
            "v85",
            "he so",
            "kich thuoc",
            "tong so gio",
            "tong thoi gian dao tao",
            "chuong trinh dao tao",
        ]
        if any(term in qa for term in table_terms):
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
            key in qa
            for key in [
                "thu tuc",
                "ho so",
                "cap doi",
                "cap lai",
                "sat hach",
                "dang ky",
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

    def _known_behavior_hits(self, qa: str, *, facets: Optional[set[str]] = None) -> List[str]:
        hits = []
        for code, terms, _description, _facet in self._known_behavior_patterns():
            if facets is not None and _facet not in facets:
                continue
            if any(term in qa for term in terms):
                hits.append(code)
        return hits

    def _known_behavior_subquestions(self, query: str, *, start_priority: int) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        slots: List[Dict[str, Any]] = []
        for code, terms, description, facet in self._known_behavior_patterns():
            if not any(term in qa for term in terms):
                continue
            if facet == "rule":
                subquery = (
                    "Tra cứu điều kiện pháp lý và căn cứ áp dụng cho riêng vấn đề: "
                    f"{description}. Ngữ cảnh câu hỏi gốc: {query}"
                )
                reason = "Tách riêng điều kiện nền để không trộn với phần xử phạt."
            elif code == "accident":
                subquery = (
                    "Tra cứu trách nhiệm, mức xử phạt tăng nặng và nghĩa vụ khi gây tai nạn giao thông "
                    f"cho riêng vấn đề: {description}. Ngữ cảnh câu hỏi gốc: {query}"
                )
                reason = "Tách riêng hậu quả tai nạn vì có thể kéo theo trách nhiệm hành chính, dân sự hoặc hình sự."
            else:
                subquery = (
                    "Tra cứu mức phạt tiền, trừ điểm giấy phép lái xe, tước giấy phép và biện pháp bổ sung "
                    f"theo Nghị định 168/2024/NĐ-CP cho riêng hành vi: {description}. "
                    f"Ngữ cảnh câu hỏi gốc: {query}"
                )
                reason = "Tách riêng từng hành vi để không bỏ sót hoặc cộng sai mức xử phạt."
            slots.append(
                {
                    "facet": facet,
                    "query": subquery,
                    "priority": start_priority + len(slots),
                    "reason": reason,
                    "must_answer": True,
                }
            )
        return slots[:8]

    def _known_behavior_patterns(self) -> List[tuple[str, List[str], str, str]]:
        return [
            (
                "underage_license",
                ["chua du tuoi", "khong du tuoi", "chua du 18", "duoi 18", "17 tuoi", "nguoi 17", "phan khoi lon", "gplx hang a", "giay phep lai xe hang a"],
                "độ tuổi, điều kiện giấy phép lái xe hạng A và xe mô tô/xe gắn máy phân khối lớn",
                "rule",
            ),
            (
                "red_light",
                ["vuot den do", "khong chap hanh tin hieu den", "den tin hieu giao thong"],
                "vượt đèn đỏ hoặc không chấp hành tín hiệu đèn giao thông",
                "penalty",
            ),
            (
                "alcohol",
                ["say xin", "hoi con", "nong do con", "ruou bia", "co con cao"],
                "điều khiển xe khi trong máu hoặc hơi thở có nồng độ cồn cao",
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
                "speed",
                ["vi pham toc do", "qua toc do", "chay qua toc", "vuot toc", "p127", "p.127"],
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
                "drug",
                ["ma tuy", "chat ma tuy", "chat kich thich"],
                "điều khiển xe khi trong cơ thể có chất ma túy hoặc chất kích thích bị cấm",
                "penalty",
            ),
            (
                "three_passengers",
                ["cho theo tu 03", "cho theo 03", "cho 3 nguoi", "cho ba nguoi"],
                "chở theo từ 03 người trở lên trên xe mô tô, xe gắn máy",
                "penalty",
            ),
            (
                "accident",
                ["gay tai nan", "tai nan cho nguoi khac", "tai nan giao thong"],
                "gây tai nạn giao thông cho người khác",
                "penalty",
            ),
        ]

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

    def _primary_intent(self, facets: List[str]) -> QueryIntent:
        for facet in ["out_of_scope", "document_overview", "legal_detail", "aggregation", "penalty", "priority", "scenario", "definition", "procedure", "sign", "table"]:
            if facet in facets:
                return QueryIntent(facet)
        return QueryIntent.GENERAL

    def _modalities_for_facets(self, facets: List[str]) -> List[str]:
        modalities = ["text"]
        if "sign" in facets:
            modalities.extend(["sign", "figure"])
        if "table" in facets:
            modalities.append("table")
        if "source_image" in facets:
            modalities.extend(["image", "figure", "table"])
        if "aggregation" in facets:
            modalities.append("text")
        return list(dict.fromkeys(modalities))

    def _documents_for_facets(self, facets: List[str]) -> List[str]:
        documents = set(self.documents)
        if "penalty" in facets:
            documents.add("Nghị định 168/2024/NĐ-CP")
        if "sign" in facets or "table" in facets:
            documents.add("QCVN 41:2024 (Thông tư 51/2024)")
        if "procedure" in facets:
            documents.add("Thông tư 35/2024/TT-BGTVT")
        if "priority" in facets or "scenario" in facets:
            documents.add("Luật Trật tự ATGT 2024")
            documents.add("Luật Đường bộ 2024")
        if "aggregation" in facets:
            documents.add("Nghị định 168/2024/NĐ-CP")
            documents.add("Nghị định 336/2025/NĐ-CP")
        return [doc for doc in self.documents if doc in documents]

    def _allowed_facets(self) -> set[str]:
        return {
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
        has_article = bool(re.search(r"\bdieu\s+\d+[a-z]?\b", qa))
        if not has_article:
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

    def _intent_from_value(self, value: Any) -> Optional[QueryIntent]:
        raw = ascii_lower(str(value or "")).replace(" ", "_")
        for intent in QueryIntent:
            if intent.value == raw:
                return intent
        return None

    def _sanitize_modalities(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        allowed = {"text", "table", "sign", "figure", "image"}
        out = []
        for item in value:
            modality = ascii_lower(str(item or ""))
            if modality in allowed:
                out.append(modality)
        return list(dict.fromkeys(out))

    def _bounded_float(self, value: Any, *, default: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return default
        return max(0.0, min(1.0, parsed))

    def _bounded_int(self, value: Any, *, default: int, min_value: int, max_value: int) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = default
        return max(min_value, min(max_value, parsed))
