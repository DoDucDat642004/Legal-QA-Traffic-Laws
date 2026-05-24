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
            '  "intent": "general|penalty|procedure|sign|table|definition|priority|scenario",\n'
            '  "confidence": 0.0,\n'
            '  "difficulty_hint": "easy|medium|hard",\n'
            '  "expected_modalities": ["text"],\n'
            '  "sign_codes": ["P.102"],\n'
            '  "subquestions": [\n'
            '    {"facet": "sign|rule|penalty|procedure|table|definition|priority|scenario|source_image|general", '
            '"query": "câu hỏi con đầy đủ ngữ cảnh", "priority": 1, '
            '"reason": "vì sao cần truy vấn nhánh này", "must_answer": true}\n'
            "  ]\n"
            "}\n"
            "Quy tắc zero-shot:\n"
            "- Luôn suy luận nội bộ trước, nhưng KHÔNG xuất chain-of-thought; chỉ xuất JSON.\n"
            "- Câu dễ vẫn phải có ít nhất 1 nhánh truy vấn đủ ngữ cảnh.\n"
            "- Nếu hỏi vừa ý nghĩa/quy tắc vừa mức phạt thì tách thành các nhánh riêng.\n"
            "- Nếu hỏi biển báo, ưu tiên nhánh sign trước rule/penalty; nếu cần ảnh/căn cứ gốc thêm source_image.\n"
            "- Nếu hỏi bảng/kích thước/tốc độ/hạng GPLX thì dùng table và giữ mã bảng/phụ lục.\n"
            "- Nếu hỏi định nghĩa/khái niệm thì dùng definition.\n"
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

        fallback_facets = [slot.get("facet") for slot in fallback.subquestions]
        ai_facets = {slot.get("facet") for slot in subquestions}
        for slot in fallback.subquestions:
            facet = slot.get("facet")
            if facet and facet != "general" and facet not in ai_facets:
                subquestions.append(dict(slot))

        subquestions = sorted(subquestions, key=lambda item: int(item.get("priority") or 1))[:10]
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
        behavior_hits = self._known_behavior_hits(qa)
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
        if any(key in qa for key in ["bang", "kich thuoc", "toc do toi da", "hang giay phep", "hang gplx"]):
            facets.append("table")
        if any(key in qa for key in ["anh", "hinh anh", "trang goc", "van ban goc", "can cu goc", "file goc", "scan", "crop"]):
            facets.append("source_image")
        if any(key in qa for key in ["xe uu tien", "quyen uu tien", "uu tien", "nhuong duong", "cuu thuong", "chua chay", "cong an", "quan su", "doan xe"]):
            facets.append("priority")
        if behavior_hits or any(key in qa for key in ["tinh huong", "truong hop", "neu", "khi", "dang", "tham gia giao thong", "sau do", "dong thoi", "cung luc", "nga tu", "giao nhau", "vong xuyen"]):
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
        ) or behavior_hits:
            facets.append("penalty")
        if any(key in qa for key in ["thu tuc", "ho so", "cap doi", "cap lai", "sat hach", "dang ky", "thoi han"]):
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
            "general": (
                "Tra cứu căn cứ pháp luật trực tiếp để trả lời câu hỏi: {query}",
                "Truy vấn tổng hợp.",
            ),
        }
        priority_order = ["scenario", "sign", "definition", "priority", "table", "rule", "procedure", "penalty", "source_image", "general"]
        ordered_facets = [facet for facet in priority_order if facet in facets]
        slots = []
        priority = 1
        for facet in ordered_facets:
            if facet == "penalty":
                compound_slots = self._compound_penalty_subquestions(query, start_priority=priority)
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

    def _known_behavior_hits(self, qa: str) -> List[str]:
        hits = []
        for code, terms, _description, _facet in self._known_behavior_patterns():
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
                ["chua du tuoi", "khong du tuoi", "chua du 18", "duoi 18", "phan khoi lon", "gplx hang a", "giay phep lai xe hang a"],
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
                ["nguoc chieu", "duong nguoc chieu", "duong cam", "cam di nguoc chieu"],
                "đi vào đường ngược chiều, đường một chiều hoặc đường cấm",
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
        for facet in ["penalty", "priority", "scenario", "procedure", "sign", "table", "definition"]:
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
        }

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
