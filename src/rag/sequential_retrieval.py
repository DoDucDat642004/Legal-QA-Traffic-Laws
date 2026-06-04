import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.rag.adaptive_query import AdaptiveQueryProfile
from src.rag.legal_utils import (
    ascii_lower,
    merge_record_assets,
    normalized_legal_reference,
    record_image_paths,
    source_text,
)
from src.rag.query_planner import QueryPlan

# --- Logging Configuration ---
logger = logging.getLogger("sequential_retrieval")


@dataclass
class SequentialEvidenceSlot:
    """Represents a single, targeted unit of retrieval."""
    id: str
    query: str
    facet: str = "general"
    priority: int = 1
    must_answer: bool = True
    reason: str = ""


@dataclass
class SequentialRetrievalResult:
    """Holds the results and metadata for a specific evidence slot."""
    slot: SequentialEvidenceSlot
    records: List[Dict[str, Any]]
    images: List[str] = field(default_factory=list)
    status: str = "pending"
    error: Optional[str] = None


class SequentialRetrievalOrchestrator:
    """
    Orchestrates the sequential retrieval flow for complex multi-faceted queries.
    """

    def __init__(self, retriever: Any, generator_fn: Callable, asset_path_fn: Optional[Callable] = None):
        self.retriever = retriever
        self.generator_fn = generator_fn
        self.asset_path_fn = asset_path_fn or (lambda x: x)

    def orchestrate(
        self, 
        query: str, 
        profile: AdaptiveQueryProfile, 
        plan: QueryPlan,
        top_k_per_slot: int = 6
    ) -> Dict[str, Any]:
        """Decomposes, retrieves, and synthesizes a final answer."""
        slots = self._prepare_slots(profile, query)
        budget = profile.retrieval_budget or {}
        slot_top_k = int(budget.get("evidence_slot_top_k") or top_k_per_slot)
        max_contexts = int(budget.get("max_contexts") or budget.get("top_k") or 24)
        max_rounds = self._env_int("RAG_RETRIEVAL_MAX_ROUNDS", 3, minimum=1, maximum=6)
        max_slots = self._env_int("RAG_RETRIEVAL_MAX_SLOTS", 18, minimum=len(slots), maximum=48)
        answer_repair_rounds = self._env_int("RAG_ANSWER_REPAIR_ROUNDS", 1, minimum=0, maximum=3)
        results: List[SequentialRetrievalResult] = []
        accumulated_records: List[Dict[str, Any]] = []
        seen_keys: set = set()
        records_by_key: Dict[str, Dict[str, Any]] = {}
        processed_slot_keys: set = set()

        for round_idx in range(max_rounds):
            pending_slots = [slot for slot in slots if self._slot_key(slot) not in processed_slot_keys]
            if not pending_slots:
                break

            for slot in pending_slots:
                self._process_slot(
                    slot,
                    plan=plan,
                    profile=profile,
                    slot_top_k=slot_top_k,
                    results=results,
                    accumulated_records=accumulated_records,
                    seen_keys=seen_keys,
                    records_by_key=records_by_key,
                )
                processed_slot_keys.add(self._slot_key(slot))

            followups = self._coverage_followup_slots(
                query=query,
                profile=profile,
                plan=plan,
                results=results,
                records=accumulated_records,
                existing_slots=slots,
                round_idx=round_idx,
            )
            added = self._append_new_slots(slots, followups, max_slots=max_slots)
            if added == 0:
                break

        if not accumulated_records:
            try:
                fallback_records = self.retriever.retrieve(
                    query,
                    top_k=max(6, min(max_contexts, int(budget.get("top_k") or 10))),
                    expand_depth=int(budget.get("expand_depth") or 1),
                )
                for record in self._tag_records(fallback_records, SequentialEvidenceSlot(
                    id="fallback",
                    query=query,
                    facet="general",
                    priority=99,
                    reason="Fallback sau khi các slot không có kết quả",
                )):
                    rid = self._record_key(record)
                    if rid in seen_keys:
                        merge_record_assets(records_by_key[rid], record)
                    else:
                        accumulated_records.append(record)
                        records_by_key[rid] = record
                        seen_keys.add(rid)
            except Exception as exc:
                logger.exception("Sequential fallback retrieval failed")
                results.append(SequentialRetrievalResult(
                    slot=SequentialEvidenceSlot(id="fallback", query=query, reason="Fallback retrieval"),
                    records=[],
                    status="error",
                    error=str(exc),
                ))

        slot_order = {slot.id: idx for idx, slot in enumerate(slots)}
        accumulated_records.sort(
            key=lambda r: (
                slot_order.get(str(r.get("retrieval_slot_id") or ""), 999),
                -float(r.get("retrieval_score") or 0),
            )
        )
        accumulated_records = accumulated_records[:max_contexts]
        final_answer = self.generator_fn(query, accumulated_records, sequential_results=results)
        for repair_idx in range(answer_repair_rounds):
            if not self._answer_has_unresolved_ambiguity(final_answer, query, profile):
                break
            repair_slots = self._answer_repair_slots(
                answer=final_answer,
                query=query,
                profile=profile,
                plan=plan,
                results=results,
                records=accumulated_records,
                existing_slots=slots,
                repair_idx=repair_idx,
            )
            if not self._append_new_slots(slots, repair_slots, max_slots=max_slots):
                break
            for slot in [s for s in slots if self._slot_key(s) not in processed_slot_keys]:
                self._process_slot(
                    slot,
                    plan=plan,
                    profile=profile,
                    slot_top_k=slot_top_k,
                    results=results,
                    accumulated_records=accumulated_records,
                    seen_keys=seen_keys,
                    records_by_key=records_by_key,
                )
                processed_slot_keys.add(self._slot_key(slot))
            slot_order = {slot.id: idx for idx, slot in enumerate(slots)}
            accumulated_records.sort(
                key=lambda r: (
                    slot_order.get(str(r.get("retrieval_slot_id") or ""), 999),
                    -float(r.get("retrieval_score") or 0),
                )
            )
            accumulated_records = accumulated_records[:max_contexts]
            final_answer = self.generator_fn(query, accumulated_records, sequential_results=results)

        return {
            "answer": final_answer,
            "results": results,
            "public_results": [self._result_to_dict(result) for result in results],
            "accumulated_records": accumulated_records,
            "slots": [slot.__dict__ for slot in slots]
        }

    def _prepare_slots(self, profile: AdaptiveQueryProfile, original_query: str) -> List[SequentialEvidenceSlot]:
        """Converts profile evidence slots into orchestrator slots."""
        raw_slots = profile.evidence_slots or []
        if not raw_slots:
            return [SequentialEvidenceSlot(id="slot_0", query=original_query, reason="Fallback")]

        slots = []
        for i, s in enumerate(raw_slots):
            slots.append(SequentialEvidenceSlot(
                id=f"slot_{i}",
                query=str(s.get("query") or original_query).strip(),
                facet=str(s.get("facet") or "general"),
                priority=int(s.get("priority") or 1),
                reason=str(s.get("reason") or "")
            ))
        slots.sort(key=lambda x: x.priority)
        return slots

    def _process_slot(
        self,
        slot: SequentialEvidenceSlot,
        *,
        plan: QueryPlan,
        profile: AdaptiveQueryProfile,
        slot_top_k: int,
        results: List[SequentialRetrievalResult],
        accumulated_records: List[Dict[str, Any]],
        seen_keys: set,
        records_by_key: Dict[str, Dict[str, Any]],
    ) -> None:
        logger.info("Processing slot %s: %s", slot.id, slot.query)
        try:
            slot_records = self._retrieve_for_slot(slot, plan, profile, slot_top_k)
        except Exception as exc:
            logger.exception("Sequential retrieval failed for slot %s", slot.id)
            results.append(SequentialRetrievalResult(
                slot=slot,
                records=[],
                images=[],
                status="error",
                error=str(exc),
            ))
            return

        slot_records = self._tag_records(slot_records, slot)
        slot_images = sorted({
            self.asset_path_fn(path)
            for r in slot_records
            for path in record_image_paths(r)
        })
        results.append(SequentialRetrievalResult(
            slot=slot,
            records=slot_records,
            images=slot_images,
            status="hit" if slot_records else "miss",
        ))

        for record in slot_records:
            rid = self._record_key(record)
            if rid in seen_keys:
                merge_record_assets(records_by_key[rid], record)
            else:
                accumulated_records.append(record)
                records_by_key[rid] = record
                seen_keys.add(rid)

    def _env_int(self, name: str, default: int, *, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = default
        return max(minimum, min(value, maximum))

    def _slot_key(self, slot: SequentialEvidenceSlot) -> str:
        return f"{slot.facet}|{ascii_lower(slot.query)}"

    def _append_new_slots(
        self,
        slots: List[SequentialEvidenceSlot],
        new_slots: List[SequentialEvidenceSlot],
        *,
        max_slots: int,
    ) -> int:
        seen = {self._slot_key(slot) for slot in slots}
        added = 0
        for slot in new_slots:
            if len(slots) >= max_slots:
                break
            key = self._slot_key(slot)
            if key in seen:
                continue
            seen.add(key)
            slots.append(slot)
            added += 1
        if added:
            slots.sort(key=lambda item: (item.priority, item.id))
        return added

    def _coverage_followup_slots(
        self,
        *,
        query: str,
        profile: AdaptiveQueryProfile,
        plan: QueryPlan,
        results: List[SequentialRetrievalResult],
        records: List[Dict[str, Any]],
        existing_slots: List[SequentialEvidenceSlot],
        round_idx: int,
    ) -> List[SequentialEvidenceSlot]:
        del plan
        qa = ascii_lower(query)
        facets = set(profile.facets or [])
        followups: List[SequentialEvidenceSlot] = []
        aggregation_only = "aggregation" in facets and "penalty" not in facets
        is_penalty = not aggregation_only and ("penalty" in facets or self._looks_like_penalty_query(qa))
        needs_vehicle_breakdown = is_penalty and self._needs_vehicle_breakdown(query)
        needs_speed_thresholds = is_penalty and self._looks_like_speed_query(qa)

        vehicles = self._vehicle_groups_for_query(query)
        if needs_speed_thresholds:
            for vehicle in vehicles:
                if not self._vehicle_group_has_records(vehicle, records, speed_only=True):
                    followups.append(self._vehicle_speed_slot(vehicle, query, round_idx, len(followups)))

        if needs_vehicle_breakdown:
            for vehicle in vehicles:
                if not self._vehicle_group_has_records(vehicle, records, speed_only=False):
                    followups.append(self._vehicle_penalty_slot(vehicle, query, round_idx, len(followups)))

        if "sign" in facets and not self._has_facet_records(records, "sign"):
            followups.append(SequentialEvidenceSlot(
                id=f"followup_{round_idx}_sign_catalog",
                query=(
                    "Tra cứu lại mã biển, nhóm biển, hình dạng, ý nghĩa và phạm vi hiệu lực "
                    f"trong QCVN 41:2024 cho câu hỏi: {query}"
                ),
                facet="sign",
                priority=self._next_priority(existing_slots, followups),
                reason="Bổ sung vì nhánh biển báo chưa có căn cứ trực tiếp.",
            ))

        if "source_image" in facets and records and not any(record_image_paths(record) for record in records):
            followups.append(SequentialEvidenceSlot(
                id=f"followup_{round_idx}_source_image",
                query=(
                    "Tìm ảnh trang gốc, phụ lục, bảng hoặc crop biển báo làm căn cứ trực quan "
                    f"cho câu hỏi: {query}"
                ),
                facet="source_image",
                priority=self._next_priority(existing_slots, followups),
                reason="Bổ sung vì chưa có ảnh/căn cứ trực quan.",
            ))

        for result in results:
            if result.slot.facet != "penalty" or result.status != "hit":
                continue
            if self._records_have_penalty_amount(result.records):
                continue
            followups.append(SequentialEvidenceSlot(
                id=f"followup_{round_idx}_amount_{len(followups)}",
                query=(
                    "Tra cứu lại khoản chứa MỨC PHẠT TIỀN CỤ THỂ bằng số, trừ điểm, "
                    "tước giấy phép và biện pháp bổ sung; không chỉ lấy điểm hành vi. "
                    f"Câu hỏi con đang thiếu số tiền: {result.slot.query}"
                ),
                facet="penalty",
                priority=self._next_priority(existing_slots, followups),
                reason="Bổ sung vì căn cứ phạt chưa có số tiền cụ thể.",
            ))

        return followups[:8]

    def _answer_has_unresolved_ambiguity(
        self,
        answer: str,
        query: str,
        profile: Optional[AdaptiveQueryProfile] = None,
    ) -> bool:
        qa = ascii_lower(query)
        aa = ascii_lower(answer)
        if not aa.strip():
            return True
        facets = set(getattr(profile, "facets", None) or [])
        if "aggregation" in facets and "penalty" not in facets:
            return False
        penalty_like = self._looks_like_penalty_query(qa)
        vague_patterns = [
            "phat tien theo quy dinh",
            "muc phat tien tham chieu",
            "tham chieu",
            "muc phat pho bien",
            "can doi chieu",
            "chua ro so tien",
            "phat tien: muc phat",
            "vui long cung cap loai phuong tien",
        ]
        if any(pattern in aa for pattern in vague_patterns):
            return True
        if penalty_like and "dong" not in aa and "vnd" not in aa:
            return True
        if penalty_like and self._needs_vehicle_breakdown(query):
            has_motorbike = "mo to" in aa or "xe may" in aa or "gan may" in aa
            if not ("o to" in aa and has_motorbike and "may chuyen dung" in aa):
                return True
        if self._looks_like_speed_query(qa) and "05 km/h" in aa and "theo quy dinh" in aa:
            return True
        return False

    def _answer_repair_slots(
        self,
        *,
        answer: str,
        query: str,
        profile: AdaptiveQueryProfile,
        plan: QueryPlan,
        results: List[SequentialRetrievalResult],
        records: List[Dict[str, Any]],
        existing_slots: List[SequentialEvidenceSlot],
        repair_idx: int,
    ) -> List[SequentialEvidenceSlot]:
        facets = set(profile.facets or [])
        if "aggregation" in facets and "penalty" not in facets:
            return []
        followups = self._coverage_followup_slots(
            query=query,
            profile=profile,
            plan=plan,
            results=results,
            records=records,
            existing_slots=existing_slots,
            round_idx=repair_idx + 10,
        )
        answer_norm = ascii_lower(answer)
        if self._looks_like_penalty_query(ascii_lower(query)) and ("dong" not in answer_norm and "vnd" not in answer_norm):
            followups.append(SequentialEvidenceSlot(
                id=f"repair_{repair_idx}_specific_amounts",
                query=(
                    "Truy vấn sửa lỗi câu trả lời còn chung chung: tìm tất cả khoản/điểm có "
                    "mức phạt tiền bằng số, trừ điểm và tước GPLX liên quan trực tiếp. "
                    f"Câu hỏi gốc: {query}"
                ),
                facet="penalty",
                priority=self._next_priority(existing_slots, followups),
                reason="Câu trả lời cuối còn thiếu con số tiền phạt.",
            ))
        return followups[:8]

    def _next_priority(
        self,
        existing_slots: List[SequentialEvidenceSlot],
        followups: List[SequentialEvidenceSlot],
    ) -> int:
        values = [slot.priority for slot in existing_slots] + [slot.priority for slot in followups]
        return (max(values) if values else 0) + 1

    def _looks_like_penalty_query(self, qa: str) -> bool:
        return any(
            term in qa
            for term in [
                "phat",
                "xu phat",
                "muc phat",
                "tru diem",
                "tuoc",
                "tam giu",
                "bi gi",
                "xu ly",
                "vi pham",
            ]
        )

    def _looks_like_speed_query(self, qa: str) -> bool:
        return any(term in qa for term in ["toc do", "qua toc", "p127", "p.127"])

    def _needs_vehicle_breakdown(self, query: str) -> bool:
        qa = ascii_lower(query)
        if self._vehicle_scope(query):
            return False
        if not self._looks_like_penalty_query(qa):
            return False
        return any(
            term in qa
            for term in [
                "xe",
                "phuong tien",
                "vi pham",
                "chay",
                "toc do",
                "bien",
                "p127",
                "p.127",
                "tat ca",
                "toan bo",
            ]
        )

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
        scope = self._vehicle_scope(query)
        if scope:
            return [scope]
        qa = ascii_lower(query)
        groups = ["car", "motorbike", "specialized"]
        if not self._looks_like_speed_query(qa) or any(term in qa for term in ["tat ca", "toan bo", "phuong tien", "chay xe"]):
            groups.append("bicycle")
        return groups

    def _vehicle_articles(self, vehicle: str) -> set[str]:
        return {
            "car": {"6", "13"},
            "motorbike": {"7", "14"},
            "specialized": {"8", "15"},
            "bicycle": {"9"},
        }.get(vehicle, set())

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

    def _vehicle_group_has_records(self, vehicle: str, records: List[Dict[str, Any]], *, speed_only: bool) -> bool:
        articles = self._vehicle_articles(vehicle)
        for record in records:
            ref = normalized_legal_reference(record)
            article = str(ref.get("article") or record.get("article") or "")
            doc = ascii_lower(ref.get("document") or record.get("doc_name") or "")
            if "nghi dinh 168" not in doc or article not in articles:
                continue
            if speed_only and not self._record_mentions_speed(record):
                continue
            return True
        return False

    def _record_mentions_speed(self, record: Dict[str, Any]) -> bool:
        text = ascii_lower(source_text(record))
        return any(term in text for term in ["qua toc do", "toc do quy dinh", "toc do toi da", "p.127", "p127"])

    def _vehicle_speed_slot(self, vehicle: str, query: str, round_idx: int, offset: int) -> SequentialEvidenceSlot:
        label = self._vehicle_label(vehicle)
        return SequentialEvidenceSlot(
            id=f"followup_{round_idx}_speed_{vehicle}_{offset}",
            query=(
                f"Tra cứu đầy đủ mọi ngưỡng chạy quá tốc độ/P.127 cho {label} theo "
                f"{self._vehicle_article_hint(vehicle)} Nghị định 168/2024/NĐ-CP: "
                "mốc km/h, điểm/khoản, phạt tiền bằng số, trừ điểm, tước GPLX/tạm giữ nếu có. "
                f"Câu hỏi gốc: {query}"
            ),
            facet="penalty",
            priority=20 + round_idx * 10 + offset,
            reason="Bổ sung nhóm phương tiện/ngưỡng tốc độ còn thiếu.",
        )

    def _vehicle_penalty_slot(self, vehicle: str, query: str, round_idx: int, offset: int) -> SequentialEvidenceSlot:
        label = self._vehicle_label(vehicle)
        return SequentialEvidenceSlot(
            id=f"followup_{round_idx}_vehicle_{vehicle}_{offset}",
            query=(
                f"Tra cứu khả năng xử phạt cho {label} theo {self._vehicle_article_hint(vehicle)} "
                "Nghị định 168/2024/NĐ-CP. Vì câu hỏi gốc mơ hồ, phải lấy đủ hành vi liên quan, "
                "mức phạt tiền bằng số, trừ điểm, tước GPLX/tạm giữ nếu có; không được chỉ ghi 'theo quy định'. "
                f"Câu hỏi gốc: {query}"
            ),
            facet="penalty",
            priority=24 + round_idx * 10 + offset,
            reason="Bổ sung nhóm phương tiện còn thiếu cho câu hỏi xử phạt mơ hồ.",
        )

    def _has_facet_records(self, records: List[Dict[str, Any]], facet: str) -> bool:
        return any(str(record.get("retrieval_slot_facet") or "") == facet for record in records)

    def _records_have_penalty_amount(self, records: List[Dict[str, Any]]) -> bool:
        for record in records:
            penalties = record.get("penalties") if isinstance(record.get("penalties"), dict) else {}
            main = penalties.get("main_penalty") if isinstance(penalties.get("main_penalty"), dict) else {}
            if main.get("min_amount_vnd") or main.get("individual_min_vnd"):
                return True
            text = source_text(record)
            if re.search(r"phạt\s+tiền\s+từ\s+\d", text, flags=re.IGNORECASE):
                return True
            if re.search(r"\d{1,3}(?:\.\d{3})+\s*đồng", text, flags=re.IGNORECASE):
                return True
        return False

    def _retrieve_for_slot(
        self, 
        slot: SequentialEvidenceSlot, 
        plan: QueryPlan, 
        profile: AdaptiveQueryProfile,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Invokes retriever based on facet."""
        expand_depth = int((profile.retrieval_budget or {}).get("expand_depth") or 1)
        if slot.facet == "document_overview" and hasattr(self.retriever, "retrieve_document_overview"):
            return self.retriever.retrieve_document_overview(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "legal_detail" and hasattr(self.retriever, "retrieve_legal_detail"):
            return self.retriever.retrieve_legal_detail(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "aggregation" and hasattr(self.retriever, "retrieve_aggregation"):
            return self.retriever.retrieve_aggregation(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "out_of_scope":
            return []
        if slot.facet == "sign":
            return self.retriever.retrieve_sign(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "table":
            return self.retriever.retrieve_table(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "penalty" and hasattr(self.retriever, "retrieve_penalty"):
            return self.retriever.retrieve_penalty(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "definition" and hasattr(self.retriever, "retrieve_definition"):
            return self.retriever.retrieve_definition(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "procedure" and hasattr(self.retriever, "retrieve_procedure"):
            return self.retriever.retrieve_procedure(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "priority" and hasattr(self.retriever, "retrieve_priority"):
            return self.retriever.retrieve_priority(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "scenario" and hasattr(self.retriever, "retrieve_scenario"):
            return self.retriever.retrieve_scenario(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "source_image" and hasattr(self.retriever, "retrieve_source_image"):
            return self.retriever.retrieve_source_image(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet in {"rule", "procedure", "definition"} and hasattr(self.retriever, "retrieve_general"):
            return self.retriever.retrieve_general(slot.query, top_k=top_k, expand_depth=expand_depth)
        return self.retriever.retrieve(slot.query, top_k=top_k, expand_depth=expand_depth)

    def _tag_records(self, records: List[Dict[str, Any]], slot: SequentialEvidenceSlot) -> List[Dict[str, Any]]:
        tagged: List[Dict[str, Any]] = []
        for record in records or []:
            item = dict(record)
            item["retrieval_slot_id"] = slot.id
            item["retrieval_slot_facet"] = slot.facet
            item["retrieval_slot_query"] = slot.query
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [f"slot:{slot.facet}"]))
            tagged.append(item)
        return tagged

    def _record_key(self, record: Dict[str, Any]) -> str:
        key = record.get("source_chunk_id") or record.get("id")
        if key:
            return str(key)
        ref = record.get("legal_reference") or {}
        return "|".join([
            str(record.get("doc_name") or ref.get("document") or ""),
            str(ref.get("article") or record.get("article") or ""),
            str(ref.get("clause") or record.get("clause") or ""),
            str(ref.get("point") or record.get("point") or ""),
            str((record.get("rag_text") or record.get("content") or "")[:120]),
        ])

    def _result_to_dict(self, result: SequentialRetrievalResult) -> Dict[str, Any]:
        return {
            "slot": result.slot.__dict__,
            "status": result.status,
            "record_count": len(result.records),
            "images": result.images,
            "error": result.error,
        }
