"""
Sequential Retrieval Orchestrator for Multi-Part Legal Queries.

This module implements a strategy to decompose complex user questions into
independent 'evidence slots'. Each slot is retrieved sequentially with targeted logic,
preventing dominant query terms from overshadowing subtler legal requirements.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.rag.adaptive_query import AdaptiveQueryProfile
from src.rag.legal_utils import merge_record_assets, record_image_paths
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
        results: List[SequentialRetrievalResult] = []
        accumulated_records: List[Dict[str, Any]] = []
        seen_keys: set = set()
        records_by_key: Dict[str, Dict[str, Any]] = {}

        for slot in slots:
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
                continue
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
                status="hit" if slot_records else "miss"
            ))

            for record in slot_records:
                rid = self._record_key(record)
                if rid in seen_keys:
                    merge_record_assets(records_by_key[rid], record)
                else:
                    accumulated_records.append(record)
                    records_by_key[rid] = record
                    seen_keys.add(rid)

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

    def _retrieve_for_slot(
        self, 
        slot: SequentialEvidenceSlot, 
        plan: QueryPlan, 
        profile: AdaptiveQueryProfile,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Invokes retriever based on facet."""
        expand_depth = int((profile.retrieval_budget or {}).get("expand_depth") or 1)
        if slot.facet == "sign":
            return self.retriever.retrieve_sign(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "table":
            return self.retriever.retrieve_table(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "penalty" and hasattr(self.retriever, "retrieve_penalty"):
            return self.retriever.retrieve_penalty(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if slot.facet == "definition" and hasattr(self.retriever, "retrieve_definition"):
            return self.retriever.retrieve_definition(slot.query, top_k=top_k, expand_depth=expand_depth, plan=plan)
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
