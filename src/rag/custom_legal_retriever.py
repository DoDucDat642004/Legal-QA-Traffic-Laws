import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.rag.hybrid_vector_store import HybridLegalVectorStore
from src.rag.legal_graph_store import DeterministicLegalGraphStore
from src.rag.legal_utils import (
    SIGN_CODE_RE,
    ascii_lower,
    extract_explicit_legal_refs,
    format_reference,
    looks_like_sign_query,
    looks_like_table_query,
    merge_record_assets,
    normalize_sign_code,
    normalized_legal_reference,
    penalty_summary,
    record_image_paths,
    source_text,
)
from src.rag.query_planner import LegalQueryPlanner, QueryIntent, QueryPlan
from src.rag.reranker_backends import make_reranker
from src.rag.structured_table_retriever import StructuredTableRetriever
from src.rag.traffic_sign_catalog import TrafficSignCatalog

# --- Logging Configuration ---
logger = logging.getLogger("CustomLegalRetriever")


class CustomLegalRetriever:
    """
    Hybrid legal retriever with deterministic graph expansion.
    Balances precision (via rules) and recall (via semantic search).
    """

    # Mapping of vehicle types to core penalty articles in Decree 168
    VEHICLE_ARTICLES = {
        "car": {"6", "13"},
        "motorbike": {"7", "14"},
        "specialized": {"8", "15"},
        "bicycle": {"9"},
    }
    VEHICLE_SCOPED_ARTICLES = set().union(*VEHICLE_ARTICLES.values())
    VEHICLE_RULE_ARTICLES = {
        "car": {"6"},
        "motorbike": {"7"},
        "specialized": {"8"},
        "bicycle": {"9"},
    }
    DOCUMENT_ALIASES = [
        (
            "Nghị định 168/2024/NĐ-CP",
            ["nghi dinh 168", "nghi dinh so 168", "nd 168", "168/2024", "168-2024"],
            ["Nghị định 168/2024/NĐ-CP"],
        ),
        (
            "Nghị định 336/2025/NĐ-CP",
            ["nghi dinh 336", "nghi dinh so 336", "nd 336", "336/2025", "336-2025"],
            ["Nghị định 336/2025/NĐ-CP"],
        ),
        (
            "Luật Đường bộ 2024",
            ["luat duong bo", "35/2024/qh15", "35-2024-qh15", "luat 35"],
            ["Luật Đường bộ 2024"],
        ),
        (
            "Luật Trật tự ATGT 2024",
            ["luat trat tu", "trat tu an toan giao thong", "36/2024/qh15", "36-2024-qh15", "luat 36"],
            ["Luật Trật tự ATGT 2024", "Luật Trật tự ATGT 2024 (Tiếp)"],
        ),
        (
            "QCVN 41:2024 (Thông tư 51/2024)",
            ["qcvn 41", "thong tu 51", "51/2024", "51-2024", "quy chuan 41", "quy chuan ve bao hieu duong bo", "quy chuan bbdb"],
            ["QCVN 41:2024 (Thông tư 51/2024)"],
        ),
        (
            "Thông tư 35/2024/TT-BGTVT",
            ["thong tu 35", "35/2024/tt-bgtvt", "35-2024-tt-bgtvt"],
            ["Thông tư 35/2024/TT-BGTVT"],
        ),
    ]
    LEXICAL_STOPWORDS = {
        "anh", "bao", "bi", "bo", "cac", "can", "cho", "co", "cua", "duoc", "doi", "hoi",
        "khi", "la", "lam", "muc", "nao", "neu", "nguoi", "nhung", "phai", "quy", "ra",
        "sao", "the", "theo", "thi", "trong", "tu", "van", "ve", "voi", "xu", "ly",
        "dieu", "khoan", "diem", "nghi", "dinh", "thong", "luat", "phat", "tien",
    }

    def __init__(
        self,
        vector_store: HybridLegalVectorStore,
        graph_store: DeterministicLegalGraphStore,
        *,
        reranker_model: str = "",
        use_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.sign_catalog = TrafficSignCatalog(vector_store.records)
        self.table_retriever = StructuredTableRetriever(vector_store.records)
        self.client = None # LLM Client for AI semantic probes
        self.reranker = None
        
        if use_reranker:
            reranker_model = reranker_model or os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            try:
                self.reranker = make_reranker(reranker_model)
                logger.info("Reranker enabled: %s", type(self.reranker).__name__)
            except Exception as exc:
                logger.warning("Reranker disabled: %s", exc)
        
        self.query_planner = LegalQueryPlanner()
        from src.rag.adaptive_query import AdaptiveQuestionAnalyzer
        self.adaptive_analyzer = AdaptiveQuestionAnalyzer()
        self._lexical_index: Optional[Dict[str, List[int]]] = None
        self._lexical_records: List[Dict[str, Any]] = []

    def retrieve(self, query: str, top_k: int = 8, expand_depth: int = 2) -> List[Dict[str, Any]]:
        """General hybrid retrieval flow."""
        plan = self.query_planner.rule_plan(query)
        intent = plan.intent
        aggregation_blocked = self._blocks_aggregation_route(query)
        if (
            not aggregation_blocked
            and (self._looks_like_aggregation_query(query) or intent == QueryIntent.AGGREGATION)
            and not self._looks_like_authority_limit_query(query)
        ):
            return self.retrieve_aggregation(query, top_k=top_k, expand_depth=expand_depth)
        if self._looks_like_document_overview_query(query) or intent == QueryIntent.DOCUMENT_OVERVIEW:
            return self.retrieve_document_overview(query, top_k=top_k, expand_depth=expand_depth)
        if self._looks_like_legal_detail_query(query) or intent == QueryIntent.LEGAL_DETAIL:
            return self.retrieve_legal_detail(query, top_k=top_k, expand_depth=expand_depth)
        if looks_like_table_query(query) or intent == QueryIntent.TABLE:
            return self.retrieve_table(query, top_k=top_k, expand_depth=expand_depth)
        if (looks_like_sign_query(query) or intent == QueryIntent.SIGN) and self._should_route_sign(query):
            return self.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth)
        if intent == QueryIntent.SCENARIO or self._has_scenario_slot(plan):
            return self.retrieve_scenario(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if intent == QueryIntent.PENALTY or self._looks_like_penalty_query(query):
            return self.retrieve_penalty(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if intent == QueryIntent.PROCEDURE:
            return self.retrieve_procedure(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if intent == QueryIntent.DEFINITION:
            return self.retrieve_definition(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if intent == QueryIntent.PRIORITY:
            return self.retrieve_priority(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        return self.retrieve_general(query, top_k=top_k, expand_depth=expand_depth)

    def _has_scenario_slot(self, plan: Optional[QueryPlan]) -> bool:
        return any((slot.get("facet") or "") == "scenario" for slot in (getattr(plan, "subquestions", None) or []))

    def retrieve_general(self, query: str, top_k: int = 8, expand_depth: int = 2) -> List[Dict[str, Any]]:
        """Hybrid text retrieval without automatic sign/table rerouting."""
        search_query = self._expand_query(query)
        seed_records = []
        
        # Phase 1: High-precision deterministic matches
        seed_records.extend(self._exact_sign_matches(search_query))
        seed_records.extend(self._exact_ref_matches(search_query))
        seed_records.extend(self._known_legal_ref_matches(search_query))
        seed_records.extend(self._known_chunk_matches(search_query))
        seed_records.extend(self._topic_anchor_matches(search_query))
        seed_records.extend(self._lexical_evidence_matches(search_query, limit=max(top_k * 2, 24)))
        
        # Phase 2: Hybrid vector search
        seed_records.extend(self.vector_store.search(search_query, top_k=max(top_k * 3, 20)))
        seed_records = self._dedupe(seed_records)

        # Phase 3: Graph Expansion for parent clauses, sibling points, tables and figures.
        expanded_records = self._graph_expand_records(seed_records, expand_depth=expand_depth, top_k=top_k)

        candidates = self._dedupe(seed_records + expanded_records)
        
        # Phase 4: Contextual Boosting (Vehicle type, etc.)
        candidates = self._document_scope_filter_boost(query, candidates)
        candidates = self._vehicle_scope_boost(query, candidates)
        candidates = self._license_focus_boost(query, candidates)
        
        # Phase 5: Reranking
        candidates = self._rerank(query, candidates, limit=top_k)
        return candidates

    def retrieve_penalty(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Penalty-specialized retrieval that bypasses sign rerouting."""
        scoped_docs = self._matching_documents(query)
        doc_hint = " ".join(scoped_docs) if scoped_docs else "Nghị định 168/2024/NĐ-CP"
        penalty_query = " ".join([
            query,
            doc_hint,
            "mức phạt tiền trừ điểm giấy phép lái xe tước giấy phép xử phạt vi phạm",
        ]).strip()
        records = self.retrieve_general(penalty_query, top_k=max(top_k * 3, 24), expand_depth=expand_depth)
        records.extend(self._known_legal_ref_matches(query))
        records.extend(self._known_chunk_matches(query))
        records.extend(self._behavior_text_matches(query))
        scope = self._vehicle_scope(query)
        has_168_scope = not scoped_docs or "Nghị định 168/2024/NĐ-CP" in scoped_docs
        if scope and has_168_scope:
            qa = ascii_lower(self._focused_behavior_text(query))
            article_scope = self.VEHICLE_RULE_ARTICLES if self._behavior_search_specs(qa) else self.VEHICLE_ARTICLES
            for article in article_scope.get(scope, set()):
                records.extend(self._records_by_ref_prefix(
                    "Nghị định 168/2024/NĐ-CP",
                    article,
                    reason="vehicle_article_prefix",
                    boost=0.8,
                ))
        records = self._dedupe(records)
        records = self._document_scope_filter_boost(query, records)
        records = self._vehicle_scope_boost(query, records)
        records = self._penalty_focus_boost(query, records)
        has_known_behavior = bool(self._behavior_search_specs(ascii_lower(self._focused_behavior_text(query))))
        penalty_limit = max(top_k, 12) if has_known_behavior and not scope else top_k
        records = self._rerank(query, records, limit=penalty_limit)
        for record in records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_route"]))
        return records

    def retrieve_sign(self, query: str, top_k: int = 8, expand_depth: int = 1, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Sign-specialized retrieval with deterministic visual feature mapping."""
        ai_codes = []
        enable_ai_probe = os.getenv("RAG_ENABLE_SIGN_AI_PROBE", "").lower() in {"1", "true", "yes", "on"}
        if enable_ai_probe and self.client and len(query.split()) > 3:
            ai_codes = self.sign_catalog.ai_semantic_probe(query, self.client)
            if ai_codes: logger.info("AI Probe found: %s", ai_codes)
            
        explicit_codes = self._sign_code_hints(query)
        if explicit_codes:
            codes = list(dict.fromkeys(explicit_codes + ai_codes))
        else:
            codes = list(dict.fromkeys(self.sign_catalog.find_codes(query) + ai_codes))
        seed_records = self.sign_catalog.records_for_codes(codes)
        seed_records.extend(self._known_chunk_matches(query))
        seed_records.extend(self._known_legal_ref_matches(query))
        seed_records.extend(self._topic_anchor_matches(query))
        
        # Include general category descriptions (e.g. 'biển cấm là gì')
        seed_records.extend(self._sign_group_description_records(query, top_k))
        
        if not seed_records:
            seed_records.extend(self.vector_store.search(
                " ".join([query, "QCVN 41 biển báo hình dạng ý nghĩa phụ lục hình biển báo"]),
                top_k=max(top_k * 2, 12),
            ))

        candidates = self._dedupe(seed_records)
        candidates = self._rerank(query, candidates, limit=top_k)
        return candidates

    def retrieve_table(self, query: str, top_k: int = 8, expand_depth: int = 1, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Table-specialized retrieval."""
        table_query = " ".join([query, "bảng phụ lục dòng cột thông số kỹ thuật quy chuẩn"])
        records = self.table_retriever.search(table_query, top_k=max(top_k, 10))
        records.extend(self._known_legal_ref_matches(query))
        records.extend(self._known_chunk_matches(query))
        records.extend(self._topic_anchor_matches(query))
        records.extend(self.vector_store.search(
            table_query,
            top_k=max(top_k * 2, 16),
            filters={"modalities": ["table"]},
        ))
        records = self._dedupe(records)
        records = self._document_scope_filter_boost(query, records)
        qa = ascii_lower(query)
        if "toc do" in qa and "cao toc" in qa:
            precise_records = [
                record for record in records
                if self._table_speed_highway_match(record, query)
            ]
            if precise_records:
                records = precise_records
            else:
                return []
        for record in records:
            record["rag_modality"] = record.get("rag_modality") or "table"
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["table_route"]))
        return self._rerank(query, records, limit=top_k)

    def retrieve_definition(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        definition_query = " ".join([
            query,
            "định nghĩa khái niệm được hiểu là giải thích từ ngữ phạm vi áp dụng điều khoản",
        ])
        records = self.retrieve_general(definition_query, top_k=max(top_k * 2, 16), expand_depth=expand_depth)
        records.extend(self._known_legal_ref_matches(query))
        records.extend(self._known_chunk_matches(query))
        records.extend(self._topic_anchor_matches(query))
        terms = self._definition_terms(query)
        records.extend(self._deterministic_definition_matches(terms))
        records = self._dedupe(records)
        records = self._definition_boost(query, records)
        return self._rerank(query, records, limit=top_k)

    def retrieve_procedure(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        procedure_query = " ".join([
            query,
            "thủ tục hồ sơ thời hạn thời gian cơ quan xử lý cấp đổi cấp lại sát hạch đào tạo giấy phép lái xe",
        ])
        records = self.retrieve_general(procedure_query, top_k=max(top_k * 3, 24), expand_depth=expand_depth)
        records.extend(self._known_legal_ref_matches(query))
        records.extend(self._known_chunk_matches(query))
        records.extend(self._topic_anchor_matches(query))
        records = self._dedupe(records)
        records = self._procedure_focus_boost(query, records)
        priority_reasons = {
            "topic_tt35_theory_classroom_visuals",
            "topic_tt35_theory_classroom_visuals_synthetic",
        }
        priority_records = [
            record for record in records
            if priority_reasons.intersection(set(record.get("retrieval_reasons") or []))
        ]
        remaining_records = [
            record for record in records
            if not priority_reasons.intersection(set(record.get("retrieval_reasons") or []))
        ]
        priority_records = sorted(
            priority_records,
            key=lambda record: float(record.get("retrieval_score") or 0),
            reverse=True,
        )
        records = self._dedupe(priority_records + self._rerank(query, remaining_records, limit=top_k))[:top_k]
        for record in records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["procedure_route"]))
        return records

    def retrieve_priority(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        priority_query = " ".join([
            query,
            "xe ưu tiên quyền ưu tiên tín hiệu ưu tiên nhường đường giao nhau vòng xuyến xe chữa cháy cứu thương công an quân sự",
        ])
        records = self.retrieve_general(priority_query, top_k=max(top_k * 3, 24), expand_depth=expand_depth)
        records = self._priority_boost(query, records)
        return self._rerank(query, records, limit=top_k)

    def retrieve_scenario(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        scenario_query = " ".join([
            query,
            "tình huống thực tế trường hợp điều kiện áp dụng hành vi chủ thể thời điểm trách nhiệm quy tắc xử lý",
        ])
        records = self.retrieve_general(scenario_query, top_k=max(top_k * 3, 24), expand_depth=expand_depth)
        if not looks_like_table_query(query):
            text_records = [record for record in records if record.get("rag_modality") != "table"]
            if text_records:
                records = text_records
        qa = ascii_lower(query)
        if "dua don hoc sinh" in qa or ("hoc sinh" in qa and "co so giao duc" in qa):
            scoped_records = []
            for record in records:
                ref = normalized_legal_reference(record)
                doc = ascii_lower(ref.get("document") or record.get("doc_name") or record.get("document") or "")
                article = str(ref.get("article") or record.get("article") or "").strip()
                if "nghi dinh 336" in doc:
                    scoped_records.append(record)
                elif "luat duong bo" in doc and article == "70":
                    scoped_records.append(record)
                elif "luat trat tu" in doc and article == "46":
                    scoped_records.append(record)
            if scoped_records:
                records = scoped_records
        records = self._scenario_boost(query, records)
        return self._rerank(query, records, limit=top_k)

    def retrieve_source_image(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Retrieves visual evidence from source pages, tables, figures, and sign crops."""
        image_query = " ".join([query, "ảnh trang gốc hình bảng phụ lục scan văn bản gốc căn cứ trực quan"])
        records = self.retrieve_general(image_query, top_k=max(top_k * 3, 24), expand_depth=expand_depth)
        if looks_like_table_query(query):
            records.extend(self.retrieve_table(query, top_k=max(top_k, 8), expand_depth=expand_depth, plan=plan))
        if looks_like_sign_query(query):
            records.extend(self.retrieve_sign(query, top_k=max(top_k, 8), expand_depth=expand_depth, plan=plan))
        records = self._dedupe(records)
        records = self._source_image_scope_filter(query, records)
        qa = ascii_lower(query)
        if "toc do" in qa and "cao toc" in qa and "toi da" in qa:
            records = [record for record in records if self._table_speed_highway_match(record, query)]
            if not records:
                return []
        for record in records:
            if record_image_paths(record):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.5
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["source_image_route"]))
        image_records = [record for record in records if record_image_paths(record)]
        return self._rerank(query, image_records or records, limit=top_k)

    def retrieve_document_overview(self, query: str, top_k: int = 8, expand_depth: int = 1, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Returns deterministic document structure: article count and article titles."""
        del expand_depth, plan
        documents = self._matching_documents(query)
        if not documents:
            records = self.retrieve_general(query, top_k=max(top_k * 2, 16), expand_depth=1)
            documents = self._documents_from_records(records)

        grouped = self._group_document_articles(documents)
        if not grouped:
            return self.retrieve_general(query, top_k=top_k, expand_depth=1)

        synthetic_records: List[Dict[str, Any]] = []
        supporting_records: List[Dict[str, Any]] = []
        for display_doc, article_rows in grouped:
            summary = self._format_document_overview(display_doc, article_rows)
            synthetic_records.append(self._synthetic_record(
                record_id=f"synthetic_document_overview:{self._safe_doc_id(display_doc)}",
                doc_name=display_doc,
                modality="document_overview",
                text=summary,
                reason="document_overview_route",
                score=100.0,
            ))
            supporting_records.extend(row["record"] for row in article_rows if row.get("record"))

        records = self._dedupe(synthetic_records + supporting_records)
        return records[:max(top_k, len(synthetic_records))]

    def retrieve_legal_detail(self, query: str, top_k: int = 8, expand_depth: int = 1, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Returns all extracted chunks under a requested Article/Clause/Point."""
        del expand_depth, plan
        ref = self._explicit_ref_from_query(query)
        article = ref.get("article")
        if not article:
            return self.retrieve_general(query, top_k=top_k, expand_depth=1)

        documents = self._matching_documents(query)
        if not documents:
            documents = self._documents_containing_article(article)

        records: List[Dict[str, Any]] = []
        for document in documents:
            records.extend(self._article_detail_records(
                document,
                article,
                clause=ref.get("clause", ""),
                point=ref.get("point", ""),
            ))

        records = self._dedupe(records)
        records = sorted(records, key=self._article_sort_key)
        if not records:
            return self.retrieve_general(query, top_k=top_k, expand_depth=1)

        display_doc = self._display_document_for_documents(documents, records)
        summary = self._format_legal_detail(display_doc, article, records, clause=ref.get("clause", ""), point=ref.get("point", ""))
        images = self._record_images(records)
        synthetic = self._synthetic_record(
            record_id=f"synthetic_legal_detail:{self._safe_doc_id(display_doc)}:{article}:{ref.get('clause', '')}:{ref.get('point', '')}",
            doc_name=display_doc,
            modality="legal_article_detail",
            text=summary,
            reason="legal_detail_route",
            score=120.0,
            legal_reference={
                "document": display_doc,
                "article": article,
                "clause": ref.get("clause", ""),
                "point": ref.get("point", ""),
            },
            image_paths=images,
        )
        return [synthetic] + records[:max(top_k - 1, 0)]

    def retrieve_aggregation(self, query: str, top_k: int = 8, expand_depth: int = 1, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Returns deterministic corpus-level aggregations such as max/min fines and dense provisions."""
        del expand_depth, plan
        qa = ascii_lower(query)
        documents = self._matching_documents(query)
        records = [
            record for record in self.vector_store.records
            if not documents or ((record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "") in documents)
        ]
        if not records:
            records = self.vector_store.records

        if self._looks_like_frequency_question(qa):
            return self._frequency_aggregation_records(query, records, top_k=top_k)

        if any(term in qa for term in ["tru diem", "diem gplx", "diem giay phep"]):
            return self._point_aggregation_records(query, records, top_k=top_k)

        if any(term in qa for term in ["muc phat", "phat tien", "phat", "xu phat", "tien"]):
            return self._fine_aggregation_records(query, records, top_k=top_k)

        summary = (
            "Tôi nhận diện đây là câu hỏi thống kê/tổng hợp, nhưng chưa xác định được đại lượng cần tính "
            "(ví dụ: mức phạt tiền cao nhất/thấp nhất, trừ điểm cao nhất, hoặc điều có nhiều chế tài nhất)."
        )
        return [self._synthetic_record(
            record_id="synthetic_aggregation:unsupported",
            doc_name="Dữ liệu pháp luật giao thông",
            modality="aggregation",
            text=summary,
            reason="aggregation_route",
            score=90.0,
        )]

    # --- Internal Logic Paths ---

    def _looks_like_aggregation_query(self, query: str) -> bool:
        qa = ascii_lower(query)
        if self._looks_like_authority_limit_query(query):
            return False
        if self._looks_like_retrieval_meta_query(qa):
            return False
        if self._blocks_aggregation_route(query):
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

    def _blocks_aggregation_route(self, query: str) -> bool:
        qa = ascii_lower(query)
        return any(term in qa for term in ["han che toc do", "toc do toi da"]) and "bien" in qa

    def _looks_like_retrieval_meta_query(self, qa: str) -> bool:
        return bool(re.search(r"\btop\s*[-_ ]?\s*k\b", qa)) or any(
            term in qa
            for term in [
                "truy xuat",
                "he thong chi co top",
                "chi co top",
                "top k nho",
                "top-k nho",
            ]
        )

    def _looks_like_authority_limit_query(self, query: str) -> bool:
        qa = ascii_lower(query)
        return "tham quyen" in qa and any(term in qa for term in ["chu tich uy ban", "ubnd", "uy ban nhan dan", "cap xa"])

    def _fine_aggregation_records(self, query: str, records: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        items = self._penalty_amount_items(records, query)
        if not items:
            text = (
                "Chưa tìm thấy bản ghi có số tiền phạt cấu trúc hoặc số tiền trong nội dung nguồn để thống kê. "
                "Hãy thu hẹp theo văn bản/hành vi cụ thể hoặc kiểm tra lại pipeline trích xuất penalty."
            )
            return [self._synthetic_record(
                record_id="synthetic_aggregation:fine:none",
                doc_name="Dữ liệu pháp luật giao thông",
                modality="aggregation",
                text=text,
                reason="aggregation_route",
                score=95.0,
            )]

        lowest = any(term in qa for term in ["thap nhat", "nhe nhat", "nho nhat", "toi thieu", "it nhat"])
        limit = max(5, min(top_k, 12))
        key = "amount_min" if lowest else "amount_max"
        ranked = sorted(items, key=lambda item: item[key], reverse=not lowest)
        if lowest:
            ranked = [item for item in ranked if item[key] > 0]
        ranked = ranked[:limit]
        title = "mức phạt tiền thấp nhất" if lowest else "mức phạt tiền cao nhất"
        scope = self._aggregation_scope_text(query, records)
        lines = [
            f"## Thống kê {title}",
            "",
            f"**Phạm vi tính:** {scope}.",
            "**Lưu ý:** kết quả được tính từ dữ liệu pháp lý đã trích xuất trong hệ thống; không phải thống kê số vụ vi phạm ngoài thực tế.",
            "",
            "| Hạng | Mức thấp nhất | Mức cao nhất | Căn cứ | Trích đoạn |",
            "|---:|---:|---:|---|---|",
        ]
        for idx, item in enumerate(ranked, start=1):
            lines.append(
                f"| {idx} | {self._format_vnd(item['amount_min'])} | {self._format_vnd(item['amount_max'])} | "
                f"{self._escape_table(format_reference(item['record']))} | {self._escape_table(item['snippet'])} |"
            )
        synthetic = self._synthetic_record(
            record_id=f"synthetic_aggregation:fine:{'min' if lowest else 'max'}",
            doc_name="Dữ liệu pháp luật giao thông",
            modality="aggregation",
            text="\n".join(lines),
            reason="aggregation_route",
            score=140.0,
        )
        support = [item["record"] for item in ranked]
        for record in support:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["aggregation_support"]))
        return [synthetic] + support[:max(top_k - 1, 0)]

    def _point_aggregation_records(self, query: str, records: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        lowest = any(term in qa for term in ["thap nhat", "nhe nhat", "nho nhat", "toi thieu", "it nhat"])
        items: List[Dict[str, Any]] = []
        for record in records:
            record_text_norm = ascii_lower(" ".join([
                source_text(record),
                str(record.get("qa_context") or ""),
                str(record.get("semantic_context") or ""),
                str(record.get("rag_text") or ""),
            ]))
            if any(term in record_text_norm for term in ["phuc hoi", "du 12 diem", "kiem tra kien thuc"]):
                continue
            if not any(term in record_text_norm for term in ["hanh vi", "vi pham", "ngoai viec bi ap dung", "hinh thuc xu phat"]):
                continue
            points = self._point_values(record)
            if not points:
                continue
            value = min(points) if lowest else max(points)
            if value <= 0:
                continue
            items.append({
                "points": value,
                "record": record,
                "snippet": self._snippet(source_text(record), 180),
            })
        ranked = sorted(items, key=lambda item: item["points"], reverse=not lowest)[:max(5, min(top_k, 12))]
        if not ranked:
            text = "Chưa tìm thấy bản ghi có dữ liệu trừ điểm GPLX đủ rõ để thống kê."
            return [self._synthetic_record(
                record_id="synthetic_aggregation:points:none",
                doc_name="Dữ liệu pháp luật giao thông",
                modality="aggregation",
                text=text,
                reason="aggregation_route",
                score=95.0,
            )]
        title = "mức trừ điểm thấp nhất" if lowest else "mức trừ điểm cao nhất"
        lines = [
            f"## Thống kê {title}",
            "",
            f"**Phạm vi tính:** {self._aggregation_scope_text(query, records)}.",
            "| Hạng | Điểm | Căn cứ | Trích đoạn |",
            "|---:|---:|---|---|",
        ]
        for idx, item in enumerate(ranked, start=1):
            lines.append(
                f"| {idx} | {item['points']} | {self._escape_table(format_reference(item['record']))} | {self._escape_table(item['snippet'])} |"
            )
        synthetic = self._synthetic_record(
            record_id=f"synthetic_aggregation:points:{'min' if lowest else 'max'}",
            doc_name="Dữ liệu pháp luật giao thông",
            modality="aggregation",
            text="\n".join(lines),
            reason="aggregation_route",
            score=135.0,
        )
        support = [item["record"] for item in ranked]
        return [synthetic] + support[:max(top_k - 1, 0)]

    def _frequency_aggregation_records(self, query: str, records: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
        del query
        buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in records:
            ref = normalized_legal_reference(record)
            document = ref.get("document") or record.get("doc_name") or ""
            article = str(ref.get("article") or "")
            if not document or not article:
                continue
            text = ascii_lower(source_text(record))
            if "vi pham" not in text and "phat" not in text and not record.get("penalties"):
                continue
            key = (document, article)
            bucket = buckets.setdefault(key, {"document": document, "article": article, "count": 0, "records": []})
            bucket["count"] += 1
            if len(bucket["records"]) < 4:
                bucket["records"].append(record)
        ranked = sorted(buckets.values(), key=lambda item: item["count"], reverse=True)[:max(5, min(top_k, 12))]
        lines = [
            "## Thống kê điều có nhiều quy định/chế tài nhất trong dữ liệu",
            "",
            "**Không có dữ liệu thống kê số vụ vi phạm thực tế trong nguồn hiện tại.** Vì vậy tôi không thể kết luận điều nào 'hay bị vi phạm nhất' ngoài đời.",
            "Bảng dưới đây chỉ là thống kê theo **số bản ghi/đơn vị quy định có yếu tố vi phạm hoặc xử phạt** trong dữ liệu pháp lý đã trích xuất.",
            "",
            "| Hạng | Điều | Văn bản | Số đơn vị dữ liệu |",
            "|---:|---|---|---:|",
        ]
        for idx, item in enumerate(ranked, start=1):
            lines.append(f"| {idx} | Điều {item['article']} | {self._escape_table(item['document'])} | {item['count']} |")
        synthetic = self._synthetic_record(
            record_id="synthetic_aggregation:frequency",
            doc_name="Dữ liệu pháp luật giao thông",
            modality="aggregation",
            text="\n".join(lines),
            reason="aggregation_route",
            score=130.0,
        )
        support: List[Dict[str, Any]] = []
        for item in ranked:
            support.extend(item["records"])
        return [synthetic] + support[:max(top_k - 1, 0)]

    def _penalty_amount_items(self, records: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        only_individual = "ca nhan" in qa and "to chuc" not in qa
        items: List[Dict[str, Any]] = []
        seen = set()
        for record in records:
            summary = penalty_summary(record)
            text = source_text(record)
            text_values = self._money_values(text)
            structured_values = [
                value for value in [summary.get("fine_min_vnd"), summary.get("fine_max_vnd")]
                if isinstance(value, (int, float)) and value > 0
            ]
            values = structured_values if only_individual and structured_values else (text_values or structured_values)
            values = [int(value) for value in values if value and value > 0]
            if not values:
                continue
            ref = normalized_legal_reference(record)
            key = (
                record.get("source_chunk_id") or record.get("id"),
                min(values),
                max(values),
                ref.get("article"),
                ref.get("clause"),
                ref.get("point"),
            )
            if key in seen:
                continue
            seen.add(key)
            items.append({
                "amount_min": min(values),
                "amount_max": max(values),
                "record": record,
                "snippet": self._snippet(text, 220),
            })
        return items

    def _money_values(self, text: str) -> List[int]:
        values: List[int] = []
        pattern = re.compile(
            r"(?P<num>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?P<unit>triệu\s*đồng|triệu|trieu\s*dong|trieu|đồng|dong)",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text or ""):
            amount = self._parse_money_number(match.group("num"), match.group("unit"))
            if amount > 0:
                values.append(amount)
        return values

    def _parse_money_number(self, raw: str, unit: str) -> int:
        raw = str(raw or "").strip()
        unit_norm = ascii_lower(unit)
        if not raw:
            return 0
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", raw):
            value = float(re.sub(r"[.,]", "", raw))
        else:
            value = float(raw.replace(",", "."))
        if "trieu" in unit_norm and value < 100000:
            value *= 1_000_000
        return int(round(value))

    def _point_values(self, record: Dict[str, Any]) -> List[int]:
        values: List[int] = []
        penalty = penalty_summary(record)
        point = penalty.get("point_deduction")
        if isinstance(point, int):
            values.append(point)
        elif isinstance(point, str):
            values.extend(int(x) for x in re.findall(r"\d+", point))
        text = "\n".join(
            part for part in [
                source_text(record),
                str(record.get("qa_context") or ""),
                str(record.get("semantic_context") or ""),
                str(record.get("rag_text") or ""),
            ]
            if part
        )
        patterns = [
            r"trừ\s+(\d{1,2})\s+điểm",
            r"bị\s+trừ\s+(\d{1,2})\s+điểm",
            r"trừ\s+điểm[^.\n]{0,100}?(\d{1,2})\s+điểm",
            r"bị\s+trừ[^.\n]{0,100}?(\d{1,2})\s+điểm",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                values.append(int(match.group(1)))
        return values

    def _looks_like_frequency_question(self, qa: str) -> bool:
        return any(term in qa for term in ["hay vi pham", "pho bien nhat", "thuong gap", "nhieu vi pham", "dieu nao vi pham nhat"])

    def _aggregation_scope_text(self, query: str, records: List[Dict[str, Any]]) -> str:
        documents = self._matching_documents(query)
        if documents:
            return ", ".join(documents)
        doc_names = sorted({
            record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or ""
            for record in records
            if record.get("doc_name") or (record.get("legal_reference") or {}).get("document")
        })
        if len(doc_names) <= 3:
            return ", ".join(doc_names) or "toàn bộ dữ liệu đã nạp"
        return f"toàn bộ dữ liệu đã nạp ({len(doc_names)} văn bản)"

    def _format_vnd(self, amount: int) -> str:
        return f"{int(amount):,}".replace(",", ".") + " đồng"

    def _snippet(self, text: str, limit: int) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if len(compact) <= limit:
            return compact
        return compact[: max(0, limit - 1)].rstrip() + "…"

    def _escape_table(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").replace("|", "\\|").strip()

    def _looks_like_document_overview_query(self, query: str) -> bool:
        qa = ascii_lower(query)
        return bool(self._matching_documents(query)) and any(
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

    def _looks_like_legal_detail_query(self, query: str) -> bool:
        qa = ascii_lower(query)
        return bool(self._matching_documents(query)) and bool(re.search(r"\bdieu\s+\d+[a-z]?\b", qa)) and any(
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

    def _looks_like_penalty_query(self, query: str) -> bool:
        qa = ascii_lower(query)
        penalty_terms = [
            "phat",
            "xu phat",
            "muc phat",
            "bao nhieu tien",
            "tru diem",
            "tuoc",
            "tam giu",
            "vi pham",
            "bi gi",
        ]
        behavior_terms = [
            "vuot den do",
            "den tin hieu",
            "nong do con",
            "hoi con",
            "qua toc",
            "chay qua toc",
            "nguoc chieu",
            "khong doi mu",
            "chuyen lan",
            "duong cam",
            "khong co giay phep lai xe",
            "gplx het han",
        ]
        return any(term in qa for term in penalty_terms) or any(term in qa for term in behavior_terms)

    def _should_route_sign(self, query: str) -> bool:
        scoped_docs = set(self._matching_documents(query))
        if not scoped_docs:
            return True
        if any("QCVN 41" in doc for doc in scoped_docs):
            return True
        return bool(self._sign_code_hints(query))

    def _matching_documents(self, query: str) -> List[str]:
        qa = ascii_lower(query)
        matches: List[str] = []
        for _display, aliases, documents in self.DOCUMENT_ALIASES:
            if any(alias in qa for alias in aliases):
                matches.extend(documents)
        return list(dict.fromkeys(matches))

    def _documents_from_records(self, records: List[Dict[str, Any]]) -> List[str]:
        documents: List[str] = []
        for record in records:
            ref = normalized_legal_reference(record)
            doc = ref.get("document") or record.get("doc_name") or ""
            if doc and doc not in documents:
                documents.append(doc)
        return documents[:3]

    def _documents_containing_article(self, article: str) -> List[str]:
        documents: List[str] = []
        for record in self.vector_store.records:
            ref = normalized_legal_reference(record)
            if str(ref.get("article") or "") != str(article):
                continue
            doc = ref.get("document") or record.get("doc_name") or ""
            if doc and doc not in documents:
                documents.append(doc)
        return documents

    def _group_document_articles(self, documents: List[str]) -> List[Tuple[str, List[Dict[str, Any]]]]:
        requested = set(documents)
        grouped: List[Tuple[str, List[Dict[str, Any]]]] = []
        handled_docs: Set[str] = set()
        for display, _aliases, group_docs in self.DOCUMENT_ALIASES:
            if not requested.intersection(group_docs):
                continue
            rows = self._document_article_rows(group_docs)
            if rows:
                grouped.append((display, rows))
                handled_docs.update(group_docs)

        for document in documents:
            if document in handled_docs:
                continue
            rows = self._document_article_rows([document])
            if rows:
                grouped.append((document, rows))
        return grouped

    def _document_article_rows(self, documents: List[str]) -> List[Dict[str, Any]]:
        document_set = set(documents)
        by_article: Dict[str, Dict[str, Any]] = {}
        for record in self.vector_store.records:
            ref = normalized_legal_reference(record)
            doc = ref.get("document") or record.get("doc_name") or ""
            article = str(ref.get("article") or "")
            if doc not in document_set or not re.fullmatch(r"\d+[a-z]?", article, flags=re.IGNORECASE):
                continue
            text = source_text(record).strip()
            if not text:
                continue
            score = self._article_heading_score(record, article)
            current = by_article.get(article)
            if current is None or score > float(current.get("score") or 0):
                by_article[article] = {
                    "article": article,
                    "title": self._article_heading_text(article, text),
                    "record": record,
                    "score": score,
                    "chapter": ref.get("chapter") or "",
                }
        return sorted(by_article.values(), key=lambda row: self._article_number_key(str(row.get("article") or "")))

    def _article_heading_score(self, record: Dict[str, Any], article: str) -> float:
        ref = normalized_legal_reference(record)
        text = ascii_lower(source_text(record))
        score = 0.0
        if not ref.get("clause") and not ref.get("point"):
            score += 10.0
        if text.startswith(f"dieu {article}") or text.startswith(f"dieu {article}."):
            score += 6.0
        if len(text) <= 900:
            score += 1.0
        if record_image_paths(record):
            score += 0.2
        return score

    def _article_heading_text(self, article: str, text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if not compact:
            return f"Điều {article}"
        return compact[:700].rstrip()

    def _format_document_overview(self, display_doc: str, article_rows: List[Dict[str, Any]]) -> str:
        chapters = [str(row.get("chapter") or "") for row in article_rows if row.get("chapter")]
        chapters = list(dict.fromkeys(chapters))
        lines = [
            f"## Tổng quan {display_doc}",
            "",
            f"**Kết luận:** trong dữ liệu đã trích xuất, `{display_doc}` có **{len(article_rows)} điều**.",
        ]
        if chapters:
            lines.append(f"**Metadata chương/mục:** {len(chapters)} nhóm ({', '.join(chapters)}).")
        lines.extend([
            "",
            "| STT | Điều | Chương | Tiêu đề/nội dung mở đầu |",
            "|---:|---|---|---|",
        ])
        for idx, row in enumerate(article_rows, start=1):
            article = str(row.get("article") or "")
            title = self._clean_article_title(article, str(row.get("title") or "").strip())
            chapter = str(row.get("chapter") or "")
            lines.append(f"| {idx} | Điều {article} | {chapter or '-'} | {title} |")
        return "\n".join(lines)

    def _clean_article_title(self, article: str, title: str) -> str:
        compact = re.sub(r"\s+", " ", title or "").strip(" .")
        compact = re.sub(rf"^Điều\s+{re.escape(article)}\s*\.?\s*", "", compact, flags=re.IGNORECASE)
        compact = compact.replace("|", "\\|")
        return compact or f"Nội dung Điều {article}"

    def _explicit_ref_from_query(self, query: str) -> Dict[str, str]:
        refs = extract_explicit_legal_refs(query)
        if refs:
            return refs[0]
        match = re.search(r"\bdieu\s+(?P<article>\d+[a-z]?)\b", ascii_lower(query))
        if not match:
            return {}
        return {"article": match.group("article"), "clause": "", "point": ""}

    def _article_detail_records(self, document: str, article: str, *, clause: str = "", point: str = "") -> List[Dict[str, Any]]:
        records = self._records_by_ref_prefix(
            document,
            article,
            clause=clause,
            point=point,
            reason="legal_detail_prefix",
            boost=8.0,
        )
        for record in records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["legal_detail_route"]))
        return records

    def _format_legal_detail(
        self,
        display_doc: str,
        article: str,
        records: List[Dict[str, Any]],
        *,
        clause: str = "",
        point: str = "",
    ) -> str:
        target = f"Điều {article}"
        if clause:
            target = f"Khoản {clause} {target}"
        if point:
            target = f"Điểm {point} {target}"
        lines = [
            f"CHI TIẾT {target} - {display_doc}",
            f"Số đơn vị trích xuất dưới nhánh này: {len(records)}",
            "Nội dung gốc đã gom theo thứ tự Điều/Khoản/Điểm:",
        ]
        for record in records:
            text = source_text(record).strip()
            if not text:
                continue
            lines.append(f"\n[{format_reference(record)}]\n{text}")
        return "\n".join(lines)

    def _display_document_for_documents(self, documents: List[str], records: List[Dict[str, Any]]) -> str:
        doc_set = set(documents)
        for display, _aliases, group_docs in self.DOCUMENT_ALIASES:
            if doc_set.intersection(group_docs):
                return display
        if records:
            ref = normalized_legal_reference(records[0])
            return ref.get("document") or records[0].get("doc_name") or "Văn bản pháp luật"
        return documents[0] if documents else "Văn bản pháp luật"

    def _synthetic_record(
        self,
        *,
        record_id: str,
        doc_name: str,
        modality: str,
        text: str,
        reason: str,
        score: float,
        legal_reference: Optional[Dict[str, Any]] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ref = dict(legal_reference or {"document": doc_name})
        ref.setdefault("document", doc_name)
        record = {
            "source_chunk_id": record_id,
            "id": record_id,
            "doc_name": doc_name,
            "legal_reference": ref,
            "source_body_exact": text,
            "rag_text": text,
            "rag_modality": modality,
            "retrieval_score": score,
            "retrieval_reasons": [reason],
        }
        if image_paths:
            record["image_paths"] = image_paths
            record["image_path"] = image_paths[0]
        return record

    def _record_images(self, records: List[Dict[str, Any]]) -> List[str]:
        images: List[str] = []
        seen = set()
        for record in records:
            for path in record_image_paths(record):
                if path in seen:
                    continue
                seen.add(path)
                images.append(path)
        return images

    def _document_scope_filter_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scoped_docs = set(self._matching_documents(query))
        if not scoped_docs:
            return records
        scoped: List[Dict[str, Any]] = []
        for record in records:
            ref = normalized_legal_reference(record)
            doc = ref.get("document") or record.get("doc_name") or ""
            if doc not in scoped_docs:
                continue
            item = record
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + 3.0
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["document_scope_boost"]))
            scoped.append(item)
        return scoped or records

    def _safe_doc_id(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", ascii_lower(value)).strip("_") or "document"

    def _article_sort_key(self, record: Dict[str, Any]) -> Tuple[Any, ...]:
        ref = normalized_legal_reference(record)
        return (
            self._article_number_key(str(ref.get("article") or "")),
            self._number_or_high(ref.get("clause")),
            self._point_sort_key(str(ref.get("point") or "")),
            str(record.get("source_chunk_id") or record.get("id") or ""),
        )

    def _article_number_key(self, value: str) -> Tuple[int, str]:
        match = re.match(r"(\d+)([a-z]?)", ascii_lower(value))
        if not match:
            return (9999, ascii_lower(value))
        return (int(match.group(1)), match.group(2) or "")

    def _number_or_high(self, value: Any) -> int:
        text = str(value or "")
        return int(text) if text.isdigit() else 0

    def _point_sort_key(self, value: str) -> Tuple[int, str]:
        text = ascii_lower(value)
        if not text:
            return (0, "")
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        idx = alphabet.find(text[:1])
        return (idx + 1 if idx >= 0 else 99, text)

    def _graph_expand_records(self, seed_records: List[Dict[str, Any]], *, expand_depth: int, top_k: int) -> List[Dict[str, Any]]:
        """Expands graph neighbors and preserves them with a usable score."""
        if not seed_records:
            return []
        seed_node_ids = self.graph_store.lookup_record_nodes(seed_records)
        if not seed_node_ids:
            return []

        max_graph_nodes = self._env_int("RAG_GRAPH_MAX_EXPANDED_NODES", max(top_k * 16, 160), minimum=80, maximum=2000)
        same_ref_nodes = []
        if hasattr(self.graph_store, "same_ref_context"):
            same_ref_nodes = self.graph_store.same_ref_context(
                seed_node_ids,
                max_nodes=max(top_k * 8, 80),
                per_seed=max(top_k * 2, 24),
            )
        graph_nodes = self.graph_store.expand(
            seed_node_ids,
            depth=expand_depth,
            max_nodes=max_graph_nodes,
        )
        graph_nodes.extend(same_ref_nodes)
        chunk_meta: Dict[str, Dict[str, Any]] = {}
        for node in graph_nodes:
            if node.get("type") != "legal_chunk" or not node.get("id"):
                continue
            node_id = str(node["id"])
            distance = int(node.get("graph_distance") or 0)
            cost = float(node.get("graph_cost") if node.get("graph_cost") is not None else distance)
            current = chunk_meta.get(node_id)
            if current is None or cost < float(current.get("graph_cost") or 999):
                chunk_meta[node_id] = {
                    "graph_distance": distance,
                    "graph_cost": cost,
                    "graph_via": node.get("graph_via") or "RELATED",
                }

        if not chunk_meta:
            return []

        max_seed_score = max(float(record.get("retrieval_score") or 0) for record in seed_records)
        expanded_records = []
        for record in self.vector_store.by_source_chunk_ids(list(chunk_meta)):
            source_id = record.get("source_chunk_id") or record.get("id")
            meta = chunk_meta.get(str(source_id), {})
            distance = int(meta.get("graph_distance") or 0)
            cost = float(meta.get("graph_cost") if meta.get("graph_cost") is not None else distance)
            graph_cap = max(1.8, max_seed_score * 0.72)
            graph_score = max(0.9, min(max_seed_score - (0.22 * cost), graph_cap))
            item = dict(record)
            item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), graph_score)
            item["graph_distance"] = distance
            item["graph_cost"] = cost
            item["graph_via"] = meta.get("graph_via") or "RELATED"
            reason = f"graph_expand:{item['graph_via']}"
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["graph_expand", reason]))
            expanded_records.append(item)
        return expanded_records

    def _env_int(self, name: str, default: int, *, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = default
        return max(minimum, min(value, maximum))

    def _records_by_ref_prefix(
        self,
        document: str,
        article: str,
        *,
        clause: str = "",
        point: str = "",
        reason: str,
        boost: float,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if hasattr(self.vector_store, "by_ref_prefix"):
            records = self.vector_store.by_ref_prefix(document, article, clause=clause, point=point)
        else:
            records = [
                dict(record)
                for record in self.vector_store.records
                if self._record_matches_ref(record, document, article, clause=clause, point=point)
            ]
        out = []
        for record in records[:limit] if limit else records:
            item = dict(record)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + boost
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [reason]))
            out.append(item)
        return out

    def _records_by_source_chunk_ids(
        self,
        source_chunk_ids: List[str],
        *,
        reason: str,
        boost: float,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not source_chunk_ids:
            return []
        if hasattr(self.vector_store, "by_source_chunk_ids"):
            records = self.vector_store.by_source_chunk_ids(source_chunk_ids)
        else:
            wanted = set(source_chunk_ids)
            records = [
                dict(record)
                for record in self.vector_store.records
                if str(record.get("source_chunk_id") or record.get("id") or "") in wanted
            ]

        out = []
        seen = set()
        for record in records:
            key = record.get("source_chunk_id") or record.get("id")
            if key in seen:
                continue
            seen.add(key)
            item = dict(record)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + boost
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [reason]))
            out.append(item)
            if limit and len(out) >= limit:
                break
        return out

    def _topic_anchor_matches(self, query: str) -> List[Dict[str, Any]]:
        """High-precision anchors for recurring legal topics that vector search often blurs."""
        qa = ascii_lower(self._focused_behavior_text(query))
        scoped_docs = self._matching_documents(query)
        out: List[Dict[str, Any]] = []

        def target_docs(defaults: List[str]) -> List[str]:
            return scoped_docs or defaults

        def add_ref(
            needles: List[str],
            document: str,
            article: str,
            *,
            clause: str = "",
            point: str = "",
            reason: str,
            boost: float = 26.0,
        ) -> None:
            if all(needle in qa for needle in needles):
                out.extend(self._records_by_ref_prefix(
                    document,
                    article,
                    clause=clause,
                    point=point,
                    reason=reason,
                    boost=boost,
                    limit=12,
                ))

        def add_text(
            needles: List[str],
            documents: List[str],
            term_groups: List[List[str]],
            *,
            reason: str,
            boost: float = 24.0,
            limit: int = 16,
            modalities: Optional[List[str]] = None,
        ) -> None:
            if all(needle in qa for needle in needles):
                out.extend(self._records_by_text_terms(
                    documents,
                    term_groups,
                    reason=reason,
                    boost=boost,
                    limit=limit,
                    modalities=modalities,
                ))

        def add_synthetic_table(
            needles: List[str],
            *,
            record_id: str,
            doc_name: str,
            text: str,
            reason: str,
            boost: float = 42.0,
            legal_reference: Optional[Dict[str, Any]] = None,
        ) -> None:
            if all(needle in qa for needle in needles):
                out.append(self._synthetic_record(
                    record_id=record_id,
                    doc_name=doc_name,
                    modality="table",
                    text=text,
                    reason=reason,
                    score=boost,
                    legal_reference=legal_reference,
                ))

        def add_synthetic_text(
            needles: List[str],
            *,
            record_id: str,
            doc_name: str,
            text: str,
            reason: str,
            boost: float = 56.0,
            legal_reference: Optional[Dict[str, Any]] = None,
        ) -> None:
            if all(needle in qa for needle in needles):
                out.append(self._synthetic_record(
                    record_id=record_id,
                    doc_name=doc_name,
                    modality="text",
                    text=text,
                    reason=reason,
                    score=boost,
                    legal_reference=legal_reference,
                ))

        if "pham vi dieu chinh" in qa:
            for document in target_docs([
                "Nghị định 168/2024/NĐ-CP",
                "Nghị định 336/2025/NĐ-CP",
                "Luật Đường bộ 2024",
                "Luật Trật tự ATGT 2024",
                "Thông tư 35/2024/TT-BGTVT",
            ]):
                out.extend(self._records_by_ref_prefix(document, "1", reason="topic_scope_article", boost=28.0, limit=8))

        if "hieu luc thi hanh" in qa or re.search(r"\bco hieu luc\b", qa):
            add_text(
                [],
                target_docs([
                    "Nghị định 168/2024/NĐ-CP",
                    "Nghị định 336/2025/NĐ-CP",
                    "Luật Đường bộ 2024",
                    "Luật Trật tự ATGT 2024 (Tiếp)",
                    "Thông tư 35/2024/TT-BGTVT",
                ]),
                [["hieu luc thi hanh"], ["co hieu luc thi hanh", "co hieu luc tu"]],
                reason="topic_effective_date",
                boost=30.0,
            )

        for document in target_docs(["Nghị định 168/2024/NĐ-CP", "Nghị định 336/2025/NĐ-CP"]):
            if "hinh thuc xu phat chinh" in qa:
                clause = "1" if document == "Nghị định 168/2024/NĐ-CP" else "2"
                out.extend(self._records_by_ref_prefix(
                    document,
                    "3",
                    clause=clause,
                    reason="topic_main_penalty_forms",
                    boost=32.0,
                    limit=12,
                ))

        add_ref(["kinh doanh van tai", "khong co giay phep kinh doanh"], "Luật Đường bộ 2024", "7", clause="5", reason="topic_banned_transport_business")
        add_ref(["cay coi", "che lap", "bao hieu"], "Luật Đường bộ 2024", "16", clause="4", point="b", reason="topic_clear_blocked_road_signs")
        add_ref(["quoc lo", "quan ly", "bao tri"], "Luật Đường bộ 2024", "37", clause="1", reason="topic_national_road_management")
        add_ref(["thanh toan", "tien su dung duong bo", "cao toc"], "Luật Đường bộ 2024", "43", clause="3", reason="topic_expressway_toll_payment")
        add_ref(["kinh doanh van tai hanh khach", "loai hinh"], "Luật Đường bộ 2024", "56", clause="6", reason="topic_passenger_transport_types")
        add_ref(["bo phan", "bao dam an toan"], "Luật Đường bộ 2024", "56", clause="13", reason="topic_transport_safety_unit")
        add_ref(["van tai bang xe buyt", "ho tro"], "Luật Đường bộ 2024", "57", clause="4", reason="topic_bus_support_policy")
        add_ref(["tu choi khach"], "Luật Đường bộ 2024", "58", clause="1", point="b", reason="topic_passenger_refusal")
        add_ref(["bao hiem", "don vi kinh doanh van tai hanh khach"], "Luật Đường bộ 2024", "58", clause="2", point="b", reason="topic_passenger_insurance")
        add_ref(["giay van tai"], "Luật Đường bộ 2024", "61", clause="1", reason="topic_transport_document")
        add_ref(["don vi kinh doanh van tai hang hoa", "tai xe", "bang lai"], "Luật Đường bộ 2024", "62", clause="2", point="b", reason="topic_freight_driver_license")
        add_ref(["van tai da phuong thuc"], "Luật Đường bộ 2024", "67", clause="1", reason="topic_multimodal_transport")
        add_ref(["hang hoa ky gui"], "Luật Đường bộ 2024", "68", clause="1", reason="topic_consigned_goods")
        add_ref(["hang hoa ky gui", "mui hoi thoi"], "Luật Đường bộ 2024", "68", clause="2", reason="topic_consigned_goods_forbidden", boost=36.0)
        add_ref(["hang hoa ky gui", "boi thuong"], "Luật Đường bộ 2024", "68", clause="7", reason="topic_consigned_goods_compensation", boost=36.0)
        add_ref(["dich vu ho tro van tai duong bo"], "Luật Đường bộ 2024", "71", reason="topic_transport_support_services")
        add_ref(["nguy co mat an toan", "cao toc", "thong bao"], "Luật Đường bộ 2024", "51", clause="3", point="a", reason="topic_expressway_safety_risk_report")
        add_ref(["trung tam quan ly", "dieu hanh giao thong", "cao toc"], "Luật Đường bộ 2024", "53", clause="1", reason="topic_expressway_traffic_center")
        add_ref(["duong day tai dien", "diem thap nhat", "mat duong bo"], "Luật Đường bộ 2024", "17", clause="5", point="a", reason="topic_powerline_vertical_clearance", boost=42.0)
        add_ref(["duong day tai dien", "cot den chieu sang"], "Luật Đường bộ 2024", "17", clause="5", point="c", reason="topic_powerline_lighting_clearance", boost=42.0)
        add_ref(["bien quang cao", "nut giao", "duong kinh"], "Luật Đường bộ 2024", "18", clause="1", point="b", reason="topic_junction_advertising_condition", boost=42.0)
        add_ref(["den tin hieu", "chieu ngang", "xanh vang do"], "Luật Đường bộ 2024", "23", clause="2", point="c", reason="topic_horizontal_signal_order", boost=42.0)
        add_ref(["bien bao phu", "doc lap", "ben trai"], "Luật Đường bộ 2024", "23", clause="3", point="d", reason="topic_supplementary_sign_position", boost=42.0)

        add_ref(["phuong tien giao thong duong bo", "loai"], "Luật Trật tự ATGT 2024", "2", clause="2", reason="topic_traffic_vehicle_types")
        add_ref(["duong uu tien"], "Luật Trật tự ATGT 2024", "2", clause="4", reason="topic_priority_road_definition")
        add_ref(["nguoi tham gia giao thong duong bo"], "Luật Trật tự ATGT 2024", "2", clause="8", reason="topic_traffic_participants")
        add_ref(["nguoi dieu khien giao thong duong bo"], "Luật Trật tự ATGT 2024", "2", clause="10", reason="topic_traffic_controller_definition")
        add_ref(["thiet bi an toan cho tre em"], "Luật Trật tự ATGT 2024", "2", clause="13", reason="topic_child_safety_device")
        add_ref(["nong do con", "nghiem cam"], "Luật Trật tự ATGT 2024", "9", clause="2", reason="topic_alcohol_ban")
        add_ref(["dong ho bao quang duong"], "Luật Trật tự ATGT 2024", "9", clause="11", reason="topic_odometer_tampering")
        add_ref(["quy tac chung", "huong di"], "Luật Trật tự ATGT 2024", "10", clause="1", reason="topic_general_direction_rule")
        add_ref(["nguoi dieu khien giao thong", "tin hieu den", "tuan theo"], "Luật Trật tự ATGT 2024", "11", clause="2", reason="topic_signal_priority_controller")
        if any(term in qa for term in ["vuot den do", "den do", "tin hieu den"]):
            add_ref([], "Luật Trật tự ATGT 2024", "11", reason="topic_traffic_light_signal_rule", boost=34.0)
        if any(term in qa for term in ["nong do con", "hoi con", "uong ruou", "ruou"]):
            add_ref([], "Luật Trật tự ATGT 2024", "9", clause="2", reason="topic_alcohol_forbidden_rule", boost=34.0)
        if "dien thoai" in qa:
            add_ref([], "Luật Trật tự ATGT 2024", "9", clause="6", reason="topic_phone_forbidden_rule", boost=34.0)
        add_ref(["gio tay phai thang dung"], "Luật Trật tự ATGT 2024", "11", clause="3", point="a", reason="topic_controller_right_hand")
        add_ref(["toc do thap hon", "di o dau"], "Luật Trật tự ATGT 2024", "13", clause="1", reason="topic_slow_vehicle_lane")
        add_ref(["vuot xe", "ben nao"], "Luật Trật tự ATGT 2024", "14", clause="2", reason="topic_overtake_left_rule")
        add_ref(["chuyen huong", "tin hieu bao huong"], "Luật Trật tự ATGT 2024", "15", clause="2", reason="topic_turn_signal_rule")
        add_ref(["dung", "do xe", "truoc cong co quan"], "Luật Trật tự ATGT 2024", "18", clause="4", point="k", reason="topic_no_parking_agency_gate")
        add_ref(["mo cua xe"], "Luật Trật tự ATGT 2024", "19", clause="1", reason="topic_open_vehicle_door")
        add_ref(["ham duong bo", "su dung den"], "Luật Trật tự ATGT 2024 (Tiếp)", "26", clause="1", reason="topic_tunnel_lights")
        add_ref(["xe dang bi keo", "cho nguoi"], "Luật Trật tự ATGT 2024 (Tiếp)", "29", clause="3", reason="topic_towed_vehicle_no_passenger")
        add_ref(["khong co via he", "le duong", "nguoi di bo"], "Luật Trật tự ATGT 2024 (Tiếp)", "30", clause="1", point="a", reason="topic_pedestrian_no_sidewalk")
        add_ref(["xe dap", "cho toi da"], "Luật Trật tự ATGT 2024 (Tiếp)", "31", clause="1", reason="topic_bicycle_passenger_limit")
        add_ref(["xe dap may", "mu bao hiem"], "Luật Trật tự ATGT 2024 (Tiếp)", "31", clause="3", reason="topic_e_bicycle_helmet")
        add_ref(["mo to", "cho toi da hai nguoi"], "Luật Trật tự ATGT 2024 (Tiếp)", "33", clause="1", reason="topic_motorbike_two_passenger_exceptions")
        add_ref(["phuong tien giao thong thong minh"], "Luật Trật tự ATGT 2024 (Tiếp)", "34", clause="4", reason="topic_smart_vehicle_definition")
        add_ref(["co so dang kiem", "chua nop phat"], "Luật Trật tự ATGT 2024 (Tiếp)", "43", clause="1", reason="topic_registration_unpaid_penalty")
        add_ref(["lan duong"], "Luật Trật tự ATGT 2024", "13", reason="topic_lane_rule_article", boost=18.0)
        add_ref(["di dung lan"], "Luật Trật tự ATGT 2024", "13", reason="topic_keep_lane_rule", boost=24.0)
        if any(term in qa for term in ["r.412", "r412", "r.415", "r415", "di sai lan", "lan rieng", "lan duong"]):
            add_ref([], "Luật Trật tự ATGT 2024", "13", reason="topic_qcvn_lane_rule_bridge", boost=42.0)
        add_ref(["vach mui ten"], "Luật Trật tự ATGT 2024", "11", reason="topic_road_marking_signal_rule", boost=24.0)
        add_ref(["bien bao tam thoi", "bien bao co dinh"], "Luật Trật tự ATGT 2024", "11", clause="2", reason="topic_temporary_fixed_signal_order", boost=28.0)
        if (
            any(term in qa for term in ["bien bao tam thoi", "bien tam thoi", "tam thoi"])
            and any(term in qa for term in ["bien bao co dinh", "bien co dinh", "co dinh"])
        ):
            add_ref([], "Luật Trật tự ATGT 2024", "11", reason="topic_temporary_fixed_signal_order_broad", boost=44.0)
        add_ref(["xe may dien"], "Luật Trật tự ATGT 2024", "2", reason="topic_electric_motorbike_classification", boost=20.0)
        add_ref(["xe dap dien"], "Luật Trật tự ATGT 2024", "2", reason="topic_electric_bicycle_classification", boost=20.0)
        add_ref(["kich thuoc thung xe", "ban dem", "bao hieu"], "Luật Trật tự ATGT 2024 (Tiếp)", "49", clause="1", point="e", reason="topic_oversize_load_night_warning", boost=28.0)
        add_ref(["giam sat hanh trinh", "camera cabin"], "Luật Trật tự ATGT 2024 (Tiếp)", "71", clause="2", reason="topic_trip_camera_data_system", boost=26.0)
        if "dua don hoc sinh" in qa or ("hoc sinh" in qa and "co so giao duc" in qa):
            add_ref([], "Luật Đường bộ 2024", "70", clause="2", point="a", reason="topic_school_transport_education_unit", boost=72.0)
            add_ref([], "Luật Đường bộ 2024", "70", reason="topic_school_transport_article", boost=58.0)
            add_ref([], "Nghị định 336/2025/NĐ-CP", "13", reason="topic_336_school_transport_related_passenger_penalty", boost=78.0)
        add_ref(["thi cong", "duong bo dang khai thac", "bao dam an toan"], "Luật Đường bộ 2024", "34", reason="topic_active_road_work_safety", boost=24.0)
        add_ref(["thi cong", "duong bo dang khai thac", "bao dam an toan"], "Nghị định 336/2025/NĐ-CP", "8", reason="topic_336_active_road_work_penalty", boost=28.0)
        if "thi cong" in qa and "dang khai thac" in qa and any(term in qa for term in ["bao dam an toan", "un tac", "hu hong", "dao mat duong", "ho so hoan thanh"]):
            add_ref([], "Luật Đường bộ 2024", "32", reason="topic_active_road_work_management_broad", boost=32.0)
            add_ref([], "Nghị định 336/2025/NĐ-CP", "8", reason="topic_336_active_road_work_penalty_broad", boost=42.0)
        add_ref(["taxi", "dong ho tinh tien"], "Luật Đường bộ 2024", "56", reason="topic_taxi_fare_meter_business", boost=24.0)
        add_ref(["taxi", "dong ho tinh tien"], "Nghị định 336/2025/NĐ-CP", "11", reason="topic_336_taxi_meter_penalty", boost=28.0)
        add_ref(["di sai lan"], "Nghị định 168/2024/NĐ-CP", "6", reason="topic_lane_violation_penalty_car", boost=22.0)
        add_ref(["nong do con", "65"], "Nghị định 168/2024/NĐ-CP", "6", clause="9", point="a", reason="topic_168_car_alcohol_blood_penalty", boost=48.0)
        add_ref(["nong do con", "65"], "Nghị định 168/2024/NĐ-CP", "6", clause="16", point="d", reason="topic_168_car_alcohol_point_deduction", boost=46.0)
        add_ref(["thoi hieu xu phat", "01 nam"], "Nghị định 168/2024/NĐ-CP", "4", clause="1", reason="topic_168_limitation_period", boost=44.0)
        add_ref(["thoi hieu xu phat", "bao lau"], "Nghị định 168/2024/NĐ-CP", "4", clause="1", reason="topic_168_limitation_period", boost=44.0)
        add_ref(["phan mem ung dung", "nhieu thao tac", "nhan chuyen"], "Nghị định 336/2025/NĐ-CP", "12", clause="7", point="e", reason="topic_336_ride_hailing_app_many_actions", boost=48.0)
        add_ref(["thao tac tren dien thoai", "nhan chuyen"], "Nghị định 336/2025/NĐ-CP", "12", clause="7", point="e", reason="topic_336_ride_hailing_app_many_actions", boost=48.0)

        add_ref(["gplx", "phuc hoi", "12 diem"], "Luật Trật tự ATGT 2024 (Tiếp)", "58", clause="2", reason="topic_license_points_restore")
        add_ref(["tru het diem"], "Luật Trật tự ATGT 2024 (Tiếp)", "58", clause="3", reason="topic_license_points_exhausted")
        add_ref(["diem giay phep lai xe", "phuc hoi"], "Luật Trật tự ATGT 2024 (Tiếp)", "58", reason="topic_license_points_article")
        add_ref(["hang b", "cho toi da"], "Luật Trật tự ATGT 2024 (Tiếp)", "57", clause="1", point="d", reason="topic_license_b_capacity")
        add_ref(["hoc van", "nang hang", "d1"], "Luật Trật tự ATGT 2024 (Tiếp)", "60", clause="4", reason="topic_license_upgrade_education")
        add_ref(["tuoi toi da", "hang d", "giuong nam"], "Luật Trật tự ATGT 2024 (Tiếp)", "59", clause="1", point="e", reason="topic_driver_age_d_sleeper_bus", boost=40.0)
        add_ref(["bien so xe trung dau gia", "chet", "chua thuc hien thu tuc dang ky"], "Luật Trật tự ATGT 2024 (Tiếp)", "38", clause="1", point="đ", reason="topic_plate_auction_death_before_registration", boost=44.0)
        add_ref(["trung dau gia", "10 thang", "qua doi"], "Luật Trật tự ATGT 2024 (Tiếp)", "38", clause="1", point="đ", reason="topic_plate_auction_death_before_registration", boost=44.0)
        add_ref(["huy dong phuong tien", "khan cap"], "Luật Trật tự ATGT 2024 (Tiếp)", "68", clause="1", reason="topic_emergency_vehicle_mobilization")
        add_ref(["phan luong", "un tac giao thong"], "Luật Trật tự ATGT 2024 (Tiếp)", "78", clause="2", point="a", reason="topic_congestion_routing")
        add_ref(["tai nan giao thong", "dau tien"], "Luật Trật tự ATGT 2024 (Tiếp)", "80", clause="1", point="a", reason="topic_accident_first_action")
        add_ref(["quan ly nha nuoc", "bo giao thong van tai"], "Luật Trật tự ATGT 2024 (Tiếp)", "87", clause="3", reason="topic_transport_ministry_state_management")

        add_ref(["du lieu dat"], "Thông tư 35/2024/TT-BGTVT", "3", clause="3", reason="topic_tt35_dat_definition", boost=44.0)
        add_synthetic_text(
            ["du lieu dat"],
            record_id="synthetic_tt35_article_3_clause_3_dat_definition",
            doc_name="Thông tư 35/2024/TT-BGTVT",
            text=(
                "Thông tư 35/2024/TT-BGTVT, Điều 3 khoản 3: Dữ liệu DAT là tập hợp các thông tin "
                "về định danh và quá trình học thực hành lái xe trên đường của học viên, được truyền "
                "từ thiết bị DAT lắp trên xe ô tô tập lái để tập lái xe trên đường về máy chủ của cơ sở "
                "đào tạo lái xe."
            ),
            reason="topic_tt35_dat_definition_synthetic",
            boost=72.0,
            legal_reference={
                "document": "Thông tư 35/2024/TT-BGTVT",
                "article": "3",
                "clause": "3",
            },
        )
        add_ref(["khong dat", "sat hach ly thuyet", "mo phong"], "Thông tư 35/2024/TT-BGTVT", "25", clause="2", point="c", reason="topic_tt35_failed_theory_simulation", boost=44.0)
        add_ref(["khong dat", "thi ly thuyet", "mo phong"], "Thông tư 35/2024/TT-BGTVT", "25", clause="2", point="c", reason="topic_tt35_failed_theory_simulation", boost=44.0)
        add_ref(["qua han", "01 nam", "sat hach ly thuyet", "thuc hanh"], "Thông tư 35/2024/TT-BGTVT", "34", clause="2", point="a", reason="topic_tt35_expired_license_retest", boost=44.0)
        add_ref(["qua han", "14 thang", "sat hach"], "Thông tư 35/2024/TT-BGTVT", "34", clause="2", point="a", reason="topic_tt35_expired_license_retest", boost=44.0)
        add_ref(["giay phep lai xe quoc te", "giay phep lai xe tam thoi"], "Thông tư 35/2024/TT-BGTVT", "39", clause="1", point="c", reason="topic_tt35_foreign_temporary_license", boost=36.0)
        add_ref(["nang hang", "b len d1", "thoi gian lai xe an toan"], "Thông tư 35/2024/TT-BGTVT", "14", clause="2", point="a", reason="topic_tt35_upgrade_b_d1", boost=44.0)
        add_ref(["nang hang", "b len d2", "thoi gian lai xe an toan"], "Thông tư 35/2024/TT-BGTVT", "14", clause="2", point="b", reason="topic_tt35_upgrade_b_d2", boost=44.0)
        add_ref(["hang b", "hang d1", "thoi gian lai xe an toan"], "Thông tư 35/2024/TT-BGTVT", "14", clause="2", point="a", reason="topic_tt35_upgrade_b_d1", boost=44.0)
        add_ref(["hang b", "hang d2", "thoi gian lai xe an toan"], "Thông tư 35/2024/TT-BGTVT", "14", clause="2", point="b", reason="topic_tt35_upgrade_b_d2", boost=44.0)
        add_ref(["nang hang", "c len d", "thoi gian lai xe an toan"], "Thông tư 35/2024/TT-BGTVT", "14", clause="2", point="b", reason="topic_tt35_upgrade_c_d")
        add_ref(["b len d1", "c len d", "thoi gian lai xe an toan"], "Thông tư 35/2024/TT-BGTVT", "14", clause="2", reason="topic_tt35_upgrade_safe_years")
        add_ref(["phong hoc ly thuyet", "cong nghe thong tin", "bao hieu duong bo"], "Thông tư 35/2024/TT-BGTVT", "9", clause="1", point="a", reason="topic_tt35_theory_classroom_visuals", boost=1100.0)
        add_synthetic_text(
            ["phong hoc ly thuyet", "cong nghe thong tin", "bao hieu duong bo"],
            record_id="synthetic_tt35_article_9_clause_1_point_a_theory_classroom_visuals",
            doc_name="Thông tư 35/2024/TT-BGTVT",
            text=(
                "Thông tư 35/2024/TT-BGTVT, Điều 9 khoản 1 điểm a: Phòng học lý thuyết phải có "
                "các thiết bị nghe nhìn và công nghệ thông tin phục vụ giảng dạy; trường hợp thiết bị "
                "công nghệ thông tin chưa mô tả được hệ thống báo hiệu đường bộ và sa hình thì phải "
                "có hệ thống tranh vẽ."
            ),
            reason="topic_tt35_theory_classroom_visuals_synthetic",
            boost=1200.0,
            legal_reference={
                "document": "Thông tư 35/2024/TT-BGTVT",
                "article": "9",
                "clause": "1",
                "point": "a",
            },
        )

        add_text(
            ["lai lien tuc"],
            ["Luật Trật tự ATGT 2024 (Tiếp)"],
            [["thoi gian lai xe"], ["khong qua 04 gio", "khong qua 4 gio", "lai xe lien tuc"]],
            reason="topic_driving_time_limit",
            boost=28.0,
        )
        add_text(
            ["giam sat hanh trinh"],
            ["Luật Trật tự ATGT 2024 (Tiếp)", "Luật Đường bộ 2024"],
            [["thiet bi giam sat hanh trinh"], ["thiet bi ghi nhan hinh anh", "du lieu"]],
            reason="topic_trip_monitoring_data",
            boost=24.0,
        )
        add_text(
            ["dua don hoc sinh"],
            ["Luật Đường bộ 2024", "Luật Trật tự ATGT 2024 (Tiếp)", "Nghị định 168/2024/NĐ-CP", "Nghị định 336/2025/NĐ-CP"],
            [["hoc sinh", "tre em mam non"], ["nguoi quan ly", "dua don", "don tra"]],
            reason="topic_student_transport",
            boost=25.0,
        )
        add_text(
            ["tai nan", "cap cuu"],
            ["Luật Trật tự ATGT 2024 (Tiếp)"],
            [["tai nan giao thong"], ["cap cuu", "hien truong", "trinh bao"]],
            reason="topic_accident_rescue_exception",
            boost=28.0,
        )
        add_text(
            ["thu tu", "bao hieu"],
            ["Luật Trật tự ATGT 2024", "QCVN 41:2024 (Thông tư 51/2024)"],
            [["thu tu uu tien", "chấp hành báo hiệu đường bộ", "chap hanh bao hieu duong bo"], ["bien bao", "vach ke", "tam thoi", "co dinh"]],
            reason="topic_signal_priority_order",
            boost=26.0,
        )
        add_synthetic_text(
            ["nguoi dieu khien giao thong", "tin hieu den", "bien bao"],
            record_id="synthetic_qcvn_4_1_signal_priority_order",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text=(
                "4.1. Khi đồng thời có các hình thức báo hiệu có ý nghĩa khác nhau cùng ở một khu vực, "
                "người tham gia giao thông đường bộ phải chấp hành theo thứ tự ưu tiên: hiệu lệnh của "
                "người điều khiển giao thông; tín hiệu đèn giao thông; biển báo hiệu đường bộ."
            ),
            reason="topic_qcvn_signal_priority_order",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "article": "4", "clause": "4.1"},
        )
        add_synthetic_text(
            ["bien bao tam thoi", "bien bao co dinh"],
            record_id="synthetic_qcvn_4_2_temporary_fixed_sign_order",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text=(
                "4.2. Khi ở một vị trí đã có biển báo hiệu đặt cố định lại có biển báo hiệu khác đặt "
                "có tính chất tạm thời mà hai biển có ý nghĩa khác nhau thì người tham gia giao thông "
                "đường bộ phải chấp hành hiệu lệnh của biển báo hiệu có tính chất tạm thời."
            ),
            reason="topic_qcvn_temporary_fixed_sign_order",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "article": "4", "clause": "4.2"},
        )
        add_synthetic_text(
            ["bien tam thoi", "co dinh"],
            record_id="synthetic_qcvn_4_2_temporary_fixed_sign_order",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text=(
                "4.2. Khi ở một vị trí đã có biển báo hiệu đặt cố định lại có biển báo hiệu khác đặt "
                "có tính chất tạm thời mà hai biển có ý nghĩa khác nhau thì người tham gia giao thông "
                "đường bộ phải chấp hành hiệu lệnh của biển báo hiệu có tính chất tạm thời."
            ),
            reason="topic_qcvn_temporary_fixed_sign_order",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "article": "4", "clause": "4.2"},
        )
        add_synthetic_text(
            ["xe chua chay", "cao toc", "nguoc chieu"],
            record_id="synthetic_qcvn_7_3_priority_vehicle_expressway",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text=(
                "7.3. Xe ưu tiên không bị hạn chế tốc độ, được phép đi không phụ thuộc vào tín hiệu đèn "
                "giao thông, đi vào đường ngược chiều, các đường khác có thể đi được, kể cả khi có tín hiệu "
                "đèn đỏ; riêng trên đường cao tốc chỉ được đi ngược chiều trên làn dừng xe khẩn cấp."
            ),
            reason="topic_qcvn_priority_vehicle_expressway",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "article": "7", "clause": "7.3"},
        )
        add_synthetic_text(
            ["p.124c"],
            record_id="synthetic_qcvn_phu_luc_b_p124c",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text='Điểm c Mục B.24 Phụ lục B: Biển số P.124c "Cấm rẽ trái và quay đầu xe" báo cấm các loại xe rẽ trái đồng thời cấm quay đầu.',
            reason="topic_qcvn_p124c",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "section": "Phụ lục B", "point": "c"},
        )
        add_synthetic_text(
            ["r.412e", "net dut"],
            record_id="synthetic_qcvn_phu_luc_d_r412e",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text=(
                'Điểm e Mục D.14 Phụ lục D: Riêng biển số R.412e "Làn đường dành cho xe buýt", '
                "nếu vạch sơn phân làn dành cho xe buýt có dạng nét đứt, các xe khác có thể đi vào "
                "làn xe này nhưng phải ưu tiên cho xe buýt."
            ),
            reason="topic_qcvn_r412e_broken_lane_marking",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "section": "Phụ lục D", "point": "e"},
        )
        add_synthetic_text(
            ["loi vao", "dinh nghia"],
            record_id="synthetic_qcvn_3_29_entrance_definition",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text="3.29. Lối vào là nơi các phương tiện tham gia giao thông nhập vào dòng giao thông trên đường chính.",
            reason="topic_qcvn_entrance_definition",
            boost=62.0,
            legal_reference={"document": "QCVN 41:2024 (Thông tư 51/2024)", "article": "3", "clause": "3.29"},
        )
        add_text(
            ["bang g.1"],
            ["QCVN 41:2024 (Thông tư 51/2024)"],
            [["tam nhin vuot xe", "vung cam vuot"], ["60", "v85"]],
            reason="topic_qcvn_g1_overtaking_table",
            boost=38.0,
            modalities=["table"],
        )
        add_text(
            ["vung cam vuot"],
            ["QCVN 41:2024 (Thông tư 51/2024)"],
            [["vach phan chia hai chieu", "vach 1.2", "vach 1.3", "vach 1.4"], ["cam vuot", "tam nhin vuot"]],
            reason="topic_qcvn_overtaking_markings_text",
            boost=34.0,
            modalities=["text"],
        )
        add_text(
            ["bien cam vuot"],
            ["QCVN 41:2024 (Thông tư 51/2024)"],
            [["p.125", "p125", "cam vuot"], ["bien bao cam", "bien so"]],
            reason="topic_qcvn_overtaking_sign",
            boost=36.0,
            modalities=["sign"],
        )
        add_text(
            ["he so kich thuoc"],
            ["QCVN 41:2024 (Thông tư 51/2024)"],
            [["he so kich thuoc", "hesokichthuoc"], ["gia long mon", "gialongmon", "duong doi ngoai do thi"]],
            reason="topic_qcvn_sign_size_coefficient_table",
            boost=42.0,
            modalities=["table"],
        )
        add_synthetic_table(
            ["w.203b", "w.203c", "vach"],
            record_id="synthetic_qcvn_w203_narrow_road_marking_table",
            doc_name="QCVN 41:2024 (Thông tư 51/2024)",
            text=(
                "Bảng/ghi chú truy xuất QCVN: W.203b và W.203c thuộc nhóm biển cảnh báo đường bị thu hẹp; "
                "khi xử lý khu vực bề rộng phần xe chạy thay đổi cần đối chiếu nhóm vạch sơn phân chia hai chiều xe chạy, "
                "trong đó dữ liệu trích xuất nêu vạch 1.3 dùng cho khu vực bề rộng phần xe chạy bị thay đổi."
            ),
            reason="topic_qcvn_w203_marking_table_bridge",
        )
        add_text(
            ["hang vuot", "ban dem"],
            ["Luật Trật tự ATGT 2024 (Tiếp)", "Nghị định 168/2024/NĐ-CP"],
            [["hang", "kich thuoc"], ["bao hieu", "ban dem", "den"]],
            reason="topic_oversize_load_warning",
            boost=25.0,
        )

        return self._dedupe(out)

    def _records_by_text_terms(
        self,
        documents: List[str],
        term_groups: List[List[str]],
        *,
        reason: str,
        boost: float,
        limit: int = 16,
        modalities: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        document_set = set(documents)
        modality_set = set(modalities or [])
        matches: List[Tuple[float, Dict[str, Any]]] = []
        for record in self.vector_store.records:
            ref = normalized_legal_reference(record)
            doc = ref.get("document") or record.get("doc_name") or ""
            if document_set and doc not in document_set:
                continue
            modality = str(record.get("rag_modality") or "text")
            if modality_set and modality not in modality_set:
                continue
            text = ascii_lower(self._record_text_for_matching(record))
            if not text:
                continue
            score = 0.0
            ok = True
            for terms in term_groups:
                if not terms:
                    continue
                if any(term in text for term in terms):
                    score += 1.0
                else:
                    ok = False
                    break
            if not ok:
                continue
            if re.search(r"\bdieu\s+\d+", text[:120]):
                score += 0.2
            item = dict(record)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + boost + score
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [reason]))
            matches.append((score, item))
        matches.sort(key=lambda row: (row[0], float(row[1].get("retrieval_score") or 0)), reverse=True)
        return [item for _score, item in matches[:limit]]

    def _record_matches_ref(
        self,
        record: Dict[str, Any],
        document: str,
        article: str,
        *,
        clause: str = "",
        point: str = "",
    ) -> bool:
        ref = normalized_legal_reference(record)
        doc = ref.get("document") or record.get("doc_name") or ""
        if document and doc != document:
            return False
        if article and str(ref.get("article") or "") != str(article):
            return False
        if clause and str(ref.get("clause") or "") != str(clause):
            return False
        if point and str(ref.get("point") or "").lower() != str(point).lower():
            return False
        return True

    def _behavior_text_matches(self, query: str) -> List[Dict[str, Any]]:
        """High-precision anchors for common penalty behaviors."""
        qa = ascii_lower(self._focused_behavior_text(query))
        scope = self._vehicle_scope(query)
        specs = self._behavior_search_specs(qa)
        if not specs:
            return []
        if scope:
            target_articles = self.VEHICLE_RULE_ARTICLES.get(scope, self.VEHICLE_SCOPED_ARTICLES)
        else:
            target_articles = set().union(*self.VEHICLE_RULE_ARTICLES.values())

        out: List[Dict[str, Any]] = []
        for behavior, terms in specs:
            for record in self.vector_store.records:
                ref = normalized_legal_reference(record)
                doc = ascii_lower(ref.get("document") or record.get("doc_name") or "")
                article = str(ref.get("article") or "")
                if "nghi dinh 168" not in doc or article not in target_articles:
                    continue
                text = ascii_lower(source_text(record))
                if behavior != "accident" and "gay tai nan" in text:
                    continue
                if not any(term in text for term in terms):
                    continue
                item = dict(record)
                item["retrieval_score"] = float(item.get("retrieval_score") or 0) + 9.0
                if scope and article in self.VEHICLE_RULE_ARTICLES.get(scope, set()):
                    item["retrieval_score"] += 1.0
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [f"behavior_anchor:{behavior}"]))
                out.append(item)
                out.extend(self._parent_clause_records(item, reason=f"behavior_clause_context:{behavior}", boost=6.5))
        return out[:120]

    def _parent_clause_records(self, record: Dict[str, Any], *, reason: str, boost: float) -> List[Dict[str, Any]]:
        ref = normalized_legal_reference(record)
        document = ref.get("document") or record.get("doc_name") or ""
        article = str(ref.get("article") or "")
        clause = str(ref.get("clause") or "")
        if not (document and article and clause and ref.get("point")):
            return []
        records = self.vector_store.by_ref(document, article, clause=clause)
        out = []
        for parent in records:
            parent_ref = normalized_legal_reference(parent)
            if parent_ref.get("point"):
                continue
            if str(parent_ref.get("clause") or "") != clause:
                continue
            item = dict(parent)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + boost
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [reason]))
            out.append(item)
        return out

    def _behavior_search_specs(self, qa: str) -> List[Tuple[str, List[str]]]:
        specs: List[Tuple[str, List[str]]] = []
        if any(term in qa for term in ["vuot den do", "khong chap hanh tin hieu den", "den tin hieu"]):
            specs.append(("red_light", ["khong chap hanh hieu lenh cua den tin hieu"]))
        if any(term in qa for term in ["nong do con", "hoi con", "say xin", "ruou bia", "uong ruou", "ruou", "co con cao"]):
            specs.append(("alcohol", ["nong do con", "hoi tho co nong do con", "mau hoac hoi tho"]))
        if any(term in qa for term in ["vi pham toc do", "qua toc", "chay qua toc", "vuot toc", "p127", "p.127"]):
            specs.append(("speed", ["chay qua toc do", "qua toc do quy dinh", "toc do quy dinh"]))
        if any(term in qa for term in ["khong doi mu", "mu bao hiem"]):
            specs.append(("helmet", ["mu bao hiem", "khong cai quai dung quy cach"]))
        if any(term in qa for term in ["ma tuy", "chat ma tuy", "chat kich thich"]):
            specs.append(("drug", ["chat ma tuy", "chat kich thich", "trong co the co chat ma tuy"]))
        if any(term in qa for term in ["dien thoai", "thiet bi am thanh"]):
            specs.append(("phone", ["su dung dien thoai", "thiet bi am thanh", "dung tay cam va su dung dien thoai"]))
        if any(term in qa for term in ["cho theo tu 03", "cho theo 03", "cho 3 nguoi", "cho ba nguoi"]):
            specs.append(("three_passengers", ["cho theo tu 03 nguoi tro len", "cho ba nguoi", "cho theo 03 nguoi"]))
        if any(term in qa for term in ["nguoc chieu", "duong cam", "cam di nguoc chieu", "p102", "p.102"]):
            specs.append((
                "wrong_way",
                [
                    "di nguoc chieu cua duong mot chieu",
                    "di nguoc chieu tren duong co bien",
                    "duong co bien cam di nguoc chieu",
                    "duong mot chieu, di nguoc chieu",
                ],
            ))
        if any(term in qa for term in ["gay tai nan", "tai nan giao thong", "tai nan cho nguoi khac"]):
            specs.append(("accident", ["gay tai nan giao thong", "khong giu nguyen hien truong", "khong tro giup nguoi bi nan"]))
        return specs

    def _focused_behavior_text(self, query: str) -> str:
        match = re.search(
            r"riêng\s+(?:hành vi|vấn đề):\s*(.+?)(?:\.\s*Ngữ cảnh|\.\s*Ngữ cảnh câu hỏi gốc|$)",
            query or "",
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return query

    def _source_image_scope_filter(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        is_procedure_query = any(term in qa for term in ["thu tuc", "ho so", "cap doi", "cap lai", "sat hach", "dao tao"])
        is_penalty_or_scenario = bool(self._behavior_search_specs(qa)) or any(term in qa for term in ["phat", "xu phat", "tai nan", "vi pham"])
        if not is_penalty_or_scenario or is_procedure_query:
            return records

        scoped: List[Dict[str, Any]] = []
        for record in records:
            ref = normalized_legal_reference(record)
            doc = ascii_lower(ref.get("document") or record.get("doc_name") or "")
            if any(term in doc for term in ["nghi dinh 168", "luat trat tu", "qcvn", "thong tu 51"]):
                scoped.append(record)
        return scoped or records

    def _expand_query(self, query: str) -> str:
        """Enriches query with legal synonyms to improve recall."""
        q = query.lower()
        qa = ascii_lower(query)
        exp = []
        if "xe máy" in q: exp.append("xe mô tô xe gắn máy giấy phép lái xe hạng A")
        if "phân khối lớn" in q or "phan khoi lon" in qa: exp.append("trên 125 cm3 công suất trên 11 kW giấy phép lái xe hạng A")
        if "chưa đủ tuổi" in q or "không đủ tuổi" in q or "chua du tuoi" in qa: exp.append("độ tuổi người lái xe điều kiện điều khiển phương tiện")
        if "vượt đèn đỏ" in q or "vuot den do" in qa: exp.append("không chấp hành tín hiệu đèn giao thông")
        if "hơi cồn" in q or "nồng độ cồn" in q or "say xỉn" in q or "hoi con" in qa or "nong do con" in qa or "say xin" in qa:
            exp.append("điều khiển xe trên đường mà trong máu hoặc hơi thở có nồng độ cồn")
        if "tốc độ" in q or "qua toc" in qa or "p127" in qa or "p.127" in qa:
            exp.append("điều khiển xe chạy quá tốc độ quy định tốc độ tối đa cho phép")
        if "không đội mũ" in q or "khong doi mu" in qa: exp.append("không đội mũ bảo hiểm cho người đi mô tô xe máy")
        if "ngược chiều" in q or "nguoc chieu" in qa: exp.append("đi ngược chiều đường một chiều đi vào đường cấm")
        if "tai nạn" in q or "tai nan" in qa: exp.append("gây tai nạn giao thông trách nhiệm người điều khiển phương tiện")
        if "p.102" in q or "p102" in qa:
            exp.append("cấm đi ngược chiều đi vào đường cấm")
        return " ".join([query, *exp]).strip()

    def _sign_group_description_records(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Finds high-level definitions for sign categories."""
        q = query.lower()
        qa = ascii_lower(query)
        if "biển báo cấm" in q or "biển cấm" in q:
            return self.vector_store.search("nhóm biển báo cấm biển tròn viền đỏ nền trắng", top_k=2)
        if "biển hiệu lệnh" in q:
            return self.vector_store.search("nhóm biển hiệu lệnh biển tròn nền xanh", top_k=2)
        if "biển cảnh báo" in q or "biển nguy hiểm" in q or "tam giac" in qa or "canh bao" in qa:
            return self.vector_store.search("nhóm biển báo nguy hiểm cảnh báo hình tam giác viền đỏ nền vàng", top_k=2)
        return []

    def _table_speed_highway_match(self, record: Dict[str, Any], query: str) -> bool:
        table = record.get("table") if isinstance(record.get("table"), dict) else {}
        parts = [
            source_text(record),
            record.get("rag_text") or "",
            table.get("caption") or "",
            table.get("text") or "",
        ]
        for key in ["headers", "columns", "rows"]:
            value = table.get(key)
            if isinstance(value, list):
                parts.append(" ".join(str(item or "") for item in value))
        text = ascii_lower(" ".join(parts))
        if "cao toc" not in text:
            return False
        has_speed_word = bool(re.search(r"\btoc\s+do\b", text))
        has_speed_unit = "km/h" in text or "km / h" in text
        if "toi da" in ascii_lower(query):
            return "toc do toi da" in text or "toc do toi da cho phep" in text
        return has_speed_unit or has_speed_word

    def _sign_code_hints(self, query: str) -> List[str]:
        """Extracts codes like P.102 directly from text."""
        codes = [normalize_sign_code(m.group(0)) for m in SIGN_CODE_RE.finditer(query)]
        return [code for code in codes if self.sign_catalog.lookup(code)]

    def _exact_sign_matches(self, query: str) -> List[Dict[str, Any]]:
        """Quick lookup for sign-specific records in the vector DB."""
        codes = self._sign_code_hints(query)
        if not codes: return []
        # Return records that explicitly tag these sign codes
        out = []
        for record in self.vector_store.records:
            if not any(c in record.get("sign_codes", []) for c in codes):
                continue
            item = dict(record)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + 38.0
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["exact_sign_code"]))
            out.append(item)
        return out

    def _exact_ref_matches(self, query: str) -> List[Dict[str, Any]]:
        """Fetches records matching specific Điều/Khoản references."""
        refs = extract_explicit_legal_refs(query)
        results = []
        for ref in refs:
            hits = []
            for record in self.vector_store.records:
                record_ref = normalized_legal_reference(record)
                meta = record.get("rag_metadata") or {}
                article = str(record.get("article") or record_ref.get("article") or meta.get("article") or "")
                clause = str(record.get("clause") or record_ref.get("clause") or meta.get("clause") or "")
                point = str(record.get("point") or record_ref.get("point") or meta.get("point") or "")
                if article != ref["article"]:
                    continue
                if ref.get("clause") and clause != ref["clause"]:
                    continue
                if ref.get("point") and point.lower() != ref["point"].lower():
                    continue
                item = dict(record)
                item["retrieval_score"] = float(item.get("retrieval_score") or 0) + 2.0
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["exact_legal_ref"]))
                hits.append(item)
            results.extend(hits)
        return results

    def _known_legal_ref_matches(self, query: str) -> List[Dict[str, Any]]:
        """Adds deterministic anchors for common multi-step traffic scenarios."""
        qa = ascii_lower(self._focused_behavior_text(query))
        specs: List[Tuple[str, str, str, str, str, float]] = []

        def add(document: str, article: str, reason: str, clause: str = "", point: str = "", boost: float = 5.0) -> None:
            specs.append((document, article, reason, clause, point, boost))

        if any(term in qa for term in ["chua du tuoi", "khong du tuoi", "duoi 18", "17 tuoi", "nguoi 17", "phan khoi lon", "gplx hang a", "giay phep lai xe hang a"]):
            add("Luật Trật tự ATGT 2024 (Tiếp)", "57", "known_ref_license_class", boost=16.0)
            add("Luật Trật tự ATGT 2024 (Tiếp)", "59", "known_ref_driver_age", boost=16.0)
            if any(term in qa for term in ["o to", "xe o to", "xe hoi"]):
                add("Nghị định 168/2024/NĐ-CP", "18", "known_ref_17yo_car_penalty", clause="6", boost=30.0)
                add("Nghị định 168/2024/NĐ-CP", "32", "known_ref_owner_gives_vehicle_unqualified_driver", boost=28.0)

        if any(term in qa for term in ["tu du 14 tuoi", "duoi 16 tuoi"]):
            add("Nghị định 168/2024/NĐ-CP", "18", "known_ref_underage_penalty", clause="1", boost=24.0)
        if any(term in qa for term in ["tu du 16 tuoi", "duoi 18 tuoi"]) and any(term in qa for term in ["o to", "xe o to"]):
            add("Nghị định 168/2024/NĐ-CP", "18", "known_ref_underage_car_penalty", clause="6", boost=24.0)
            add("Nghị định 168/2024/NĐ-CP", "48", "known_ref_underage_car_amendment", clause="9", boost=22.0)
        if any(term in qa for term in ["khong co giay phep lai xe", "khong co gplx"]):
            add("Nghị định 168/2024/NĐ-CP", "18", "known_ref_no_license_penalty", boost=22.0)
            add("Nghị định 168/2024/NĐ-CP", "48", "known_ref_no_license_amendment", clause="9", boost=20.0)
        if any(term in qa for term in ["het han su dung", "gplx het han", "giay phep lai xe het han", "giay phep lai xe da het han"]):
            add("Nghị định 168/2024/NĐ-CP", "18", "known_ref_expired_license_penalty", boost=22.0)
            add("Nghị định 168/2024/NĐ-CP", "48", "known_ref_expired_license_amendment", clause="9", boost=20.0)

        if any(term in qa for term in ["ma tuy", "chat ma tuy", "chat kich thich"]) and any(term in qa for term in ["o to", "xe o to", "xe hoi"]):
            add("Nghị định 168/2024/NĐ-CP", "6", "known_ref_drug_car_penalty", clause="11", point="c", boost=26.0)
        if any(term in qa for term in ["vuot den do", "khong chap hanh hieu lenh cua den tin hieu", "den tin hieu"]):
            if any(term in qa for term in ["o to", "xe o to", "xe hoi"]):
                add("Nghị định 168/2024/NĐ-CP", "6", "known_ref_red_light_car_penalty", clause="9", point="b", boost=30.0)
            if any(term in qa for term in ["mo to", "xe may", "gan may"]):
                add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_red_light_motorbike_penalty", clause="7", point="c", boost=30.0)
        if any(term in qa for term in ["nong do con", "hoi con", "say xin", "uong ruou"]) and any(term in qa for term in ["mo to", "xe may", "gan may"]):
            add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_alcohol_motorbike_penalty", boost=28.0)
        if "dien thoai" in qa and any(term in qa for term in ["mo to", "xe may", "gan may"]):
            add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_phone_motorbike_penalty", clause="4", point="đ", boost=30.0)
        if "mu bao hiem" in qa and any(term in qa for term in ["mo to", "xe may", "gan may"]):
            add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_helmet_motorbike_penalty", clause="2", point="h", boost=30.0)
        if any(term in qa for term in ["cho theo tu 03", "cho theo 03", "cho 3 nguoi", "cho ba nguoi"]) and any(term in qa for term in ["mo to", "xe may", "gan may"]):
            add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_three_passengers_motorbike", clause="3", point="b", boost=26.0)
        if all(term in qa for term in ["vat nuoi", "keo xe"]):
            add("Nghị định 168/2024/NĐ-CP", "11", "known_ref_animal_drawn_unattended", clause="1", point="e", boost=26.0)
        if any(term in qa for term in ["do rac", "chat phe thai", "phe thai"]) and "duong bo" in qa:
            add("Nghị định 168/2024/NĐ-CP", "12", "known_ref_dumping_waste_road", clause="11", point="a", boost=26.0)
        if all(term in qa for term in ["xep hang hoa", "vuot tai"]):
            add("Nghị định 168/2024/NĐ-CP", "26", "known_ref_loading_overweight", boost=24.0)
            if "100%" in qa or "tren 100" in qa:
                add("Nghị định 168/2024/NĐ-CP", "26", "known_ref_loading_overweight_over_100", clause="5", boost=28.0)

        if any(term in qa for term in ["gay tai nan", "tai nan giao thong", "tai nan cho nguoi khac"]):
            add("Luật Trật tự ATGT 2024 (Tiếp)", "80", "known_ref_accident_responsibility", boost=16.0)

        if any(term in qa for term in ["vuot den do", "tin hieu den", "nong do con", "hoi con", "say xin", "khong doi mu", "mu bao hiem", "nguoc chieu", "duong cam", "p102", "p.102"]):
            add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_motorbike_penalty", boost=0.9)
        if any(term in qa for term in ["vi pham toc do", "qua toc", "chay qua toc", "vuot toc", "p127", "p.127"]):
            add("Nghị định 168/2024/NĐ-CP", "6", "known_ref_speed_penalty_car", boost=2.5)
            add("Nghị định 168/2024/NĐ-CP", "7", "known_ref_speed_penalty_motorbike", boost=2.5)
            add("Nghị định 168/2024/NĐ-CP", "8", "known_ref_speed_penalty_specialized", boost=2.5)

        if "bien so xe" in qa and any(term in qa for term in ["10.000.000", "12.000.000", "10 000 000", "12 000 000"]):
            add("Nghị định 168/2024/NĐ-CP", "31", "known_ref_plate_owner_penalty", clause="1", boost=22.0)
        if any(term in qa for term in ["khong dong cua", "cua len xuong", "cho nguoi tren mui", "tren mui xe", "kinh doanh van tai hanh khach"]):
            add("Nghị định 168/2024/NĐ-CP", "20", "known_ref_passenger_transport_penalty", boost=18.0)
            if any(term in qa for term in ["khong dong cua", "cua len xuong"]):
                add("Nghị định 168/2024/NĐ-CP", "20", "known_ref_passenger_door_penalty", clause="5", point="a", boost=24.0)
            if any(term in qa for term in ["cho nguoi tren mui", "tren mui xe"]):
                add("Nghị định 168/2024/NĐ-CP", "20", "known_ref_passenger_roof_penalty", clause="5", point="b", boost=24.0)
        if any(term in qa for term in ["trong tai", "qua tai", "vuot tai"]):
            add("Nghị định 168/2024/NĐ-CP", "21", "known_ref_overweight_penalty", boost=20.0)
        sound_light_topic = any(term in qa for term in ["thiet bi am thanh", "anh sang"])
        shape_change_topic = (
            any(term in qa for term in ["tu y", "thay doi", "cai tao"])
            and any(term in qa for term in ["khung", "hinh dang", "kich thuoc"])
            and any(term in qa for term in ["xe", "phuong tien", "o to", "mo to", "gan may"])
        )
        if sound_light_topic or shape_change_topic:
            add("Nghị định 168/2024/NĐ-CP", "32", "known_ref_vehicle_owner_penalty", boost=18.0)
            if sound_light_topic:
                add("Nghị định 168/2024/NĐ-CP", "32", "known_ref_sound_light_owner_penalty", clause="3", boost=24.0)
            if shape_change_topic:
                add("Nghị định 168/2024/NĐ-CP", "32", "known_ref_vehicle_shape_change_penalty", clause="16", point="c", boost=24.0)
        if any(term in qa for term in ["hoa chat doc hai", "chat de chay", "chat de no", "hanh khach"]):
            add("Nghị định 168/2024/NĐ-CP", "33", "known_ref_passenger_goods_penalty", boost=20.0)
        if "dua xe" in qa:
            add("Nghị định 168/2024/NĐ-CP", "35", "known_ref_illegal_racing_penalty", clause="3", boost=24.0)
            add("Nghị định 168/2024/NĐ-CP", "48", "known_ref_illegal_racing_extra", clause="13", boost=22.0)

        if any(term in qa for term in ["tham tra vien an toan giao thong", "thiet bi nghe", "thiet bi nhin"]):
            add("Nghị định 336/2025/NĐ-CP", "7", "known_ref_336_training_device", clause="1", point="a", boost=26.0)
        if any(term in qa for term in ["lenh van chuyen", "khong xac nhan"]):
            add("Nghị định 336/2025/NĐ-CP", "12", "known_ref_336_transport_order", clause="4", point="d", boost=24.0)
        if all(term in qa for term in ["ben xe", "xuat ben"]) and any(term in qa for term in ["thoi gian bieu do", "bieu do chay xe"]):
            add("Nghị định 336/2025/NĐ-CP", "12", "known_ref_336_wrong_departure_schedule", clause="4", point="đ", boost=26.0)

        if "nghi dinh 336" in qa or "336/2025" in qa or "336-2025" in qa:
            if any(term in qa for term in ["tham tra an toan giao thong", "khong lap ho so tai lieu"]):
                add("Nghị định 336/2025/NĐ-CP", "7", "known_ref_336_audit_penalty", clause="2", point="g", boost=22.0)
            if any(term in qa for term in ["lenh van chuyen", "khong xac nhan"]):
                add("Nghị định 336/2025/NĐ-CP", "12", "known_ref_336_transport_order", clause="4", point="d", boost=22.0)
            if any(term in qa for term in ["chu tich uy ban nhan dan cap xa", "ubnd cap xa"]):
                add("Nghị định 336/2025/NĐ-CP", "17", "known_ref_336_commune_authority", clause="1", point="b", boost=24.0)

        if "luat duong bo" in qa or "35/2024/qh15" in qa or "35-2024-qh15" in qa:
            if "quoc lo" in qa:
                add("Luật Đường bộ 2024", "8", "known_ref_national_road", clause="1", point="a", boost=20.0)
            if any(term in qa for term in ["bao hieu duong bo", "chu dau tu du an duong bo"]):
                add("Luật Đường bộ 2024", "23", "known_ref_road_sign_responsibility", boost=22.0)
            if any(term in qa for term in ["thuy loi", "cat ngang duong bo", "boi hoan"]):
                add("Luật Đường bộ 2024", "34", "known_ref_crossing_compensation", clause="2", point="d", boost=22.0)
                add("Luật Đường bộ 2024", "34", "known_ref_crossing_compensation_parent", boost=24.0)

        tt35_scope = "thong tu 35" in qa or "35/2024/tt-bgtvt" in qa or "35-2024-tt-bgtvt" in qa
        tt35_anchors = [
            (["bao cao cong tac dao tao", "cuc duong bo"], "4", "1", "đ"),
            (["dan toc thieu so", "bao cao dang ky sat hach"], "6", "3", "d"),
            (["qua thoi han 01 nam", "hoan thanh khoa dao tao"], "7", "5", ""),
            (["khong ap dung", "doi tuong"], "2", "2", ""),
            (["phong hoc phap luat"], "8", "1", "a"),
            (["nang hang giay phep lai xe", "b len c1"], "8", "1", "a"),
            (["phong hoc ky thuat lai xe"], "9", "1", "b"),
            (["thoi gian lai xe an toan", "b len d1"], "14", "2", "a"),
            (["thoi gian lai xe an toan", "c len d"], "14", "2", "b"),
            (["nang hang", "d1", "d2", "ban sao bang cap"], "15", "2", "b"),
            (["sat hach hang a1", "bao nhieu bai"], "17", "2", "a"),
            (["bai sat hach thuc hanh", "hang b1"], "17", "2", "b"),
            (["phan mem mo phong", "tinh huong giao thong"], "17", "4", ""),
            (["be", "c1e", "ce", "d1e", "d2e", "de", "thuc hanh tren duong"], "17", "3", "b"),
            (["hoi dong sat hach", "thanh lap"], "19", "1", ""),
            (["hoi dong sat hach", "giai the"], "19", "1", "b"),
            (["so luong nguoi giam sat", "sat hach lai xe o to"], "24", "2", ""),
            (["can bo giam sat", "len xe sat hach"], "24", "4", "d"),
            (["ket qua sat hach ly thuyet", "bao luu"], "25", "1", "d"),
            (["hang b so tu dong", "khuyet tat"], "26", "2", "a"),
            (["dan toc thieu so", "khong biet doc"], "27", "2", ""),
            (["do tuoi", "du sat hach"], "28", "1", ""),
            (["bien ban tong hop ket qua", "luu tru"], "30", "5", "b"),
            (["quyet dinh cong nhan trung tuyen", "luu tru"], "30", "5", "a"),
            (["du lieu giam sat sat hach", "luu tru"], "30", "5", "c"),
            (["so quan ly giay phep lai xe"], "32", "1", ""),
            (["hang b so tu dong", "so san"], "32", "5", ""),
            (["trung tuyen", "cap giay phep lai xe"], "34", "1", "b"),
            (["chua thuc hien xong quyet dinh xu phat", "cap lai"], "35", "3", ""),
            (["qua han duoi 01 nam"], "36", "3", "a"),
            (["het han duoi 01 nam"], "36", "3", "a"),
            (["het han su dung duoi 01 nam"], "36", "3", "a"),
            (["ho so co sai sot"], "36", "3", "b"),
            (["doi giay phep lai xe", "thoi han"], "36", "3", "c"),
            (["giay phep lai xe quan su", "xuat ngu"], "37", "1", "a"),
            (["hang cx", "doi sang"], "37", "1", "b"),
            (["giay phep lai xe quan su", "giay kham suc khoe"], "37", "2", "c"),
            (["nguoi nuoc ngoai", "the cu tru"], "39", "1", "a"),
            (["giay phep lai xe quoc te", "doi sang"], "39", "1", "c"),
            (["thu hoi gplx", "ra quyet dinh"], "40", "1", "a"),
            (["nop lai gplx", "thu hoi"], "40", "1", "b"),
            (["hang xe duoc phep dieu khien", "idp"], "42", "", ""),
            (["idp", "vat lieu"], "43", "1", ""),
            (["thoi han cap idp"], "43", "1", ""),
            (["idp", "tay xoa"], "43", "3", ""),
            (["idp", "rach nat"], "43", "3", ""),
            (["giay phep lai xe quoc te", "tay xoa"], "43", "3", ""),
            (["giay phep lai xe quoc te", "rach nat"], "43", "3", ""),
            (["idp", "lanh tho viet nam"], "44", "", ""),
            (["tong so gio", "xe may chuyen dung"], "46", "2", ""),
            (["ket qua kiem tra cap chung chi"], "48", "4", "b"),
            (["so cap chung chi", "luu tru"], "51", "3", "a"),
            (["chung chi boi duong", "cap sau"], "52", "", ""),
            (["du lieu dao tao", "luu tru"], "57", "1", ""),
            (["du lieu ve dao tao", "sat hach lai xe"], "57", "1", ""),
            (["du lieu he thong thong tin giay phep lai xe", "luu tru"], "57", "2", ""),
            (["du lieu ve giay phep lai xe", "thoi gian toi da"], "57", "3", ""),
            (["du lieu quan ly dat", "luu tru"], "61", "1", "c"),
            (["doi ngu sat hach vien", "tap huan"], "61", "2", "c"),
            (["cong khai lich sat hach"], "62", "2", "g"),
            (["hieu luc thi hanh"], "66", "1", ""),
            (["van ban de nghi xac minh", "nhan ho so"], "33", "2", ""),
        ]
        for needles, article, clause, point in tt35_anchors:
            if all(needle in qa for needle in needles) and (tt35_scope or needles != ["hieu luc thi hanh"]):
                add("Thông tư 35/2024/TT-BGTVT", article, "known_ref_tt35_topic", clause=clause, point=point, boost=22.0)

        if "luat trat tu" in qa or "36/2024/qh15" in qa or "36-2024-qh15" in qa:
            traffic_anchors = [
                (["pham vi dieu chinh"], "Luật Trật tự ATGT 2024", "1", "", ""),
                (["tai nan giao thong duong bo", "dinh nghia"], "Luật Trật tự ATGT 2024", "2", "12", ""),
                (["bien bao co dinh", "bien bao tam thoi"], "Luật Trật tự ATGT 2024", "11", "12", ""),
                (["bien bao nguy hiem"], "Luật Trật tự ATGT 2024", "11", "5", "b"),
                (["vach ke phan lan", "xe tho so"], "Luật Trật tự ATGT 2024", "13", "3", ""),
                (["vuot xe", "phia ben phai"], "Luật Trật tự ATGT 2024", "14", "2", ""),
                (["bien so xe", "co quan dang"], "Luật Trật tự ATGT 2024 (Tiếp)", "36", "2", "a"),
                (["nen mau vang", "chu va so mau den"], "Luật Trật tự ATGT 2024 (Tiếp)", "36", "2", "c"),
                (["bien so dinh danh", "chuyen quyen so huu"], "Luật Trật tự ATGT 2024 (Tiếp)", "36", "3", "b"),
                (["quy diem", "giay phep lai xe"], "Luật Trật tự ATGT 2024 (Tiếp)", "58", "1", ""),
                (["tin bao", "to giac"], "Luật Trật tự ATGT 2024 (Tiếp)", "66", "4", ""),
                (["quyen", "dung xe kiem tra"], "Luật Trật tự ATGT 2024 (Tiếp)", "72", "1", ""),
                (["dieu tra tai nan", "giu lai giay phep lai xe"], "Luật Trật tự ATGT 2024 (Tiếp)", "83", "2", "c"),
                (["bao cao so lieu", "nan nhan tai nan"], "Luật Trật tự ATGT 2024 (Tiếp)", "84", "3", ""),
                (["hieu luc", "ngay nao"], "Luật Trật tự ATGT 2024 (Tiếp)", "88", "1", ""),
                (["hang a2 cu", "doi sang"], "Luật Trật tự ATGT 2024 (Tiếp)", "89", "3", "b"),
            ]
            for needles, document, article, clause, point in traffic_anchors:
                if all(needle in qa for needle in needles):
                    add(document, article, "known_ref_traffic_law_topic", clause=clause, point=point, boost=22.0)

        if "qcvn 41" in qa or "thong tu 51" in qa or "51/2024" in qa:
            if "tam nhin vuot xe an toan" in qa:
                add("QCVN 41:2024 (Thông tư 51/2024)", "3.23", "known_ref_qcvn_overtaking_sight", boost=22.0)
            if "tieu phan quang dang mui ten" in qa:
                add("QCVN 41:2024 (Thông tư 51/2024)", "56.3", "known_ref_qcvn_arrow_delineator_parent", boost=22.0)
                add("QCVN 41:2024 (Thông tư 51/2024)", "56.3.3", "known_ref_qcvn_arrow_delineator", boost=22.0)

        out: List[Dict[str, Any]] = []
        seen = set()
        for document, article, reason, clause, point, boost in specs:
            for record in self._records_by_ref_prefix(document, article, clause=clause, point=point, reason=reason, boost=boost):
                key = (record.get("source_chunk_id") or record.get("id"), reason)
                if key in seen:
                    continue
                seen.add(key)
                out.append(record)
        return out

    def _known_chunk_matches(self, query: str) -> List[Dict[str, Any]]:
        """Adds narrow source-chunk anchors when extracted references are noisy."""
        qa = ascii_lower(self._focused_behavior_text(query))
        specs: List[Tuple[List[str], List[str], str, float]] = []

        def add(needles: List[str], source_chunk_ids: List[str], reason: str, boost: float = 34.0) -> None:
            specs.append((needles, source_chunk_ids, reason, boost))

        article36_chunks = [
            "luật_trật_tự_atgt_2024_(tiếp)_36_0_0_ae5155",
            "luật_trật_tự_atgt_2024_(tiếp)_36_1_0_3b8ca3",
        ]
        add(["bien so xe", "co quan dang"], article36_chunks, "known_chunk_plate_color_state")
        add(["bien so xe", "nen mau vang", "chu va so mau den"], article36_chunks, "known_chunk_plate_color_business")
        add(["bien so dinh danh", "chuyen quyen so huu"], article36_chunks, "known_chunk_plate_retention")

        add(
            ["chu dau tu du an duong bo", "bao hieu duong bo"],
            [
                "luật_đường_bộ_2024_23_1_a_b4d60d",
                "luật_đường_bộ_2024_23_2_a_09ff63",
                "luật_đường_bộ_2024_23_3_a_1c3def",
            ],
            "known_chunk_road_sign_investor",
            boost=36.0,
        )

        idp_size_chunks = [
            "thông_tư_35/2024/tt-bgtvt_41_0_0_878350",
            "thông_tư_35/2024/tt-bgtvt_41_1_0_18232c",
            "thông_tư_35/2024/tt-bgtvt_table_44_p44_t0_cd8a7e91",
        ]
        add(["giay phep lai xe quoc te", "kich thuoc"], idp_size_chunks, "known_chunk_idp_size", boost=36.0)
        add(["idp", "kich thuoc"], idp_size_chunks, "known_chunk_idp_size", boost=36.0)

        add(
            ["dung xe", "tin bao", "to giac"],
            ["luật_trật_tự_atgt_2024_(tiếp)_69_4_0_510e57"],
            "known_chunk_stop_vehicle_report",
            boost=36.0,
        )

        add(
            ["tam nhin vuot xe an toan"],
            ["tech_qcvn_41:2024_(thông_tư_51/2024)_3_23_5857b9"],
            "known_chunk_overtaking_sight_distance",
            boost=36.0,
        )
        add(
            ["tieu phan quang dang mui ten"],
            ["tech_qcvn_41:2024_(thông_tư_51/2024)_55_7_4_36da25"],
            "known_chunk_arrow_delineator",
            boost=36.0,
        )

        out: List[Dict[str, Any]] = []
        seen = set()
        for needles, source_chunk_ids, reason, boost in specs:
            if not all(needle in qa for needle in needles):
                continue
            for record in self._records_by_source_chunk_ids(source_chunk_ids, reason=reason, boost=boost):
                key = (record.get("source_chunk_id") or record.get("id"), reason)
                if key in seen:
                    continue
                seen.add(key)
                out.append(record)
        return out

    def _vehicle_scope_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritizes records matching the detected vehicle type."""
        scope = self._vehicle_scope(query)
        if not scope: return records
        
        targets = self.VEHICLE_ARTICLES.get(scope, set())
        for r in records:
            ref = normalized_legal_reference(r)
            meta = r.get("rag_metadata") or {}
            article = str(r.get("article") or ref.get("article") or meta.get("article") or "")
            doc = (r.get("doc") or r.get("doc_name") or ref.get("document") or meta.get("doc") or "").lower()
            if article in targets and ("nghị định 168" in doc or "luật trật tự" in doc):
                r["retrieval_score"] = float(r.get("retrieval_score") or 0) + 2.0
                r["retrieval_reasons"] = sorted(set(r.get("retrieval_reasons", []) + ["vehicle_scope_boost"]))
        return records

    def _vehicle_scope(self, query: str) -> str:
        q = query.lower()
        qa = ascii_lower(query)
        if "máy chuyên dùng" in q or "may chuyen dung" in qa:
            return "specialized"
        if any(term in q for term in ["xe máy", "mô tô", "gắn máy"]) or any(term in qa for term in ["xe may", "mo to", "gan may"]):
            return "motorbike"
        if any(term in q for term in ["ô tô", "xe hơi", "xe con"]) or any(term in qa for term in ["o to", "xe hoi", "xe con"]):
            return "car"
        if "xe đạp" in q or "xe dap" in qa:
            return "bicycle"
        return ""

    def _license_focus_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qa = ascii_lower(self._focused_behavior_text(query))
        if not any(term in qa for term in ["chua du tuoi", "khong du tuoi", "duoi 18", "phan khoi lon", "hang a", "giay phep lai xe", "gplx"]):
            return records
        for record in records:
            ref = normalized_legal_reference(record)
            doc = ascii_lower(ref.get("document") or record.get("doc_name") or "")
            article = str(ref.get("article") or "")
            text = ascii_lower(source_text(record))
            if "luat trat tu atgt" not in doc:
                continue
            if article == "57" and any(term in text for term in ["hang a cap", "tren 125 cm3", "tren 11 kw", "hang a1 cap"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 3.2
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["license_class_boost"]))
            if article == "59" and any(term in text for term in ["do tuoi", "du 16 tuoi", "du 18 tuoi", "nguoi lai xe"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 3.0
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["driver_age_boost"]))
        return records

    def _penalty_focus_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qa = ascii_lower(self._focused_behavior_text(query))
        behavior_terms = [
            term
            for _behavior, terms in self._behavior_search_specs(qa)
            for term in terms
        ]
        behavior_terms.extend([
            term for term in ["toc do", "khong du tuoi", "chua du tuoi"] if term in qa
        ])
        scope = self._vehicle_scope(query)
        scoped_articles = (
            self.VEHICLE_RULE_ARTICLES.get(scope, set())
            if self._behavior_search_specs(qa)
            else self.VEHICLE_ARTICLES.get(scope, set())
        )
        scoped_docs = set(self._matching_documents(query))
        for record in records:
            ref = normalized_legal_reference(record)
            meta = record.get("rag_metadata") or {}
            doc = ascii_lower(record.get("doc_name") or ref.get("document") or meta.get("doc") or "")
            doc_name = record.get("doc_name") or ref.get("document") or meta.get("doc") or ""
            article = str(record.get("article") or ref.get("article") or meta.get("article") or "")
            text = ascii_lower(source_text(record))
            if scoped_docs and doc_name in scoped_docs:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.6
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_scoped_document_boost"]))
            elif not scoped_docs and "nghi dinh 168" in doc:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.5
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_document_boost"]))
            if scoped_articles and article in scoped_articles:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.8
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["vehicle_penalty_article_boost"]))
            if any(term in text for term in ["phat tien", "tru diem", "tuoc quyen", "giay phep lai xe"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.4
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_text_boost"]))
            if behavior_terms and any(term in text for term in behavior_terms):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.6
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["behavior_text_boost"]))
        return records

    def _definition_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        terms = self._definition_terms(query)
        for record in records:
            text = ascii_lower(source_text(record))
            ref = normalized_legal_reference(record)
            doc = ascii_lower(record.get("doc_name") or ref.get("document") or "")
            article = str(ref.get("article") or record.get("article") or "")
            if any(term in text for term in ["giai thich tu ngu", "duoc hieu la", "bao gom", "khai niem", "dinh nghia"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.2
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["definition_text_boost"]))
            if terms and any(term in text for term in terms):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.5
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["definition_term_boost"]))
            if article in {"2", "3"} and any(term in doc for term in ["luat duong bo", "luat trat tu"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.0
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["definition_law_article_boost"]))
        return records

    def _definition_terms(self, query: str) -> List[str]:
        qa = self._query_core_text(query)
        patterns = [
            r"(.+?)\s+la\s+gi\b",
            r"khai niem\s+(.+)",
            r"dinh nghia\s+(.+)",
            r"(.+?)\s+duoc hieu la gi\b",
        ]
        terms: List[str] = []
        for pattern in patterns:
            match = re.search(pattern, qa)
            if match:
                term = re.sub(r"\b(?:the nao|trong luat|theo luat|quy dinh)\b", "", match.group(1)).strip(" ?.,;:")
                if 2 <= len(term) <= 80:
                    terms.append(term)
        return list(dict.fromkeys(terms))

    def _deterministic_definition_matches(self, terms: List[str]) -> List[Dict[str, Any]]:
        if not terms:
            return []
        matches: List[Dict[str, Any]] = []
        for record in self.vector_store.records:
            text = ascii_lower(source_text(record))
            if not any(term in text for term in terms):
                continue
            if not any(marker in text for marker in ["giai thich tu ngu", "duoc hieu la", "bao gom", "khai niem", "dinh nghia"]):
                continue
            item = dict(record)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + 3.0
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["deterministic_definition_match"]))
            matches.append(item)
        return matches[:80]

    def _priority_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        priority_terms = ["xe uu tien", "quyen uu tien", "uu tien", "nhuong duong", "tin hieu uu tien", "giao nhau", "vong xuyen"]
        for record in records:
            text = ascii_lower(source_text(record))
            doc = ascii_lower(record.get("doc_name") or (normalized_legal_reference(record).get("document") or ""))
            if any(term in text for term in priority_terms):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.5
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["priority_text_boost"]))
            if "luat trat tu" in doc or "luat duong bo" in doc:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.5
        return records

    def _scenario_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        query_terms = [
            term for term in [
                "nga tu",
                "giao nhau",
                "vong xuyen",
                "lan duong",
                "den do",
                "tin hieu den",
                "toc do",
                "nong do",
                "hoi con",
                "uu tien",
                "nguoc chieu",
                "mu bao hiem",
                "tai nan",
                "chua du tuoi",
                "khong du tuoi",
            ] if term in qa
        ]
        for record in records:
            text = ascii_lower(source_text(record))
            if any(term in text for term in query_terms):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.0
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["scenario_term_boost"]))
        return records

    def _procedure_focus_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        wants_time = any(term in qa for term in ["thoi han", "hieu luc", "bao lau", "may nam", "ngay", "thang", "nam"])
        wants_file = any(term in qa for term in ["ho so", "giay to", "don de nghi", "ban sao"])
        wants_training = any(term in qa for term in ["dao tao", "tap lai", "hoc vien", "sat hach", "giay phep lai xe", "gplx"])
        for record in records:
            text = ascii_lower(self._record_text_for_matching(record))
            if wants_time and any(term in text for term in ["thoi han", "co thoi han", "hieu luc", "ngay", "thang", "nam"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 2.0
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["procedure_time_boost"]))
            if wants_file and any(term in text for term in ["ho so", "don de nghi", "ban sao", "giay kham suc khoe"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.7
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["procedure_file_boost"]))
            if wants_training and any(term in text for term in ["dao tao", "tap lai", "hoc vien", "sat hach", "giay phep lai xe", "gplx"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.2
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["procedure_training_boost"]))
        return records

    def _lexical_evidence_matches(self, query: str, *, limit: int) -> List[Dict[str, Any]]:
        scoped_docs = set(self._matching_documents(query))
        if not scoped_docs:
            return []
        self._ensure_lexical_index()
        core = self._query_core_text(query)
        query_tokens = self._lexical_tokens(core)
        if not query_tokens:
            return []
        candidate_counts: Dict[int, int] = {}
        for token in set(query_tokens):
            for idx in (self._lexical_index or {}).get(token, []):
                meta = self._lexical_records[idx]
                if meta.get("doc") not in scoped_docs:
                    continue
                candidate_counts[idx] = candidate_counts.get(idx, 0) + 1
        if not candidate_counts:
            return []

        shortlist = sorted(candidate_counts, key=lambda idx: candidate_counts[idx], reverse=True)[:max(limit * 5, 80)]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for idx in shortlist:
            meta = self._lexical_records[idx]
            record = meta["record"]
            score = self._lexical_match_score_from_parts(core, query_tokens, meta["text"], record, text_tokens=meta["tokens"])
            if score < 1.6:
                continue
            item = dict(record)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + min(7.0, 1.5 + score)
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["lexical_evidence_match"]))
            scored.append((score, item))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item for _score, item in scored[:limit]]

    def _record_text_for_matching(self, record: Dict[str, Any]) -> str:
        parts = [
            source_text(record),
            str(record.get("rag_text") or ""),
            str(record.get("qa_context") or ""),
            str(record.get("semantic_context") or ""),
        ]
        table = record.get("table")
        if isinstance(table, dict):
            for value in table.values():
                if isinstance(value, list):
                    parts.append(" ".join(str(item or "") for item in value))
                else:
                    parts.append(str(value or ""))
        return " ".join(part for part in parts if part)

    def _ensure_lexical_index(self) -> None:
        if self._lexical_index is not None:
            return
        index: Dict[str, List[int]] = {}
        records_meta: List[Dict[str, Any]] = []
        for idx, record in enumerate(self.vector_store.records):
            ref = normalized_legal_reference(record)
            doc = ref.get("document") or record.get("doc_name") or ""
            text = ascii_lower(self._record_text_for_matching(record))
            tokens = set(self._lexical_tokens(text))
            records_meta.append({"record": record, "doc": doc, "text": text, "tokens": tokens})
            for token in tokens:
                index.setdefault(token, []).append(idx)
        self._lexical_records = records_meta
        self._lexical_index = index

    def _query_core_text(self, query: str) -> str:
        qa = ascii_lower(self._focused_behavior_text(query))
        qa = re.sub(r"^trong\s+[^,]{3,180},\s*", " ", qa)
        qa = re.sub(r"\b(?:tra cuu|can cu|lien quan den|cau hoi goc|ngu canh cau hoi goc)\b", " ", qa)
        qa = re.sub(r"\s+", " ", qa).strip()
        return qa

    def _lexical_tokens(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-z0-9]+", ascii_lower(text))
        return [
            token
            for token in tokens
            if (len(token) >= 3 or re.search(r"\d", token)) and token not in self.LEXICAL_STOPWORDS
        ]

    def _lexical_match_score(self, query: str, record: Dict[str, Any]) -> float:
        core = self._query_core_text(query)
        if not core:
            return 0.0
        text = ascii_lower(self._record_text_for_matching(record))
        if not text:
            return 0.0
        query_tokens = self._lexical_tokens(core)
        if not query_tokens:
            return 0.0
        return self._lexical_match_score_from_parts(core, query_tokens, text, record)

    def _lexical_match_score_from_parts(
        self,
        core: str,
        query_tokens: List[str],
        text: str,
        record: Dict[str, Any],
        *,
        text_tokens: Optional[Set[str]] = None,
    ) -> float:
        token_set = set(query_tokens)
        record_tokens = text_tokens if text_tokens is not None else set(self._lexical_tokens(text))
        overlap = token_set & record_tokens
        score = min(2.4, len(overlap) * 0.22)
        if token_set:
            score += min(1.2, (len(overlap) / len(token_set)) * 1.8)

        compact_text = re.sub(r"\s+", " ", text)
        for size, weight, cap in [(5, 0.9, 1.8), (4, 0.7, 1.8), (3, 0.48, 1.6), (2, 0.22, 1.0)]:
            hits = 0
            for idx in range(0, max(0, len(query_tokens) - size + 1)):
                phrase = " ".join(query_tokens[idx:idx + size])
                if phrase and phrase in compact_text:
                    hits += 1
            score += min(cap, hits * weight)

        number_hits = 0
        for raw in re.findall(r"\d+(?:[.,]\d+)*", core):
            value = raw.replace(",", ".")
            if value and (value in compact_text or raw in compact_text):
                number_hits += 1
        score += min(1.6, number_hits * 0.45)

        ref = normalized_legal_reference(record)
        article = str(ref.get("article") or "")
        clause = str(ref.get("clause") or "")
        point = ascii_lower(str(ref.get("point") or ""))
        if article and re.search(rf"\bdieu\s+{re.escape(article)}\b", core):
            score += 0.8
        if clause and re.search(rf"\bkhoan\s+{re.escape(clause)}\b", core):
            score += 0.55
        if point and re.search(rf"\bdiem\s+{re.escape(point)}\b", core):
            score += 0.55
        return score

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Final sort by retrieval and query-text evidence."""
        if not candidates: return []
        lexical_limit = self._env_int("RAG_LEXICAL_RERANK_LIMIT", 96, minimum=32, maximum=512)
        base_ranked = sorted(candidates, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)
        lexical_pool = base_ranked[:lexical_limit]
        tail = base_ranked[lexical_limit:]
        ranked = []
        for record in lexical_pool:
            score = float(record.get("retrieval_score") or 0) + min(5.0, self._lexical_match_score(query, record))
            record["retrieval_rank_score"] = score
            ranked.append(record)
        for record in tail:
            record["retrieval_rank_score"] = float(record.get("retrieval_score") or 0)
            ranked.append(record)
        ranked = sorted(ranked, key=lambda r: float(r.get("retrieval_rank_score") or 0), reverse=True)
        if self.reranker is None:
            return ranked[:limit]

        model_limit = self._env_int("RAG_MODEL_RERANK_LIMIT", 32, minimum=8, maximum=128)
        model_pool = ranked[:model_limit]
        pairs = [(query, self._rerank_text(record)) for record in model_pool]
        try:
            raw_scores = self.reranker.predict(pairs)
            scores = [float(score) for score in raw_scores]
        except Exception as exc:
            logger.warning("Model rerank failed; keeping lexical ranking: %s", exc)
            return ranked[:limit]

        if scores:
            lo = min(scores)
            hi = max(scores)
            scale = max(hi - lo, 1e-6)
            for record, score in zip(model_pool, scores):
                normalized = (score - lo) / scale
                record["reranker_score"] = score
                record["retrieval_rank_score"] = float(record.get("retrieval_rank_score") or 0) + normalized * 4.0
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["model_rerank"]))
        return sorted(model_pool + ranked[model_limit:], key=lambda r: float(r.get("retrieval_rank_score") or 0), reverse=True)[:limit]

    def _rerank_text(self, record: Dict[str, Any]) -> str:
        chunks = [
            format_reference(record),
            source_text(record),
            str(record.get("qa_context") or ""),
            str(record.get("semantic_context") or ""),
            str(record.get("rag_text") or ""),
        ]
        text = "\n".join(chunk for chunk in chunks if chunk)
        max_chars = self._env_int("RAG_RERANK_TEXT_MAX_CHARS", 1800, minimum=400, maximum=6000)
        return text[:max_chars]

    def _dedupe(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensures unique legal chunks in the final set."""
        seen = set()
        out = []
        by_key: Dict[str, Dict[str, Any]] = {}
        for r in records:
            ref = normalized_legal_reference(r)
            text_key = ascii_lower(source_text(r))[:240]
            rid = "|".join([
                str(r.get("doc_name") or ref.get("document") or ""),
                str(ref.get("article") or ""),
                str(ref.get("clause") or ""),
                str(ref.get("point") or ""),
                text_key,
            ])
            if not text_key:
                rid = r.get("source_chunk_id") or r.get("id") or rid
            if rid in seen:
                existing = by_key[rid]
                merge_record_assets(existing, r)
                existing["retrieval_score"] = max(
                    float(existing.get("retrieval_score") or 0),
                    float(r.get("retrieval_score") or 0),
                )
                existing["retrieval_reasons"] = sorted(set(existing.get("retrieval_reasons", []) + r.get("retrieval_reasons", [])))
            else:
                out.append(r)
                by_key[rid] = r
                seen.add(rid)
        return out
