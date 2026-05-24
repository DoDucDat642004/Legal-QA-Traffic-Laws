"""
Custom Legal Retriever for Vietnamese Traffic Law.

This module implements a multi-stage retrieval strategy combining deterministic
rule-based matching, traffic sign catalog lookups, semantic vector search,
and graph-based context expansion.
"""

import logging
import os
import re
from pathlib import Path
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
    record_image_paths,
    source_text,
)
from src.rag.query_planner import LegalQueryPlanner, QueryPlan
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

    def __init__(
        self,
        vector_store: HybridLegalVectorStore,
        graph_store: DeterministicLegalGraphStore,
        *,
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        use_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.sign_catalog = TrafficSignCatalog(vector_store.records)
        self.table_retriever = StructuredTableRetriever(vector_store.records)
        self.client = None # LLM Client for AI semantic probes
        self.reranker = None
        
        if use_reranker:
            model_path = Path(reranker_model).expanduser()
            allow_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes", "on"}
            if model_path.exists() or allow_download:
                try:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder(str(model_path) if model_path.exists() else reranker_model, max_length=512)
                except Exception as exc:
                    logger.warning("Reranker disabled: %s", exc)
            else:
                logger.info("Reranker skipped because model is not local and downloads are disabled.")
        
        self.query_planner = LegalQueryPlanner()
        from src.rag.adaptive_query import AdaptiveQuestionAnalyzer
        self.adaptive_analyzer = AdaptiveQuestionAnalyzer()

    def retrieve(self, query: str, top_k: int = 8, expand_depth: int = 2) -> List[Dict[str, Any]]:
        """General hybrid retrieval flow."""
        if looks_like_table_query(query):
            return self.retrieve_table(query, top_k=top_k, expand_depth=expand_depth)
        if looks_like_sign_query(query):
            return self.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth)
        return self.retrieve_general(query, top_k=top_k, expand_depth=expand_depth)

    def retrieve_general(self, query: str, top_k: int = 8, expand_depth: int = 2) -> List[Dict[str, Any]]:
        """Hybrid text retrieval without automatic sign/table rerouting."""
        search_query = self._expand_query(query)
        seed_records = []
        
        # Phase 1: High-precision deterministic matches
        seed_records.extend(self._exact_sign_matches(search_query))
        seed_records.extend(self._exact_ref_matches(search_query))
        seed_records.extend(self._known_legal_ref_matches(search_query))
        
        # Phase 2: Hybrid vector search
        seed_records.extend(self.vector_store.search(search_query, top_k=max(top_k * 3, 20)))
        seed_records = self._dedupe(seed_records)

        # Phase 3: Graph Expansion for related clauses
        seed_node_ids = self.graph_store.lookup_record_nodes(seed_records)
        graph_nodes = self.graph_store.expand(seed_node_ids, depth=expand_depth, max_nodes=max(top_k * 6, 40))
        graph_ids = [n.get("id") for n in graph_nodes if n.get("type") == "legal_chunk"]
        expanded_records = self.vector_store.by_source_chunk_ids(graph_ids)

        candidates = self._dedupe(seed_records + expanded_records)
        
        # Phase 4: Contextual Boosting (Vehicle type, etc.)
        candidates = self._vehicle_scope_boost(query, candidates)
        
        # Phase 5: Reranking
        candidates = self._rerank(query, candidates, limit=top_k)
        return candidates

    def retrieve_penalty(self, query: str, top_k: int = 8, expand_depth: int = 2, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Penalty-specialized retrieval that bypasses sign rerouting."""
        penalty_query = " ".join([
            query,
            "Nghị định 168/2024/NĐ-CP mức phạt tiền trừ điểm giấy phép lái xe tước giấy phép xử phạt vi phạm",
        ]).strip()
        records = self.retrieve_general(penalty_query, top_k=max(top_k * 3, 24), expand_depth=expand_depth)
        records.extend(self._known_legal_ref_matches(query))
        scope = self._vehicle_scope(query)
        if scope:
            for article in self.VEHICLE_ARTICLES.get(scope, set()):
                records.extend(self.vector_store.by_ref("Nghị định 168/2024/NĐ-CP", article))
        records = self._dedupe(records)
        records = self._vehicle_scope_boost(query, records)
        records = self._penalty_focus_boost(query, records)
        records = self._rerank(query, records, limit=top_k)
        for record in records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_route"]))
        return records

    def retrieve_sign(self, query: str, top_k: int = 8, expand_depth: int = 1, plan: Optional[QueryPlan] = None) -> List[Dict[str, Any]]:
        """Sign-specialized retrieval with visual feature mapping and AI probing."""
        ai_codes = []
        if self.client and len(query.split()) > 3:
            ai_codes = self.sign_catalog.ai_semantic_probe(query, self.client)
            if ai_codes: logger.info("AI Probe found: %s", ai_codes)
            
        codes = list(dict.fromkeys(self.sign_catalog.find_codes(query) + self._sign_code_hints(query) + ai_codes))
        seed_records = self.sign_catalog.records_for_codes(codes)
        
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
        records.extend(self.vector_store.search(
            table_query,
            top_k=max(top_k * 2, 16),
            filters={"modalities": ["table"]},
        ))
        records = self._dedupe(records)
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
        terms = self._definition_terms(query)
        records.extend(self._deterministic_definition_matches(terms))
        records = self._dedupe(records)
        records = self._definition_boost(query, records)
        return self._rerank(query, records, limit=top_k)

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
        for record in records:
            if record_image_paths(record):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.5
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["source_image_route"]))
        image_records = [record for record in records if record_image_paths(record)]
        return self._rerank(query, image_records or records, limit=top_k)

    # --- Internal Logic Paths ---

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
        if "không đội mũ" in q or "khong doi mu" in qa: exp.append("không đội mũ bảo hiểm cho người đi mô tô xe máy")
        if "ngược chiều" in q or "nguoc chieu" in qa: exp.append("đi ngược chiều đường một chiều đi vào đường cấm")
        if "tai nạn" in q or "tai nan" in qa: exp.append("gây tai nạn giao thông trách nhiệm người điều khiển phương tiện")
        if "p.102" in q or "p102" in qa:
            exp.append("cấm đi ngược chiều đi vào đường cấm")
        return " ".join([query, *exp]).strip()

    def _sign_group_description_records(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Finds high-level definitions for sign categories."""
        q = query.lower()
        if "biển báo cấm" in q or "biển cấm" in q: return self.vector_store.search("nhóm biển báo cấm", top_k=2)
        if "biển hiệu lệnh" in q: return self.vector_store.search("nhóm biển hiệu lệnh", top_k=2)
        return []

    def _sign_code_hints(self, query: str) -> List[str]:
        """Extracts codes like P.102 directly from text."""
        return [normalize_sign_code(m.group(0)) for m in SIGN_CODE_RE.finditer(query)]

    def _exact_sign_matches(self, query: str) -> List[Dict[str, Any]]:
        """Quick lookup for sign-specific records in the vector DB."""
        codes = self._sign_code_hints(query)
        if not codes: return []
        # Return records that explicitly tag these sign codes
        return [r for r in self.vector_store.records if any(c in r.get("sign_codes", []) for c in codes)]

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
        qa = ascii_lower(query)
        specs: List[Tuple[str, str, str]] = []

        if any(term in qa for term in ["chua du tuoi", "khong du tuoi", "duoi 18", "phan khoi lon", "gplx hang a", "giay phep lai xe hang a"]):
            specs.extend([
                ("Luật Trật tự ATGT 2024 (Tiếp)", "57", "known_ref_license_class"),
                ("Luật Trật tự ATGT 2024 (Tiếp)", "59", "known_ref_driver_age"),
            ])

        if any(term in qa for term in ["gay tai nan", "tai nan giao thong", "tai nan cho nguoi khac"]):
            specs.append(("Luật Trật tự ATGT 2024 (Tiếp)", "80", "known_ref_accident_responsibility"))

        if any(term in qa for term in ["vuot den do", "tin hieu den", "nong do con", "hoi con", "say xin", "khong doi mu", "mu bao hiem", "nguoc chieu", "duong cam"]):
            specs.append(("Nghị định 168/2024/NĐ-CP", "7", "known_ref_motorbike_penalty"))

        out: List[Dict[str, Any]] = []
        seen = set()
        for document, article, reason in specs:
            for record in self.vector_store.by_ref(document, article):
                key = (record.get("source_chunk_id") or record.get("id"), reason)
                if key in seen:
                    continue
                seen.add(key)
                item = dict(record)
                item["retrieval_score"] = float(item.get("retrieval_score") or 0) + 3.0
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + [reason]))
                out.append(item)
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
        if any(term in q for term in ["xe máy", "mô tô", "gắn máy"]) or any(term in qa for term in ["xe may", "mo to", "gan may"]):
            return "motorbike"
        if any(term in q for term in ["ô tô", "xe hơi", "xe con"]) or any(term in qa for term in ["o to", "xe hoi", "xe con"]):
            return "car"
        if "xe đạp" in q or "xe dap" in qa:
            return "bicycle"
        if "máy chuyên dùng" in q or "may chuyen dung" in qa:
            return "specialized"
        return ""

    def _penalty_focus_boost(self, query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        qa = ascii_lower(query)
        behavior_terms = [
            term for term in [
                "den do",
                "tin hieu den",
                "nguoc chieu",
                "duong cam",
                "doi mu",
                "mu bao hiem",
                "nong do",
                "hoi con",
                "toc do",
                "tai nan",
                "khong du tuoi",
                "chua du tuoi",
            ] if term in qa
        ]
        for record in records:
            ref = normalized_legal_reference(record)
            meta = record.get("rag_metadata") or {}
            doc = ascii_lower(record.get("doc_name") or ref.get("document") or meta.get("doc") or "")
            text = ascii_lower(source_text(record))
            if "nghi dinh 168" in doc:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.5
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_document_boost"]))
            if any(term in text for term in ["phat tien", "tru diem", "tuoc quyen", "giay phep lai xe"]):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.4
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["penalty_text_boost"]))
            if behavior_terms and any(term in text for term in behavior_terms):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.8
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
        qa = ascii_lower(query)
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

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """Final sort by retrieval score."""
        if not candidates: return []
        return sorted(candidates, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)[:limit]

    def _dedupe(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensures unique legal chunks in the final set."""
        seen = set()
        out = []
        by_key: Dict[str, Dict[str, Any]] = {}
        for r in records:
            rid = r.get("source_chunk_id") or r.get("id")
            if not rid:
                rid = "|".join([
                    str(r.get("doc_name") or ""),
                    str((r.get("legal_reference") or {}).get("article") or ""),
                    str((r.get("legal_reference") or {}).get("clause") or ""),
                    str((r.get("legal_reference") or {}).get("point") or ""),
                    str(r.get("image_path") or ""),
                    str(source_text(r)[:120]),
                ])
            if rid in seen:
                merge_record_assets(by_key[rid], r)
            else:
                out.append(r)
                by_key[rid] = r
                seen.add(rid)
        return out
