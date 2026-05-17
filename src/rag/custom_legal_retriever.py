import logging
import os
import re
from typing import Any

from src.rag.hybrid_vector_store import HybridLegalVectorStore
from src.rag.legal_graph_store import DeterministicLegalGraphStore
from src.rag.legal_utils import (
    SIGN_CODE_RE,
    extract_explicit_legal_refs,
    normalize_sign_code,
    normalized_legal_reference,
    source_text,
)
from src.rag.structured_table_retriever import StructuredTableRetriever
from src.rag.traffic_sign_catalog import TrafficSignCatalog


logger = logging.getLogger("CustomLegalRetriever")


class CustomLegalRetriever:
    """Hybrid legal retriever with deterministic graph expansion."""

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
        self.reranker = None
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder

                device = os.getenv("RAG_RERANKER_DEVICE")
                self.reranker = CrossEncoder(reranker_model, max_length=512, device=device)
            except Exception as exc:
                logger.warning("Reranker disabled because it could not be loaded: %s", exc)

    def retrieve(self, query: str, top_k: int = 8, expand_depth: int = 2) -> list[dict[str, Any]]:
        if self._looks_like_table_query(query):
            return self.retrieve_table(query, top_k=top_k, expand_depth=expand_depth)
        if self._looks_like_sign_query(query):
            return self.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth)

        search_query = self._expand_query(query)
        seed_records = []
        seed_records.extend(self._exact_sign_matches(search_query))
        seed_records.extend(self._exact_ref_matches(search_query))
        seed_records.extend(self._exact_phrase_matches(query))
        seed_records.extend(self.vector_store.search(search_query, top_k=max(top_k * 3, 20)))
        seed_records = self._dedupe(seed_records)
        seed_records = self._vehicle_scope_boost(query, seed_records)

        seed_node_ids = self.graph_store.lookup_record_nodes(seed_records)
        graph_nodes = self.graph_store.expand(seed_node_ids, depth=expand_depth, max_nodes=max(top_k * 6, 40))
        graph_source_ids = [
            node.get("id")
            for node in graph_nodes
            if node.get("type") == "legal_chunk" and node.get("id")
        ]
        expanded_records = self.vector_store.by_source_chunk_ids(graph_source_ids)
        for record in expanded_records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["graph_expansion"]))
            record["retrieval_score"] = max(float(record.get("retrieval_score") or 0.0), 0.45)

        candidates = self._dedupe(seed_records + expanded_records)
        candidates = self._graph_boost(candidates, seed_node_ids)
        candidates = self._vehicle_scope_boost(query, candidates)
        candidates = self._domain_boost(query, candidates)
        candidates = self._add_same_article_sanctions(query, candidates)
        candidates = self._rerank(query, candidates, limit=max(top_k * 2, top_k))
        candidates = self._pack_context(candidates, top_k)
        return candidates

    def retrieve_sign(self, query: str, top_k: int = 8, expand_depth: int = 1) -> list[dict[str, Any]]:
        """Dedicated traffic-sign retrieval constrained to QCVN/Thông tư 51 sign evidence."""
        search_query = self._expand_query(query)
        codes = list(dict.fromkeys(self.sign_catalog.find_codes(query) + self._sign_code_hints(query)))
        seed_records = self.sign_catalog.records_for_codes(codes)
        seed_records.extend(self._exact_sign_matches(search_query))
        seed_records.extend(
            record
            for record in self.vector_store.search(search_query, top_k=max(top_k * 4, 20))
            if self._is_qcvn_sign_record(record)
        )
        seed_records = self._dedupe(seed_records)
        seed_records = self._boost_sign_specific_records(query, seed_records)

        seed_node_ids = self.graph_store.lookup_record_nodes(seed_records)
        graph_nodes = self.graph_store.expand(seed_node_ids, depth=expand_depth, max_nodes=max(top_k * 4, 24))
        graph_source_ids = [
            node.get("id")
            for node in graph_nodes
            if node.get("type") == "legal_chunk" and node.get("id")
        ]
        expanded_records = [
            record
            for record in self.vector_store.by_source_chunk_ids(graph_source_ids)
            if self._is_qcvn_sign_record(record)
        ]
        for record in expanded_records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["sign_graph_expansion"]))
            record["retrieval_score"] = max(float(record.get("retrieval_score") or 0.0), 0.65)

        candidates = self._dedupe(seed_records + expanded_records)
        candidates = self._boost_sign_specific_records(query, candidates)
        candidates = self._rerank(query, candidates, limit=max(top_k * 2, top_k))
        return self._pack_context(candidates, top_k)

    def retrieve_table(self, query: str, top_k: int = 8, expand_depth: int = 1) -> list[dict[str, Any]]:
        """Dedicated table lookup that scores rows/cells before graph expansion."""
        seed_records = self.table_retriever.search(query, top_k=max(top_k * 2, top_k))
        seed_node_ids = self.graph_store.lookup_record_nodes(seed_records)
        graph_nodes = self.graph_store.expand(seed_node_ids, depth=expand_depth, max_nodes=max(top_k * 3, 18))
        graph_source_ids = [
            node.get("id")
            for node in graph_nodes
            if node.get("type") == "legal_chunk" and node.get("id")
        ]
        expanded_records = [
            record
            for record in self.vector_store.by_source_chunk_ids(graph_source_ids)
            if record.get("rag_modality") == "table" or record.get("table")
        ]
        for record in expanded_records:
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["table_graph_expansion"]))
            record["retrieval_score"] = max(float(record.get("retrieval_score") or 0.0), 0.55)
        candidates = self._dedupe(seed_records + expanded_records)
        candidates = self._rerank(query, candidates, limit=max(top_k * 2, top_k))
        return self._pack_context(candidates, top_k)

    def _expand_query(self, query: str) -> str:
        q = (query or "").lower()
        expansions = []
        if "vượt đèn đỏ" in q or "đèn đỏ" in q:
            expansions.append("không chấp hành hiệu lệnh của đèn tín hiệu giao thông")
        if "xe máy" in q:
            expansions.append("xe mô tô xe gắn máy")
        if "phạt" in q:
            expansions.append("phạt tiền mức phạt trừ điểm giấy phép lái xe")
        if "biển báo" in q or SIGN_CODE_RE.search(query or ""):
            expansions.append("biển báo hiệu đường bộ ý nghĩa hình ảnh")
        if "biển báo cấm" in q or "biển cấm" in q or "viền đỏ" in q:
            expansions.append("QCVN 41:2024 Thông tư 51/2024 Phụ lục B nhóm biển báo cấm P.101 P.102 P.103 P.104")
        if "bảo lưu" in q or "sát hạch lý thuyết" in q:
            expansions.append("bảo lưu kết quả sát hạch lý thuyết 01 năm")
        if "sát hạch viên" in q and ("ngồi trên xe" in q or "bao nhiêu" in q or "khuyết tật" in q):
            expansions.append("hai sát hạch viên ngồi trên xe chấm điểm trực tiếp")
        if "dân tộc thiểu số" in q or "không biết đọc" in q or "không biết viết" in q:
            expansions.append("người dân tộc thiểu số không biết đọc viết tiếng Việt hỏi đáp người phiên dịch")
        return " ".join([query or "", *expansions]).strip()

    def _looks_like_sign_query(self, query: str) -> bool:
        q = (query or "").lower()
        return (
            SIGN_CODE_RE.search(query or "") is not None
            or "biển báo" in q
            or "biển cấm" in q
            or ("viền đỏ" in q and "biển" in q)
        )

    def _looks_like_table_query(self, query: str) -> bool:
        q = (query or "").lower()
        table_words = ["bảng", "dòng", "cột", "ô bảng", "tra bảng", "table"]
        return any(word in q for word in table_words)

    def _is_qcvn_sign_record(self, record: dict[str, Any]) -> bool:
        doc = (record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "").lower()
        modality = record.get("rag_modality", "text")
        text = (record.get("rag_text") or source_text(record)).lower()
        is_qcvn = "qcvn" in doc or "thông tư 51" in doc
        has_sign = record.get("figure") or record.get("figures") or SIGN_CODE_RE.search(text or "")
        return bool(is_qcvn and has_sign and modality in {"sign", "figure", "text"})

    def _boost_sign_specific_records(self, query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q_codes = set(self.sign_catalog.find_codes(query) + self._sign_code_hints(query))
        for record in records:
            figure = record.get("figure") if isinstance(record.get("figure"), dict) else {}
            record_code = normalize_sign_code(figure.get("code") or "")
            text = (record.get("rag_text") or source_text(record)).lower()
            boost = 0.0
            if record.get("retrieval_reasons") and "traffic_sign_catalog" in record.get("retrieval_reasons", []):
                boost += 2.0
            if q_codes and record_code in q_codes:
                boost += 1.2
            if "phụ lục" in text or "hình" in text:
                boost += 0.35
            if "khoảng cách mép ngoài" in text and "điều 22" in text:
                boost -= 0.4
            if record.get("rag_modality") in {"sign", "figure"}:
                boost += 0.25
            if boost:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + boost
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["sign_scope_boost"]))
        return records

    def _sign_code_hints(self, query: str) -> list[str]:
        q = (query or "").lower()
        hints = [normalize_sign_code(match.group(0)) for match in SIGN_CODE_RE.finditer(query or "")]
        if any(k in q for k in ["cấm đi ngược chiều", "ngược chiều", "no entry", "thanh ngang", "gạch ngang", "vạch ngang", "dấu trừ"]):
            hints.append("P102")
        if "đường cấm" in q or ("viền đỏ" in q and "nền trắng" in q and not any(k in q for k in ["thanh ngang", "gạch ngang", "ô tô", "xe máy", "người đi bộ", "rẽ", "quay đầu"])):
            hints.append("P101")
        if "cấm xe ô tô" in q or "xe hơi" in q:
            hints.append("P103A")
        if "cấm xe máy" in q or "mô tô" in q:
            hints.append("P104")
        if "cấm xe ô tô tải" in q or "xe tải" in q:
            hints.append("P106A")
        if "cấm người đi bộ" in q:
            hints.append("P112")
        if "cấm rẽ trái" in q:
            hints.append("P123A")
        if "cấm rẽ phải" in q:
            hints.append("P123B")
        if "cấm quay đầu" in q or "quay đầu" in q:
            hints.append("P124A")
        if "cấm vượt" in q:
            hints.append("P125")
        if "tốc độ tối đa" in q or "hạn chế tốc độ" in q:
            hints.append("P127")
        if "cấm dừng" in q:
            hints.append("P130")
        if "cấm đỗ" in q:
            hints.append("P131")
        return list(dict.fromkeys(hints))

    def _exact_sign_matches(self, query: str) -> list[dict[str, Any]]:
        codes = self._sign_code_hints(query)
        if not codes:
            return []
        results = []
        for rank, code in enumerate(codes):
            for record in self.vector_store.records:
                record_code = ""
                if isinstance(record.get("figure"), dict):
                    record_code = normalize_sign_code(record["figure"].get("code") or "")
                text = (record.get("rag_text") or "").replace(".", "").replace(" ", "").upper()
                modality = record.get("rag_modality")
                if modality in {"sign", "figure"}:
                    if record_code != code:
                        continue
                elif code not in text:
                    continue
                item = dict(record)
                base = 3.0 if record_code == code else 1.5
                item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), base - rank * 0.05)
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["exact_sign_code"]))
                results.append(item)
        return results[:30]

    def _exact_ref_matches(self, query: str) -> list[dict[str, Any]]:
        refs = extract_explicit_legal_refs(query)
        if not refs:
            return []
        records = []
        documents = sorted(
            {
                (record.get("legal_reference") or {}).get("document") or record.get("doc_name") or ""
                for record in self.vector_store.records
            }
        )
        for ref in refs:
            for document in documents:
                hits = self.vector_store.by_ref(document, ref["article"], ref.get("clause", ""), ref.get("point", ""))
                for record in hits:
                    item = dict(record)
                    item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), 1.1)
                    item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["exact_legal_ref"]))
                    records.append(item)
        return records

    def _exact_phrase_matches(self, query: str) -> list[dict[str, Any]]:
        q = (query or "").lower()
        phrases = []
        if "đèn đỏ" in q or "vượt đèn" in q:
            phrases.append("không chấp hành hiệu lệnh của đèn tín hiệu giao thông")
        if "phạm vi điều chỉnh" in q:
            phrases.append("phạm vi điều chỉnh")
        if "thời hiệu xử phạt" in q:
            phrases.append("thời hiệu xử phạt vi phạm hành chính")
        if "hình thức xử phạt" in q:
            phrases.append("hình thức xử phạt")
        if "bảo lưu" in q or "sát hạch lý thuyết" in q:
            phrases.append("bảo lưu kết quả sát hạch lý thuyết")
        if "sát hạch viên" in q and ("ngồi trên xe" in q or "bao nhiêu" in q or "khuyết tật" in q):
            phrases.extend(["hai sát hạch viên ngồi trên xe", "sát hạch viên ngồi trên xe chấm điểm trực tiếp"])
        if "dân tộc thiểu số" in q or "không biết đọc" in q or "không biết viết" in q:
            phrases.extend(
                [
                    "người dân tộc thiểu số không biết đọc, viết tiếng Việt",
                    "sát hạch lý thuyết bằng hình thức hỏi - đáp",
                    "người phiên dịch",
                ]
            )
        if "giấy phép lái xe" in q and "điểm" in q:
            phrases.extend(["bao gồm 12 điểm", "sau thời hạn ít nhất là 06 tháng kể từ ngày bị trừ hết điểm"])
        if not phrases:
            return []
        records = []
        for record in self.vector_store.records:
            text = (record.get("rag_text") or "").lower()
            if any(phrase in text for phrase in phrases):
                item = dict(record)
                item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), 1.35)
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["exact_legal_phrase"]))
                records.append(item)
        return records[:50]

    def _graph_boost(self, records: list[dict[str, Any]], seed_node_ids: list[str]) -> list[dict[str, Any]]:
        seed_set = set(seed_node_ids)
        for record in records:
            source_id = record.get("source_chunk_id")
            if source_id in seed_set:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.2
            if record.get("tables"):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.05
            if record.get("figures") or record.get("figure"):
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 0.08
        return records

    def _vehicle_scope(self, query: str) -> str:
        q = (query or "").lower()
        if "xe máy chuyên dùng" in q or "máy chuyên dùng" in q:
            return "specialized"
        if "xe đạp máy" in q or "xe đạp điện" in q:
            return "bicycle"
        if re.search(r"\b(mô\s*tô|xe\s*mô\s*tô|xe\s*máy|xe\s*gắn\s*máy)\b", q):
            return "motorbike"
        if re.search(r"\b(ô\s*tô|ôtô|oto|xe\s*con|xe\s*tải|xe\s*khách|xe\s*chở\s*người\s*bốn\s*bánh|xe\s*chở\s*hàng\s*bốn\s*bánh)\b", q):
            return "car"
        return ""

    def _vehicle_scope_boost(self, query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        scope = self._vehicle_scope(query)
        if not scope:
            return records
        target_articles = self.VEHICLE_ARTICLES.get(scope, set())
        for record in records:
            ref = normalized_legal_reference(record)
            doc = (ref.get("document") or record.get("doc_name") or "").lower()
            article = str(ref.get("article") or "")
            if "nghị định 168" not in doc or article not in self.VEHICLE_SCOPED_ARTICLES:
                continue
            if article in target_articles:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + 1.4
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["vehicle_scope_match"]))
            else:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) - 1.0
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["vehicle_scope_mismatch"]))
        return records

    def _domain_boost(self, query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q = (query or "").lower()
        wants_penalty = any(k in q for k in ["phạt", "mức phạt", "bao nhiêu tiền", "trừ điểm", "tước"])
        wants_sign = "biển" in q or SIGN_CODE_RE.search(query or "") is not None
        wants_red_light = "đèn đỏ" in q or "vượt đèn" in q
        wants_theory_reserve = "bảo lưu" in q or "sát hạch lý thuyết" in q
        wants_examiner_count = "sát hạch viên" in q and ("ngồi trên xe" in q or "bao nhiêu" in q or "khuyết tật" in q)
        wants_ethnic_oral = "dân tộc thiểu số" in q or "không biết đọc" in q or "không biết viết" in q
        wants_license_points = "giấy phép lái xe" in q and "điểm" in q
        alcohol_band = self._alcohol_band(q)
        for record in records:
            text = (record.get("rag_text") or "").lower()
            doc = (record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "").lower()
            ref = normalized_legal_reference(record)
            boost = 0.0
            if wants_red_light and "không chấp hành hiệu lệnh của đèn tín hiệu giao thông" in text:
                boost += 0.75
            if wants_penalty and any(k in text for k in ["phạt tiền", "trừ điểm", "tước quyền", "xử phạt"]):
                boost += 0.35
            if wants_penalty and "nghị định" in doc:
                boost += 0.12
            if wants_sign and (record.get("figures") or record.get("figure") or "biển báo" in text):
                boost += 0.25
            if wants_theory_reserve and "bảo lưu kết quả sát hạch lý thuyết" in text:
                boost += 0.7
            if wants_examiner_count and "sát hạch viên ngồi trên xe" in text:
                boost += 0.7
            if wants_ethnic_oral and (
                "người dân tộc thiểu số không biết đọc, viết tiếng việt" in text
                or "sát hạch lý thuyết bằng hình thức hỏi - đáp" in text
            ):
                boost += 0.7
            if wants_license_points:
                article = str(ref.get("article") or "")
                if "luật trật tự" in doc and article == "58":
                    if "bao gồm 12 điểm" in text or "ít nhất là 06 tháng" in text or "bị trừ hết điểm" in text:
                        boost += 2.4
                        record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["license_points_law"]))
                elif "nghị định 168" in doc and article == "51":
                    boost += 0.35
            if alcohol_band:
                alcohol_score = self._alcohol_band_score(alcohol_band, text)
                boost += alcohol_score
                if alcohol_score > 0:
                    record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["alcohol_band_match"]))
                elif alcohol_score < 0:
                    record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["alcohol_band_mismatch"]))
            if boost:
                record["retrieval_score"] = float(record.get("retrieval_score") or 0) + boost
                record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["domain_boost"]))
        return records

    def _alcohol_band(self, query_lower: str) -> str:
        if any(k in query_lower for k in ["chưa vượt quá 50", "không vượt quá 50", "dưới 50", "0,25"]):
            return "low"
        if any(k in query_lower for k in ["vượt quá 80", "trên 80", "0,4"]):
            return "high"
        if "vượt quá 50" in query_lower and "80" in query_lower:
            return "mid"
        return ""

    def _alcohol_band_score(self, band: str, text: str) -> float:
        text = re.sub(r"\s+", " ", text or "")
        if not any(k in text for k in ["nồng độ cồn", "miligam/100", "khí thở"]):
            return 0.0
        has_low = "chưa vượt quá 50" in text or "chưa vượt quá 0,25" in text
        has_mid = "vượt quá 50" in text and ("đến 80" in text or "80 miligam" in text or "0,4" in text) and "chưa vượt quá" not in text
        has_high = "vượt quá 80" in text or "vượt quá 0,4" in text
        if band == "low":
            return 1.35 if has_low else (-2.0 if has_mid or has_high else 0.0)
        if band == "mid":
            return 1.2 if has_mid else (-1.8 if has_low or has_high else 0.0)
        if band == "high":
            return 1.35 if has_high else (-2.0 if has_low or has_mid else 0.0)
        return 0.0

    def _add_same_article_sanctions(self, query: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        q = (query or "").lower()
        wants_sanctions = any(
            k in q
            for k in [
                "phạt",
                "mức phạt",
                "bao nhiêu tiền",
                "trừ điểm",
                "tước",
                "nồng độ cồn",
                "đèn đỏ",
                "vượt đèn",
                "không chấp hành",
            ]
        )
        if not wants_sanctions or not records:
            return records

        additions = []
        ranked_seeds = sorted(records, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)
        best_score = float(ranked_seeds[0].get("retrieval_score") or 0)
        seeds = [record for record in ranked_seeds[:12] if float(record.get("retrieval_score") or 0) >= best_score - 0.85]
        for seed in seeds:
            seed_ref = normalized_legal_reference(seed)
            seed_doc = seed_ref.get("document") or seed.get("doc_name") or ""
            seed_article = str(seed_ref.get("article") or "")
            seed_clause = str(seed_ref.get("clause") or "")
            if not seed_doc or not seed_article or not seed_clause:
                continue
            if self._is_sanction_record(seed):
                continue
            if "vehicle_scope_mismatch" in (seed.get("retrieval_reasons") or []):
                continue
            if "phạt tiền" not in (seed.get("rag_text") or source_text(seed)).lower():
                continue
            for sanction in self._same_article_sanction_records(seed_doc, seed_article):
                if not self._sanction_applies_to_seed(sanction, seed):
                    continue
                item = dict(sanction)
                item["retrieval_score"] = max(float(item.get("retrieval_score") or 0), float(seed.get("retrieval_score") or 0) - 0.05)
                item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["same_article_sanction"]))
                additions.append(item)
        return self._dedupe(records + additions)

    def _same_article_sanction_records(self, document: str, article: str) -> list[dict[str, Any]]:
        records = []
        for record in self.vector_store.records:
            ref = normalized_legal_reference(record)
            if (ref.get("document") or record.get("doc_name") or "") != document:
                continue
            if str(ref.get("article") or "") != str(article):
                continue
            if self._is_sanction_record(record):
                records.append(dict(record))
        return records

    def _is_sanction_record(self, record: dict[str, Any]) -> bool:
        text = source_text(record).lower()
        if not any(k in text for k in ["trừ điểm giấy phép lái xe", "tước quyền sử dụng giấy phép lái xe"]):
            return False
        return "thực hiện hành vi quy định" in text or "ngoài việc" in text

    def _sanction_applies_to_seed(self, sanction: dict[str, Any], seed: dict[str, Any]) -> bool:
        seed_ref = normalized_legal_reference(seed)
        sanction_ref = normalized_legal_reference(sanction)
        if (seed_ref.get("document") or seed.get("doc_name") or "") != (sanction_ref.get("document") or sanction.get("doc_name") or ""):
            return False
        if str(seed_ref.get("article") or "") != str(sanction_ref.get("article") or ""):
            return False

        target_clause = str(seed_ref.get("clause") or "")
        target_point = str(seed_ref.get("point") or "").lower()
        if not target_clause:
            return False

        text = self._normalize_legal_text(source_text(sanction))
        if f"khoản {target_clause}" not in text:
            return False
        for segment in re.split(r"[.;]", text):
            if f"khoản {target_clause}" not in segment:
                continue
            if "điểm" not in segment:
                return True
            if target_point and re.search(rf"\bđiểm\s+{re.escape(target_point)}\b", segment):
                return True
        return False

    def _normalize_legal_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    def _rerank(self, query: str, records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if not self.reranker or not records:
            return sorted(records, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)[:limit]
        pairs = [(query, (record.get("rag_text") or record.get("source_body_exact") or "")[:1600]) for record in records]
        try:
            scores = self.reranker.predict(pairs)
        except Exception as exc:
            logger.warning("Reranker failed, using hybrid scores only: %s", exc)
            return sorted(records, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)[:limit]
        out = []
        for record, score in zip(records, scores):
            item = dict(record)
            item["rerank_score"] = float(score)
            item["retrieval_score"] = float(item.get("retrieval_score") or 0) + float(score)
            out.append(item)
        return sorted(out, key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)[:limit]

    def _pack_context(self, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        packed = []
        seen_source = set()
        seen_assets = set()
        ordered_records = [
            record
            for record in records
            if "vehicle_scope_mismatch" not in (record.get("retrieval_reasons") or [])
            and "alcohol_band_mismatch" not in (record.get("retrieval_reasons") or [])
        ]
        ordered_records.extend(record for record in records if record not in ordered_records)
        for record in ordered_records:
            modality = record.get("rag_modality", "text")
            source_id = record.get("source_chunk_id") or record.get("id")
            asset_key = (modality, source_id, record.get("image_path"))
            if modality == "text":
                if source_id in seen_source:
                    continue
                seen_source.add(source_id)
            else:
                if asset_key in seen_assets:
                    continue
                seen_assets.add(asset_key)
            packed.append(record)
            if len(packed) >= top_k:
                break
        return packed

    def _dedupe(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lookup = {}
        for record in records:
            key = (
                record.get("source_chunk_id") or record.get("id"),
                record.get("rag_modality", "text"),
                record.get("image_path") or "",
                (record.get("figure") or {}).get("id") if isinstance(record.get("figure"), dict) else "",
                (record.get("table") or {}).get("id") if isinstance(record.get("table"), dict) else "",
            )
            existing = lookup.get(key)
            if existing is None or float(record.get("retrieval_score") or 0) > float(existing.get("retrieval_score") or 0):
                lookup[key] = dict(record)
        return list(lookup.values())
