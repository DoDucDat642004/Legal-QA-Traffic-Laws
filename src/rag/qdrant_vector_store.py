import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.rag.legal_utils import normalized_legal_reference, tokenize
from src.rag.rag_store_config import DEFAULT_EMBEDDING_MODEL, RAGStoreConfig
from src.rag.record_expander import load_expanded_records
from src.rag.embedding_backends import make_embedder


logger = logging.getLogger("QdrantLegalVectorStore")

QDRANT_INDEX_VERSION = "qdrant_openvino_vietnamese_bi_encoder_768_v1"


def _embedding_text(record: dict[str, Any]) -> str:
    text = record.get("rag_text", "") or ""
    max_chars = int(os.getenv("QDRANT_EMBED_TEXT_MAX_CHARS", "1800"))
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return text[:head] + "\n...\n" + text[-tail:]


def _point_id(record: dict[str, Any]) -> str:
    raw = "|".join(
        str(x or "")
        for x in [
            record.get("id"),
            record.get("rag_parent_id"),
            record.get("source_chunk_id") or record.get("id"),
            record.get("rag_modality", "text"),
            record.get("image_path"),
            (record.get("figure") or {}).get("id") if isinstance(record.get("figure"), dict) else "",
            (record.get("table") or {}).get("id") if isinstance(record.get("table"), dict) else "",
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def _clean_payload(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_clean_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _clean_payload(item) for key, item in value.items()}
    return value


class QdrantLegalVectorStore:
    """Qdrant-backed vector store implementing the same interface as HybridLegalVectorStore."""

    def __init__(
        self,
        processed_path: str | Path = "data/processed",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        force_reindex: bool = False,
        config: RAGStoreConfig | None = None,
    ):
        self.processed_path = Path(processed_path)
        self.embedding_model_name = embedding_model
        self.config = config or RAGStoreConfig()
        self.records: list[dict[str, Any]] = []
        self.documents: list[str] = []
        self.record_by_source_chunk: dict[str, list[dict[str, Any]]] = {}
        self.record_by_ref: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        self.bm25 = None
        self.client = self._client()
        self.embedder = self._embedder(force_reindex=force_reindex)
        self.dimension = self._validated_dimension()

        if force_reindex:
            self.build(recreate=True)
        else:
            self._load_or_build()

    def _client(self):
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:
            raise RuntimeError("Qdrant backend requires qdrant-client. Install requirements first.") from exc
        return QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key or None,
            timeout=self.config.qdrant_timeout,
        )

    def _embedder(self, *, force_reindex: bool):
        return make_embedder(self.embedding_model_name)

    def _validated_dimension(self) -> int:
        actual = int(self.embedder.get_embedding_dimension())
        expected = int(getattr(self.config, "embedding_dimension", 0) or 0)
        if expected and actual != expected:
            raise RuntimeError(
                "Embedding dimension mismatch: "
                f"configured RAG_EMBEDDING_DIMENSION={expected}, "
                f"but {type(self.embedder).__name__} returned {actual}."
            )
        return actual

    def _index_metadata(self) -> dict[str, Any]:
        return {
            "rag_index_version": QDRANT_INDEX_VERSION,
            "rag_embedding_backend": os.getenv("RAG_EMBEDDING_BACKEND", getattr(self.config, "embedding_backend", "openvino")),
            "rag_embedding_model": self.embedding_model_name,
            "rag_embedding_dimension": self.dimension,
        }

    def _load_or_build(self) -> None:
        source_records = load_expanded_records(self.processed_path)
        if not source_records:
            raise ValueError(f"No legal records found at {self.processed_path}")

        expected_points = len({_point_id(record) for record in source_records})
        exists = self.client.collection_exists(self.config.qdrant_collection)
        point_count = self._collection_point_count() if exists else 0
        vector_size = self._collection_vector_size() if exists else 0
        if exists and vector_size and vector_size != self.dimension:
            logger.warning(
                "Qdrant collection %s has vector size %s, expected %s. Recreating collection.",
                self.config.qdrant_collection,
                vector_size,
                self.dimension,
            )
            self.build(recreate=True, source_records=source_records)
            return
        if exists and point_count and not self._collection_index_matches():
            logger.warning(
                "Qdrant collection %s was built with stale/missing embedding metadata. Recreating collection.",
                self.config.qdrant_collection,
            )
            self.build(recreate=True, source_records=source_records)
            return
        if exists and point_count == expected_points:
            self.records = source_records
            self.documents = [record.get("rag_text", "") or "" for record in self.records]
            self._build_bm25()
            self._build_lookup_maps()
            logger.info(
                "Using Qdrant collection %s with %s points; loaded %s local records for deterministic indexes.",
                self.config.qdrant_collection,
                point_count,
                len(self.records),
            )
            return
        if exists and point_count > expected_points:
            logger.warning(
                "Qdrant collection %s has %s points, expected %s. Recreating collection.",
                self.config.qdrant_collection,
                point_count,
                expected_points,
            )
            self.build(recreate=True, source_records=source_records)
            return

        if exists and point_count:
            logger.warning(
                "Qdrant collection %s is incomplete: %s/%s points. Resuming indexing.",
                self.config.qdrant_collection,
                point_count,
                expected_points,
            )
        self.build(recreate=False, source_records=source_records)

    def build(self, *, recreate: bool = True, source_records: list[dict[str, Any]] | None = None) -> None:
        self.records = source_records if source_records is not None else load_expanded_records(self.processed_path)
        self.documents = [record.get("rag_text", "") or "" for record in self.records]
        if not self.records:
            raise ValueError(f"No legal records found at {self.processed_path}")

        self._ensure_collection(recreate=recreate)
        logger.info("Indexing %s expanded records into Qdrant collection %s.", len(self.records), self.config.qdrant_collection)

        from qdrant_client import models

        embedding_batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "16"))
        upsert_batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "32"))
        batch_size = max(1, min(embedding_batch_size, upsert_batch_size))
        existing_ids = self._existing_point_ids() if not recreate else set()
        if existing_ids:
            logger.info("Skipping %s records already present in Qdrant.", len(existing_ids))
        for start in range(0, len(self.records), batch_size):
            batch_records = [
                record
                for record in self.records[start : start + batch_size]
                if _point_id(record) not in existing_ids
            ]
            if not batch_records:
                continue
            batch_texts = [_embedding_text(record) for record in batch_records]
            vectors = self._encode_batch(batch_texts, batch_records=batch_records, batch_size=batch_size)
            points = []
            for record, vector in zip(batch_records, vectors):
                payload = {
                    "record": _clean_payload(record),
                    "text": _clean_payload(record.get("rag_text", "")),
                    **self._index_metadata(),
                    **_clean_payload(record.get("rag_metadata") or {}),
                }
                points.append(models.PointStruct(id=_point_id(record), vector=vector.tolist(), payload=payload))
            self.client.upsert(collection_name=self.config.qdrant_collection, points=points, wait=True)
            indexed = min(start + batch_size, len(self.records))
            if indexed == len(self.records) or indexed % max(batch_size * 20, 200) == 0:
                logger.info("Qdrant indexing progress: %s/%s", indexed, len(self.records))
        self._build_lookup_maps()
        self._build_bm25()

    def _encode_batch(self, texts: list[str], *, batch_records: list[dict[str, Any]], batch_size: int):
        try:
            return self.embedder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")
        except IndexError as exc:
            logger.warning(
                "Embedding batch failed with IndexError; retrying one record at a time to isolate bad input: %s",
                exc,
            )
            vectors = []
            for text, record in zip(texts, batch_records):
                try:
                    vector = self.embedder.encode(
                        [text],
                        batch_size=1,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    ).astype("float32")[0]
                    vectors.append(vector)
                except IndexError as item_exc:
                    ref = normalized_legal_reference(record)
                    raise RuntimeError(
                        "Embedding failed for record "
                        f"{record.get('source_chunk_id') or record.get('id')} "
                        f"at {ref.get('document')} Điều {ref.get('article')} "
                        f"Khoản {ref.get('clause')} Điểm {ref.get('point')}. "
                        "Check RAG_EMBEDDING_MAX_LENGTH and model tokenizer limits."
                    ) from item_exc
            import numpy as np

            return np.vstack(vectors).astype("float32")

    def _collection_point_count(self) -> int:
        try:
            info = self.client.get_collection(self.config.qdrant_collection)
            return int(info.points_count or 0)
        except Exception:
            return 0

    def _collection_vector_size(self) -> int:
        try:
            info = self.client.get_collection(self.config.qdrant_collection)
            config = getattr(info, "config", None)
            params = getattr(config, "params", None)
            vectors = getattr(params, "vectors", None)
            if isinstance(vectors, dict):
                vectors = next(iter(vectors.values()), None)
            if isinstance(vectors, dict):
                return int(vectors.get("size") or 0)
            return int(getattr(vectors, "size", 0) or 0)
        except Exception:
            return 0

    def _collection_index_matches(self) -> bool:
        try:
            points, _next_offset = self.client.scroll(
                collection_name=self.config.qdrant_collection,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return False
        if not points:
            return True
        payload = points[0].payload or {}
        expected = self._index_metadata()
        return all(payload.get(key) == value for key, value in expected.items())

    def _existing_point_ids(self) -> set[str]:
        ids: set[str] = set()
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.config.qdrant_collection,
                limit=1024,
                offset=next_offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(point.id) for point in points)
            if next_offset is None:
                break
        return ids

    def load(self) -> None:
        self._ensure_collection(recreate=False)
        self.records = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.config.qdrant_collection,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                record = payload.get("record")
                if isinstance(record, dict):
                    self.records.append(record)
            if next_offset is None:
                break
        self.documents = [record.get("rag_text", "") for record in self.records]
        self._build_bm25()
        self._build_lookup_maps()
        logger.info("Loaded %s records from Qdrant collection %s.", len(self.records), self.config.qdrant_collection)

    def search(self, query: str, top_k: int = 20, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0].tolist()
        query_filter = self._query_filter(filters)
        vector_limit = min(max(top_k * int(os.getenv("RAG_QDRANT_VECTOR_MULTIPLIER", "4")), top_k), max(len(self.records), top_k))
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.config.qdrant_collection,
                query_vector=query_vec,
                limit=vector_limit,
                with_payload=True,
                query_filter=query_filter,
            )
        else:
            response = self.client.query_points(
                collection_name=self.config.qdrant_collection,
                query=query_vec,
                limit=vector_limit,
                with_payload=True,
                query_filter=query_filter,
            )
            hits = response.points
        by_key: dict[str, dict[str, Any]] = {}
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            record = dict(payload.get("record") or {})
            if not record:
                continue
            record["retrieval_score"] = float(getattr(hit, "score", 0.0) or 0.0) + 1.0 / (60.0 + rank)
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["qdrant_vector"]))
            by_key[_point_id(record)] = record

        if os.getenv("RAG_QDRANT_ENABLE_LEXICAL", "true").lower() in {"1", "true", "yes", "on"}:
            for rank, record in enumerate(self._bm25_search(query, top_k=max(top_k * 4, top_k), filters=filters), start=1):
                key = _point_id(record)
                lexical_score = float(record.get("retrieval_score") or 0.0) + 1.0 / (60.0 + rank)
                existing = by_key.get(key)
                if existing:
                    existing["retrieval_score"] = float(existing.get("retrieval_score") or 0.0) + lexical_score
                    existing["retrieval_reasons"] = sorted(set(existing.get("retrieval_reasons", []) + ["bm25"]))
                else:
                    item = dict(record)
                    item["retrieval_score"] = lexical_score
                    item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["bm25"]))
                    by_key[key] = item

        return sorted(by_key.values(), key=lambda r: float(r.get("retrieval_score") or 0), reverse=True)[:top_k]

    def _build_bm25(self) -> None:
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.documents]) if self.documents else None

    def _bm25_search(self, query: str, *, top_k: int, filters: dict[str, Any]) -> list[dict[str, Any]]:
        if self.bm25 is None or not self.records:
            return []
        import numpy as np

        scores = self.bm25.get_scores(tokenize(query))
        max_score = float(max(scores)) if len(scores) else 0.0
        if max_score <= 0:
            return []
        ranked = np.argsort(scores)[::-1][: max(top_k * 3, top_k)]
        out: list[dict[str, Any]] = []
        for idx in ranked:
            record = self.records[int(idx)]
            if filters and not self._matches_filters(record, filters):
                continue
            item = dict(record)
            item["retrieval_score"] = 0.35 * (float(scores[idx]) / (max_score + 1e-9))
            item["retrieval_reasons"] = sorted(set(item.get("retrieval_reasons", []) + ["bm25"]))
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def _query_filter(self, filters: dict[str, Any]):
        if not filters:
            return None
        from qdrant_client import models

        must = []
        documents = filters.get("documents") or []
        if filters.get("qcvn") and not documents:
            documents = ["QCVN 41:2024 (Thông tư 51/2024)"]
        if documents:
            must.append(models.FieldCondition(key="doc", match=models.MatchAny(any=list(documents))))
        modalities = filters.get("modalities") or []
        if modalities:
            must.append(models.FieldCondition(key="modality", match=models.MatchAny(any=list(modalities))))
        for field in ["has_table", "has_sign", "has_penalty", "has_procedure"]:
            if filters.get(field) is True:
                must.append(models.FieldCondition(key=field, match=models.MatchValue(value=True)))
        if not must:
            return None
        return models.Filter(must=must)

    def _matches_filters(self, record: dict[str, Any], filters: dict[str, Any]) -> bool:
        meta = record.get("rag_metadata") or {}
        documents = set(filters.get("documents") or [])
        if filters.get("qcvn") and not documents:
            documents.add("QCVN 41:2024 (Thông tư 51/2024)")
        if documents:
            doc = meta.get("doc") or record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or ""
            if doc not in documents:
                return False
        modalities = set(filters.get("modalities") or [])
        if modalities and record.get("rag_modality", "text") not in modalities:
            return False
        for field in ["has_table", "has_sign", "has_penalty", "has_procedure"]:
            if filters.get(field) is True and not meta.get(field):
                return False
        return True

    def by_source_chunk_ids(self, source_chunk_ids: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for source_id in source_chunk_ids:
            records.extend(dict(record) for record in self.record_by_source_chunk.get(source_id, []))
        return records

    def by_ref(self, document: str, article: str, clause: str = "", point: str = "") -> list[dict[str, Any]]:
        keys = []
        if point:
            keys.append((document, article, clause, point))
        if clause:
            keys.append((document, article, clause, ""))
        if article:
            keys.append((document, article, "", ""))
        records: list[dict[str, Any]] = []
        for key in keys:
            records.extend(dict(record) for record in self.record_by_ref.get(key, []))
        return records

    def _ensure_collection(self, *, recreate: bool) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.config.qdrant_collection)
        if exists and recreate:
            self.client.delete_collection(self.config.qdrant_collection)
            exists = False
        if exists:
            vector_size = self._collection_vector_size()
            if vector_size and vector_size != self.dimension:
                self.client.delete_collection(self.config.qdrant_collection)
                exists = False
            else:
                return
        if exists:
            return
        self.client.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
        )
        for field in ["doc", "article", "clause", "point", "modality", "has_table", "has_sign", "has_penalty", "has_procedure"]:
            try:
                schema = models.PayloadSchemaType.BOOL if field.startswith("has_") else models.PayloadSchemaType.KEYWORD
                self.client.create_payload_index(
                    collection_name=self.config.qdrant_collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                logger.debug("Qdrant payload index already exists or is unsupported: %s", field)

    def _build_lookup_maps(self) -> None:
        self.record_by_source_chunk = {}
        self.record_by_ref = {}
        for record in self.records:
            source_id = record.get("source_chunk_id")
            if source_id:
                self.record_by_source_chunk.setdefault(source_id, []).append(record)
            ref = normalized_legal_reference(record)
            key = (
                ref.get("document") or record.get("doc_name") or "",
                str(ref.get("article") or ""),
                str(ref.get("clause") or ""),
                str(ref.get("point") or ""),
            )
            self.record_by_ref.setdefault(key, []).append(record)
