import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from src.rag.legal_utils import normalized_legal_reference
from src.rag.rag_store_config import RAGStoreConfig
from src.rag.record_expander import load_expanded_records
from src.rag.embedding_backends import make_embedder


logger = logging.getLogger("QdrantLegalVectorStore")


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
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
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
        self.client = self._client()
        self.embedder = self._embedder(force_reindex=force_reindex)
        self.dimension = self.embedder.get_embedding_dimension()

        if force_reindex:
            self.build()
        else:
            self.load()
            if not self.records:
                self.build()

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

    def build(self) -> None:
        self.records = load_expanded_records(self.processed_path)
        self.documents = [record.get("rag_text", "") for record in self.records]
        if not self.records:
            raise ValueError(f"No legal records found at {self.processed_path}")

        self._ensure_collection(recreate=True)
        logger.info("Indexing %s expanded records into Qdrant collection %s.", len(self.records), self.config.qdrant_collection)
        import numpy as np

        cache_dir = Path(os.getenv("QDRANT_EMBEDDINGS_CACHE_DIR", "data/vector_db/qdrant_cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_key = hashlib.sha256(self.embedding_model_name.encode("utf-8")).hexdigest()[:12]
        cache_path = cache_dir / f"{self.config.qdrant_collection}_{model_key}_{len(self.records)}x{self.dimension}.npy"
        if cache_path.exists() and os.getenv("QDRANT_REUSE_EMBEDDINGS_CACHE", "true").lower() in {"1", "true", "yes", "on"}:
            logger.info("Loading cached Qdrant embeddings from %s.", cache_path)
            vectors = np.load(cache_path).astype("float32")
        else:
            vectors = self.embedder.encode(
                self.documents,
                batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")),
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype("float32")
            np.save(cache_path, vectors)
            logger.info("Saved Qdrant embeddings cache to %s.", cache_path)

        from qdrant_client import models

        batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128"))
        for start in range(0, len(self.records), batch_size):
            points = []
            for record, vector in zip(self.records[start : start + batch_size], vectors[start : start + batch_size]):
                payload = {
                    "record": _clean_payload(record),
                    "text": _clean_payload(record.get("rag_text", "")),
                    **_clean_payload(record.get("rag_metadata") or {}),
                }
                points.append(models.PointStruct(id=_point_id(record), vector=vector.tolist(), payload=payload))
            self.client.upsert(collection_name=self.config.qdrant_collection, points=points, wait=True)
        self._build_lookup_maps()

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
        self._build_lookup_maps()
        logger.info("Loaded %s records from Qdrant collection %s.", len(self.records), self.config.qdrant_collection)

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")[0].tolist()
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.config.qdrant_collection,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.config.qdrant_collection,
                query=query_vec,
                limit=top_k,
                with_payload=True,
            )
            hits = response.points
        results: list[dict[str, Any]] = []
        for hit in hits:
            payload = hit.payload or {}
            record = dict(payload.get("record") or {})
            if not record:
                continue
            record["retrieval_score"] = float(getattr(hit, "score", 0.0) or 0.0)
            record["retrieval_reasons"] = sorted(set(record.get("retrieval_reasons", []) + ["qdrant_vector"]))
            results.append(record)
        return results

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
            return
        self.client.create_collection(
            collection_name=self.config.qdrant_collection,
            vectors_config=models.VectorParams(size=self.dimension, distance=models.Distance.COSINE),
        )
        for field in ["doc", "article", "clause", "point", "modality", "has_table", "has_sign", "has_penalty"]:
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
