import logging
import os
import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.rag.legal_utils import tokenize
from src.rag.record_expander import expand_record, load_processed_records
from src.rag.embedding_backends import make_embedder


logger = logging.getLogger("HybridLegalVectorStore")
HYBRID_VECTOR_INDEX_VERSION = "legal_graph_rag_vector_v3_source_id_ref_repair"


class HybridLegalVectorStore:
    """FAISS + BM25 vector DB for legal records.

    This is the local vector DB implementation. It keeps the public API small so
    Qdrant/Weaviate/Neo4j-vector can replace it later without changing retrieval.
    """

    def __init__(
        self,
        processed_path: str | Path = "data/processed",
        index_dir: str | Path = "data/vector_db/legal_graph_rag",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        force_reindex: bool = False,
    ):
        self.processed_path = Path(processed_path)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.dimension = 0
        enable_embeddings = os.getenv("RAG_ENABLE_EMBEDDINGS", "").lower() in {"1", "true", "yes", "on"}
        allow_model_download = os.getenv("RAG_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes", "on"}
        local_model_path = Path(embedding_model).expanduser()
        should_load_embedder = bool(
            local_model_path.exists()
            or (enable_embeddings and (force_reindex or allow_model_download))
        )
        if should_load_embedder:
            try:
                self.embedder = make_embedder(str(local_model_path) if local_model_path.exists() else embedding_model)
                self.dimension = self.embedder.get_embedding_dimension()
            except Exception as exc:
                logger.warning("Embedding model disabled; BM25 + graph retrieval will still work: %s", exc)
        else:
            logger.info("Embedding model skipped. Set RAG_ENABLE_EMBEDDINGS=true to allow model loading.")
        self.records: list[dict[str, Any]] = []
        self.documents: list[str] = []
        self.record_by_source_chunk: dict[str, list[dict[str, Any]]] = {}
        self.record_by_ref: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        self.index = None
        self.bm25 = None

        if not force_reindex and self._has_compatible_index():
            self.load()
        elif not force_reindex and self._has_loadable_bm25_index():
            logger.warning(
                "Compatible semantic index is not available; loading existing BM25 index. "
                "Run with force_reindex=True after model download to build FAISS embeddings."
            )
            self.embedder = None
            self.dimension = 0
            self.load()
        else:
            self.build()

    @property
    def faiss_path(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25.pkl"

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "metadata.pkl"

    def _has_compatible_index(self) -> bool:
        if not (self.bm25_path.exists() and self.metadata_path.exists()):
            return False
        if self.embedder is not None and not self.faiss_path.exists():
            return False
        try:
            with self.metadata_path.open("rb") as f:
                meta = pickle.load(f)
            return (
                meta.get("index_version") == HYBRID_VECTOR_INDEX_VERSION
                and meta.get("embedding_model") == (self.embedding_model_name if self.embedder is not None else "bm25_only")
            )
        except Exception:
            return False

    def _has_loadable_bm25_index(self) -> bool:
        if not (self.bm25_path.exists() and self.metadata_path.exists()):
            return False
        try:
            with self.metadata_path.open("rb") as f:
                meta = pickle.load(f)
            return bool(meta.get("records") and meta.get("documents"))
        except Exception:
            return False

    def _load_processed_records(self) -> list[dict[str, Any]]:
        return load_processed_records(self.processed_path)

    def _expand_record(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        return expand_record(record)

    def build(self) -> None:
        raw_records = self._load_processed_records()
        self.records = []
        for record in raw_records:
            self.records.extend(self._expand_record(record))
        self.documents = [r.get("rag_text", "") for r in self.records]
        if not self.documents:
            raise ValueError(f"No legal records found at {self.processed_path}")

        logger.info("Building hybrid vector DB for %s expanded records.", len(self.records))
        if self.embedder is not None:
            import faiss

            embeddings = self.embedder.encode(
                self.documents,
                batch_size=32,
                convert_to_numpy=True,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype("float32")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
        else:
            self.index = None
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])
        self._build_lookup_maps()
        self.save()

    def _build_lookup_maps(self) -> None:
        self.record_by_source_chunk = {}
        self.record_by_ref = {}
        for record in self.records:
            source_id = record.get("source_chunk_id")
            if source_id:
                self.record_by_source_chunk.setdefault(source_id, []).append(record)
            ref = record.get("legal_reference") or {}
            key = (
                ref.get("document") or record.get("doc_name") or "",
                str(ref.get("article") or ""),
                str(ref.get("clause") or ""),
                str(ref.get("point") or ""),
            )
            self.record_by_ref.setdefault(key, []).append(record)

    def save(self) -> None:
        if self.index is not None:
            import faiss

            faiss.write_index(self.index, str(self.faiss_path))
        with self.bm25_path.open("wb") as f:
            pickle.dump(self.bm25, f)
        with self.metadata_path.open("wb") as f:
            pickle.dump(
                {
                    "index_version": HYBRID_VECTOR_INDEX_VERSION,
                    "embedding_model": self.embedding_model_name if self.embedder is not None else "bm25_only",
                    "records": self.records,
                    "documents": self.documents,
                },
                f,
            )

    def load(self) -> None:
        logger.info("Loading hybrid vector DB from %s", self.index_dir)
        if self.embedder is not None and self.faiss_path.exists():
            import faiss

            self.index = faiss.read_index(str(self.faiss_path))
        else:
            self.index = None
        with self.bm25_path.open("rb") as f:
            self.bm25 = pickle.load(f)
        with self.metadata_path.open("rb") as f:
            meta = pickle.load(f)
        self.records = meta["records"]
        self.documents = meta["documents"]
        self._build_lookup_maps()

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        bm25_scores = self.bm25.get_scores(tokenize(query))
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) else 0.0

        scores: dict[int, float] = {}
        reasons: dict[int, list[str]] = {}
        if self.embedder is not None and self.index is not None:
            query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            faiss_scores, faiss_idx = self.index.search(query_vec, min(max(top_k * 4, top_k), len(self.records)))
            for rank, idx in enumerate(faiss_idx[0]):
                if idx == -1:
                    continue
                scores[idx] = max(scores.get(idx, 0.0), 0.65 * float(faiss_scores[0][rank]))
                reasons.setdefault(idx, []).append("vector")

        if max_bm25 > 0:
            import numpy as np

            top_bm25 = np.argsort(bm25_scores)[::-1][: max(top_k * 4, top_k)]
            for idx in top_bm25:
                score = 0.35 * (float(bm25_scores[idx]) / (max_bm25 + 1e-9))
                scores[int(idx)] = scores.get(int(idx), 0.0) + score
                reasons.setdefault(int(idx), []).append("bm25")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            record = dict(self.records[idx])
            record["retrieval_score"] = score
            record["retrieval_reasons"] = sorted(set(reasons.get(idx, [])))
            results.append(record)
        return results

    def by_source_chunk_ids(self, source_chunk_ids: list[str]) -> list[dict[str, Any]]:
        records = []
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
        records = []
        for key in keys:
            records.extend(dict(record) for record in self.record_by_ref.get(key, []))
        return records
