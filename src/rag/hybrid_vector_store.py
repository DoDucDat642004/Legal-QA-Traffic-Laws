import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

from src.rag.legal_utils import normalized_legal_reference, tokenize
from src.rag.record_expander import expand_record, load_processed_records
from src.rag.embedding_backends import make_embedder
from src.rag.rag_store_config import DEFAULT_EMBEDDING_MODEL


logger = logging.getLogger("HybridLegalVectorStore")

# Versioning for the index to handle schema changes or re-indexing needs
HYBRID_VECTOR_INDEX_VERSION = "legal_graph_rag_vector_v6_asset_scoped_qcvn_sign_code_guard"


class HybridLegalVectorStore:
    """
    Local implementation of a hybrid vector database combining FAISS (semantic) and BM25 (keyword) search.
    Designed for legal records to handle both conceptual queries and exact term matching.
    """

    def __init__(
        self,
        processed_path: Union[str, Path] = "data/processed",
        index_dir: Union[str, Path] = "data/vector_db/legal_graph_rag",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        force_reindex: bool = False,
    ):
        """
        Initializes the hybrid store. Loads existing indices or builds them if needed.

        Args:
            processed_path: Directory containing the processed legal records (JSONL).
            index_dir: Directory where the FAISS and BM25 indices are persisted.
            embedding_model: The identifier of the model used for generating embeddings.
            force_reindex: If True, bypasses loading and rebuilds indices from scratch.
        """
        self.processed_path = Path(processed_path)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model
        
        # State initialization
        self.embedder = None
        self.dimension = 0
        self.records: List[Dict[str, Any]] = []
        self.documents: List[str] = []
        self.record_by_source_chunk: Dict[str, List[Dict[str, Any]]] = {}
        self.record_by_ref: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
        self.index: Any = None
        self.bm25: Optional[BM25Okapi] = None

        # Determine if we should attempt to load or download the embedding model
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
                logger.warning("Embedding model failed to load; falling back to BM25-only: %s", exc)
        else:
            logger.info("Embedding model loading skipped by configuration.")

        # Main loading/building logic
        if not force_reindex and self._has_compatible_index():
            self.load()
        elif not force_reindex and self._has_loadable_bm25_index():
            logger.warning("Compatible semantic index not found; falling back to existing BM25 index.")
            self.embedder = None
            self.dimension = 0
            self.load()
        else:
            self.build()

    @property
    def faiss_path(self) -> Path:
        """Returns the path to the persisted FAISS index."""
        return self.index_dir / "index.faiss"

    @property
    def bm25_path(self) -> Path:
        """Returns the path to the persisted BM25 index."""
        return self.index_dir / "bm25.pkl"

    @property
    def metadata_path(self) -> Path:
        """Returns the path to the persisted metadata (records and documents)."""
        return self.index_dir / "metadata.pkl"

    def _has_compatible_index(self) -> bool:
        """Verifies if the persisted index matches the current version and model configuration."""
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
        """Checks if a valid BM25-only index is available for fallback."""
        if not (self.bm25_path.exists() and self.metadata_path.exists()):
            return False
        try:
            with self.metadata_path.open("rb") as f:
                meta = pickle.load(f)
            return bool(
                meta.get("index_version") == HYBRID_VECTOR_INDEX_VERSION
                and meta.get("embedding_model") == "bm25_only"
                and meta.get("records")
                and meta.get("documents")
            )
        except Exception:
            return False

    def build(self) -> None:
        """
        Builds the hybrid indices from processed records. 
        Expands records (splitting Articles into Clauses/Points) and indexes them.
        """
        raw_records = load_processed_records(self.processed_path)
        self.records = []
        for record in raw_records:
            self.records.extend(expand_record(record))
        
        # Clean up memory immediately
        del raw_records
        import gc
        gc.collect()

        self.documents = [r.get("rag_text", "") for r in self.records]
        if not self.documents:
            raise ValueError(f"No valid records found in {self.processed_path}")

        logger.info("Building hybrid index for %s legal records.", len(self.records))
        
        # Build Semantic Index (FAISS)
        if self.embedder is not None:
            import faiss
            batch_size = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32"))
            self.index = faiss.IndexFlatIP(self.dimension)

            # Chunked indexing to prevent OOM on large datasets
            chunk_size = 5000
            for i in range(0, len(self.documents), chunk_size):
                chunk_docs = self.documents[i : i + chunk_size]
                logger.info("Processing semantic batch %s-%s...", i, min(i + chunk_size, len(self.documents)))
                embeddings = self.embedder.encode(
                    chunk_docs,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=(len(self.documents) <= chunk_size),
                    normalize_embeddings=True,
                ).astype("float32")
                self.index.add(embeddings)
                del embeddings
                gc.collect()
        else:
            self.index = None

        # Build Keyword Index (BM25)
        self.bm25 = BM25Okapi([tokenize(doc) for doc in self.documents])
        
        self._build_lookup_maps()
        self.save()

    def _build_lookup_maps(self) -> None:
        """Internal helper to build fast lookup indices for ID-based and reference-based access."""
        self.record_by_source_chunk = {}
        self.record_by_ref = {}
        for record in self.records:
            # ID-based lookup
            source_id = record.get("source_chunk_id")
            if source_id:
                self.record_by_source_chunk.setdefault(source_id, []).append(record)
            
            # Legal reference based lookup (Doc, Article, Clause, Point)
            ref = normalized_legal_reference(record)
            key = (
                ref.get("document") or record.get("doc_name") or "",
                str(ref.get("article") or ""),
                str(ref.get("clause") or ""),
                str(ref.get("point") or ""),
            )
            self.record_by_ref.setdefault(key, []).append(record)

    def save(self) -> None:
        """Persists all indices and metadata to disk."""
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
        """Loads indices and metadata from the index directory."""
        logger.info("Loading hybrid index from %s", self.index_dir)
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

    def search(
        self, 
        query: str, 
        top_k: int = 20, 
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes a hybrid search combining Vector (Semantic) and BM25 (Keyword) logic.
        Uses a weighted fusion approach (65% semantic, 35% keyword).

        Args:
            query: The search string.
            top_k: Number of results to return after filtering.
            filters: Metadata filters (document names, modalities, etc.).

        Returns:
            A list of matching records with 'retrieval_score' and 'retrieval_reasons'.
        """
        # 1. Keyword search (BM25)
        bm25_scores = self.bm25.get_scores(tokenize(query))
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) else 0.0

        scores: Dict[int, float] = {}
        reasons: Dict[int, List[str]] = {}

        # 2. Semantic search (FAISS)
        if self.embedder is not None and self.index is not None:
            query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            # Increase initial K to allow for hybrid fusion and filtering
            faiss_scores, faiss_idx = self.index.search(query_vec, min(max(top_k * 4, 60), len(self.records)))
            
            for rank, idx in enumerate(faiss_idx[0]):
                if idx == -1: continue
                # Apply semantic weight
                scores[idx] = max(scores.get(idx, 0.0), 0.65 * float(faiss_scores[0][rank]))
                reasons.setdefault(idx, []).append("vector")

        # 3. Fuse BM25 results
        if max_bm25 > 0:
            import numpy as np
            top_bm25_indices = np.argsort(bm25_scores)[::-1][: max(top_k * 4, 60)]
            for idx in top_bm25_indices:
                # Normalize BM25 and apply keyword weight (0.35)
                score = 0.35 * (float(bm25_scores[idx]) / (max_bm25 + 1e-9))
                scores[int(idx)] = scores.get(int(idx), 0.0) + score
                reasons.setdefault(int(idx), []).append("bm25")

        # 4. Rank and hydrate records
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(top_k * 3, 100)]
        results = []
        for idx, score in ranked:
            record = dict(self.records[idx])
            record["retrieval_score"] = score
            record["retrieval_reasons"] = sorted(set(reasons.get(idx, [])))
            results.append(record)

        # 5. Apply metadata filters
        if not filters:
            return results[:top_k]

        filtered = [record for record in results if self._matches_filters(record, filters)]
        strict_filters = os.getenv("RAG_STRICT_FILTERS", "true").lower() in {"1", "true", "yes", "on"}
        
        return (filtered if strict_filters else (filtered or results))[:top_k]

    def _matches_filters(self, record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Internal helper to apply metadata constraints to a search result."""
        meta = record.get("rag_metadata") or {}
        
        # Document filtering
        documents = set(filters.get("documents") or [])
        if filters.get("qcvn") and not documents:
            documents.add("QCVN 41:2024 (Thông tư 51/2024)")
        
        if documents:
            doc = meta.get("doc") or record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or ""
            if doc not in documents:
                return False

        # Modality filtering (text, table, sign, figure)
        modalities = set(filters.get("modalities") or [])
        if modalities and record.get("rag_modality", "text") not in modalities:
            return False

        # Boolean capability flags
        for key in ["has_sign", "has_table", "has_penalty", "has_procedure"]:
            if filters.get(key) is True and not meta.get(key):
                return False

        # Implicit QCVN filtering
        if filters.get("qcvn"):
            doc = (record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "").lower()
            if "qcvn" not in doc and "thông tư 51" not in doc:
                return False
                
        return True

    def by_source_chunk_ids(self, source_chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Returns all records associated with a specific set of source chunk IDs."""
        records = []
        for source_id in source_chunk_ids:
            records.extend(dict(record) for record in self.record_by_source_chunk.get(source_id, []))
        return records

    def by_ref(
        self, 
        document: str, 
        article: str, 
        clause: str = "", 
        point: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Retrieves records by their exact legal coordinates.
        Supports hierarchical fallback (Point -> Clause -> Article).
        """
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

    def by_ref_prefix(
        self,
        document: str,
        article: str = "",
        clause: str = "",
        point: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all records under a legal coordinate prefix.

        `by_ref(document, "7")` intentionally returns the direct Article 7
        chunk. For retrieval, a question often needs every clause/point under
        Article 7, so this method performs a prefix scan over the lookup map.
        """
        records: List[Dict[str, Any]] = []
        seen = set()
        for (doc, art, cl, pt), values in self.record_by_ref.items():
            if document and doc != document:
                continue
            if article and art != str(article):
                continue
            if clause and cl != str(clause):
                continue
            if point and pt.lower() != str(point).lower():
                continue
            for record in values:
                rid = record.get("source_chunk_id") or record.get("id") or id(record)
                if rid in seen:
                    continue
                seen.add(rid)
                records.append(dict(record))
        return records
