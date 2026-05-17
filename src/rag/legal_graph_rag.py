import logging
import os
import time
from pathlib import Path
from typing import Any

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag.custom_legal_retriever import CustomLegalRetriever
from src.rag.hybrid_vector_store import HybridLegalVectorStore
from src.rag.legal_graph_store import DeterministicLegalGraphStore
from src.rag.legal_utils import format_reference
from src.rag.rag_store_config import RAGStoreConfig


logger = logging.getLogger("LegalGraphRAG")
load_dotenv()


class LegalGraphRAG:
    """LlamaIndex-style orchestration around vector DB + deterministic graph DB.

    The current implementation uses local FAISS/BM25 and graph JSON stores. The
    retriever/orchestrator boundaries mirror LlamaIndex components, so the same
    flow can be moved to PropertyGraphIndex + Neo4j/Qdrant when infrastructure is
    ready.
    """

    def __init__(
        self,
        processed_path: str | Path = "data/processed",
        graph_path: str | Path = "data/graph/legal_graph.json",
        index_dir: str | Path = "data/vector_db/legal_graph_rag",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        force_reindex: bool = False,
        use_reranker: bool = True,
    ):
        self.project_root = Path(__file__).resolve().parents[2]
        self.config = RAGStoreConfig()
        embedding_model = os.getenv("RAG_EMBEDDING_MODEL", embedding_model)
        self.graph_store = self._build_graph_store(graph_path)
        self.vector_store = self._build_vector_store(
            processed_path=processed_path,
            index_dir=index_dir,
            embedding_model=embedding_model,
            force_reindex=force_reindex,
        )
        self.retriever = CustomLegalRetriever(
            self.vector_store,
            self.graph_store,
            use_reranker=use_reranker,
        )
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None

    def _build_graph_store(self, graph_path: str | Path):
        if self.config.graph_backend == "neo4j":
            from src.rag.neo4j_graph_store import Neo4jLegalGraphStore

            return Neo4jLegalGraphStore(self.config)
        return DeterministicLegalGraphStore(graph_path)

    def _build_vector_store(
        self,
        *,
        processed_path: str | Path,
        index_dir: str | Path,
        embedding_model: str,
        force_reindex: bool,
    ):
        if self.config.vector_backend == "qdrant":
            from src.rag.qdrant_vector_store import QdrantLegalVectorStore

            return QdrantLegalVectorStore(
                processed_path=processed_path,
                embedding_model=embedding_model,
                force_reindex=force_reindex,
                config=self.config,
            )
        return HybridLegalVectorStore(
            processed_path=processed_path,
            index_dir=index_dir,
            embedding_model=embedding_model,
            force_reindex=force_reindex,
        )

    def retrieve(self, query: str, top_k: int = 8, expand_depth: int = 2) -> list[dict[str, Any]]:
        return self.retriever.retrieve(query, top_k=top_k, expand_depth=expand_depth)

    def retrieve_sign(self, query: str, top_k: int = 8, expand_depth: int = 1) -> list[dict[str, Any]]:
        return self.retriever.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth)

    def retrieve_table(self, query: str, top_k: int = 8, expand_depth: int = 1) -> list[dict[str, Any]]:
        return self.retriever.retrieve_table(query, top_k=top_k, expand_depth=expand_depth)

    def query(self, query: str, top_k: int = 8, expand_depth: int = 2) -> dict[str, Any]:
        contexts = self.retrieve(query, top_k=top_k, expand_depth=expand_depth)
        answer = self.generate_answer(query, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
            "references": self.format_references(contexts),
            "images": sorted({self._public_asset_path(ctx.get("image_path")) for ctx in contexts if ctx.get("image_path")}),
        }

    def query_sign(self, query: str, top_k: int = 8) -> dict[str, Any]:
        contexts = self.retrieve_sign(query, top_k=top_k)
        answer = self.generate_answer(query, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
            "references": self.format_references(contexts),
            "images": sorted({self._public_asset_path(ctx.get("image_path")) for ctx in contexts if ctx.get("image_path")}),
        }

    def query_table(self, query: str, top_k: int = 8) -> dict[str, Any]:
        contexts = self.retrieve_table(query, top_k=top_k)
        answer = self.generate_answer(query, contexts)
        return {
            "answer": answer,
            "contexts": contexts,
            "references": self.format_references(contexts),
            "images": sorted({self._public_asset_path(ctx.get("image_path")) for ctx in contexts if ctx.get("image_path")}),
        }

    def generate_answer(self, query: str, contexts: list[dict[str, Any]]) -> str:
        if not contexts:
            return "Tôi chưa tìm thấy căn cứ phù hợp trong dữ liệu luật giao thông đã trích xuất."
        if self.client is None:
            return self._extractive_answer(query, contexts)

        contents: list[Any] = [self._system_prompt()]
        for idx, ctx in enumerate(contexts, start=1):
            ref = format_reference(ctx)
            reasons = ", ".join(ctx.get("retrieval_reasons") or [])
            text = (ctx.get("rag_text") or ctx.get("source_body_exact") or ctx.get("content") or "")[:4000]
            contents.append(
                f"\n[NGUỒN {idx}]\n"
                f"Căn cứ: {ref}\n"
                f"source_chunk_id: {ctx.get('source_chunk_id')}\n"
                f"modality: {ctx.get('rag_modality', 'text')}; retrieval: {reasons}\n"
                f"Nội dung:\n{text}\n"
            )
            image = self._load_image(ctx.get("image_path"))
            if image is not None:
                contents.append(image)

        contents.append(f"\n[CÂU HỎI]\n{query}\n")
        contents.append(self._answer_requirements())
        models_to_try = self._answer_models()
        last_exc = None
        for model in models_to_try:
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=4096,
                            safety_settings=[
                                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                            ],
                        ),
                    )
                    return response.text or self._extractive_answer(query, contexts)
                except Exception as exc:
                    last_exc = exc
                    if self._is_quota_exhausted(exc):
                        logger.warning("Gemini model %s hit quota/rate limit; falling back to next model: %s", model, exc)
                        break
                    if self._is_retryable_error(exc) and attempt < 2:
                        wait_sec = (attempt + 1) * 2
                        logger.warning("Gemini API error on %s (attempt %s/3): %s. Retrying in %ss...", model, attempt + 1, exc, wait_sec)
                        time.sleep(wait_sec)
                        continue
                    break

        logger.warning("RAG answer generation failed after retries: %s", last_exc)
        return self._extractive_answer(query, contexts)

    def _answer_models(self) -> list[str]:
        primary = os.getenv("RAG_ANSWER_MODEL", os.getenv("QA_PRIMARY_MODEL", "gemini-3.1-flash-lite")).strip()
        fallback_env = os.getenv("RAG_ANSWER_FALLBACK_MODELS", "gemma-4-31b-it")
        fallbacks = [model.strip() for model in fallback_env.split(",") if model.strip()]
        models = [primary, *fallbacks]
        return list(dict.fromkeys(model for model in models if model))

    def _is_quota_exhausted(self, exc: Exception) -> bool:
        err_str = str(exc).lower()
        return any(
            token in err_str
            for token in [
                "429",
                "resource_exhausted",
                "quota",
                "rate limit",
                "rate_limit",
                "too many requests",
            ]
        )

    def _is_retryable_error(self, exc: Exception) -> bool:
        err_str = str(exc).lower()
        return any(
            token in err_str
            for token in [
                "502",
                "503",
                "unavailable",
                "overloaded",
                "deadline_exceeded",
                "timeout",
            ]
        )

    def format_references(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        refs = []
        for ctx in contexts:
            refs.append(
                {
                    "source_chunk_id": ctx.get("source_chunk_id"),
                    "record_id": ctx.get("id"),
                    "modality": ctx.get("rag_modality", "text"),
                    "legal_reference": ctx.get("legal_reference"),
                    "reference_text": format_reference(ctx),
                    "image_path": ctx.get("image_path"),
                    "rag_metadata": ctx.get("rag_metadata"),
                    "retrieval_reasons": ctx.get("retrieval_reasons", []),
                    "retrieval_score": ctx.get("retrieval_score"),
                }
            )
        return refs

    def _system_prompt(self) -> str:
        return (
            "Bạn là hệ thống Legal Graph RAG cho luật giao thông Việt Nam. "
            "Chỉ trả lời dựa trên NGUỒN được cung cấp. Khi một nguồn có dẫn chiếu "
            "sang điều/khoản khác và nguồn liên quan đã được cung cấp, phải giải "
            "thích luôn nội dung liên quan, không được trả lời úp mở kiểu 'xem Điều X'."
        )

    def _answer_requirements(self) -> str:
        return (
            "\n[YÊU CẦU TRẢ LỜI]\n"
            "- Trả lời trực tiếp, đầy đủ các điều kiện, ngoại lệ, mức phạt/thời hạn nếu có.\n"
            "- Với Điều/Khoản/Điểm lồng nhau, nêu rõ quan hệ: điểm thuộc khoản nào, khoản thuộc điều nào.\n"
            "- Nếu có bảng hoặc ảnh/biển báo, giải thích ý nghĩa và nói rõ ảnh/bảng nào là căn cứ.\n"
            "- Mọi con số pháp lý phải lấy từ nguồn, không suy đoán.\n"
            "- Cuối câu trả lời có mục 'Căn cứ' liệt kê nguồn đã dùng theo Điều/Khoản/Điểm và văn bản.\n"
        )

    def _extractive_answer(self, query: str, contexts: list[dict[str, Any]]) -> str:
        parts = ["Tôi tìm thấy các căn cứ liên quan sau:"]
        for idx, ctx in enumerate(contexts[:5], start=1):
            text = (ctx.get("rag_text") or ctx.get("source_body_exact") or ctx.get("content") or "").strip()
            parts.append(f"{idx}. {format_reference(ctx)}: {text[:700]}")
        parts.append("Cần dùng các căn cứ trên để tổng hợp câu trả lời chi tiết và không thêm thông tin ngoài nguồn.")
        return "\n\n".join(parts)

    def _load_image(self, path: str | None):
        if not path:
            return None
        img_path = Path(path)
        if not img_path.is_absolute():
            img_path = self.project_root / img_path
        if not img_path.exists():
            return None
        try:
            return PIL.Image.open(img_path)
        except Exception:
            return None

    def _public_asset_path(self, path: str) -> str:
        if not path:
            return ""
        normalized = path.replace("\\", "/")
        if normalized.startswith("data/processed/"):
            return "/processed/" + normalized[len("data/processed/"):]
        return "/processed/" + normalized.split("data/processed/")[-1]
