"""
Main LlamaIndex-style Orchestrator for Legal Graph RAG.

This module coordinates vector storage, deterministic graph expansion, and LLM synthesis.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag.custom_legal_retriever import CustomLegalRetriever
from src.rag.hybrid_vector_store import HybridLegalVectorStore
from src.rag.legal_graph_store import DeterministicLegalGraphStore
from src.rag.legal_utils import format_reference, public_asset_path, record_image_paths, source_text
from src.rag.model_policy import generate_content_with_fallback
from src.rag.rag_store_config import RAGStoreConfig

# --- Global Logger & Environment ---
logger = logging.getLogger("LegalGraphRAG")
load_dotenv(override=False)


class LegalGraphRAG:
    """
    Orchestration layer managing the hybrid retrieval pipeline.
    """

    def __init__(
        self,
        processed_path: Union[str, Path] = "data/processed",
        graph_path: Union[str, Path] = "data/graph/legal_graph.json",
        index_dir: Union[str, Path] = "data/vector_db/legal_graph_rag",
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
        self.retriever.client = self.client # Inject client for internal probes

    def _build_graph_store(self, graph_path: Union[str, Path]):
        if self.config.graph_backend == "neo4j":
            try:
                from src.rag.neo4j_graph_store import Neo4jLegalGraphStore
                return Neo4jLegalGraphStore(self.config)
            except Exception as exc:
                strict = os.getenv("RAG_STRICT_GRAPH_BACKEND", "false").lower() in {"1", "true", "yes", "on"}
                if strict:
                    raise
                logger.warning("Neo4j backend unavailable; falling back to local graph JSON: %s", exc)
        return DeterministicLegalGraphStore(graph_path)

    def _build_vector_store(
        self,
        *,
        processed_path: Union[str, Path],
        index_dir: Union[str, Path],
        embedding_model: str,
        force_reindex: bool,
    ):
        if self.config.vector_backend == "qdrant":
            try:
                from src.rag.qdrant_vector_store import QdrantLegalVectorStore
                return QdrantLegalVectorStore(
                    processed_path=processed_path,
                    embedding_model=embedding_model,
                    force_reindex=force_reindex,
                    config=self.config,
                )
            except Exception as exc:
                strict = os.getenv("RAG_STRICT_VECTOR_BACKEND", "false").lower() in {"1", "true", "yes", "on"}
                if strict:
                    raise
                logger.warning("Qdrant backend unavailable; falling back to local vector store: %s", exc)
        return HybridLegalVectorStore(
            processed_path=processed_path,
            index_dir=index_dir,
            embedding_model=embedding_model,
            force_reindex=force_reindex,
        )

    def retrieve(self, query: str, top_k: int = 8, expand_depth: int = 2) -> List[Dict[str, Any]]:
        return self.retriever.retrieve(query, top_k=top_k, expand_depth=expand_depth)

    def retrieve_sign(self, query: str, top_k: int = 8, expand_depth: int = 1) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth)

    def retrieve_table(self, query: str, top_k: int = 8, expand_depth: int = 1) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_table(query, top_k=top_k, expand_depth=expand_depth)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        plan, profile = self._build_query_profile(query)
        return self._analysis_payload(plan, profile)

    def query_adaptive(self, query: str) -> Dict[str, Any]:
        """Runs the complete adaptive QA pipeline for text questions."""
        plan, profile = self._build_query_profile(query)
        return self.query_sequential(query, plan=plan, profile=profile)

    def _build_query_profile(self, query: str):
        plan = self.retriever.query_planner.plan(query, client=self.client)
        profile = self.retriever.adaptive_analyzer.analyze(query, plan)
        return plan, profile

    def _analysis_payload(self, plan: Any, profile: Any) -> Dict[str, Any]:
        summary = profile.public_summary()
        summary["plan"] = plan.public_summary()
        return summary

    def _should_use_sequential(self, profile: Any) -> bool:
        return True

    def query_sequential(
        self,
        query: str,
        top_k_per_slot: int = 6,
        *,
        plan: Any = None,
        profile: Any = None,
    ) -> Dict[str, Any]:
        """Executes a sequential retrieval flow for complex queries."""
        from src.rag.sequential_retrieval import SequentialRetrievalOrchestrator

        if plan is None or profile is None:
            plan, profile = self._build_query_profile(query)
        self.retriever.client = self.client
        budget = profile.retrieval_budget or {}
        effective_top_k = int(budget.get("evidence_slot_top_k") or top_k_per_slot)
        
        orchestrator = SequentialRetrievalOrchestrator(
            retriever=self.retriever,
            generator_fn=lambda q, c, sequential_results=None: self.generate_answer(q, c, sequential_results=sequential_results),
            asset_path_fn=public_asset_path
        )
        
        result = orchestrator.orchestrate(
            query=query,
            profile=profile,
            plan=plan,
            top_k_per_slot=effective_top_k
        )
        
        contexts = result["accumulated_records"]
        images = self._context_images(contexts, limit=int((profile.retrieval_budget or {}).get("max_images") or 30))
        return {
            "answer": result["answer"],
            "contexts": contexts,
            "references": self.format_references(contexts),
            "images": images,
            "sequential_results": result["public_results"],
            "slots": result["slots"],
            "query_analysis": self._analysis_payload(plan, profile),
            "metadata": {
                "sequential": True,
                "route": "sequential",
                "plan_source": plan.plan_source,
                "retrieval_budget": budget,
                "slots": result["slots"],
                "slot_results": result["public_results"],
                "images": images,
            },
        }

    def _retrieve_direct(self, query: str, plan: Any, profile: Any) -> List[Dict[str, Any]]:
        budget = profile.retrieval_budget or {}
        top_k = int(budget.get("top_k") or 10)
        expand_depth = int(budget.get("expand_depth") or 1)
        facets = set(profile.facets or [])
        if "sign" in facets:
            return self.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth)
        if "table" in facets:
            return self.retrieve_table(query, top_k=top_k, expand_depth=expand_depth)
        return self.retrieve(query, top_k=top_k, expand_depth=expand_depth)

    def generate_answer(
        self, 
        query: str, 
        contexts: List[Dict[str, Any]], 
        *, 
        sequential_results: Optional[List[Any]] = None
    ) -> str:
        """Synthesizes final answer from retrieved contexts."""
        if not contexts:
            return "Tôi chưa tìm thấy căn cứ phù hợp trong dữ liệu luật giao thông đã trích xuất."
        if self.client is None:
            return self._extractive_answer(query, contexts)

        contents: List[Any] = [self._system_prompt()]
        
        if sequential_results:
            branch_info = "\n[CẤU TRÚC PHÂN NHÁNH CÂU HỎI]\n"
            for res in sequential_results:
                slot = getattr(res, "slot", None)
                if slot is None and isinstance(res, dict):
                    slot = res.get("slot") or {}
                    branch_info += (
                        f"- Nhánh {slot.get('id')}: {slot.get('reason')} "
                        f"(Câu hỏi: {slot.get('query')})\n"
                    )
                    continue
                branch_info += f"- Nhánh {slot.id}: {slot.reason} (Câu hỏi: {slot.query})\n"
            contents.append(branch_info)

        max_prompt_images = int(os.getenv("RAG_MAX_PROMPT_IMAGES", "8"))
        loaded_prompt_images = 0
        for idx, ctx in enumerate(contexts, start=1):
            ref = format_reference(ctx)
            text = (ctx.get("source_body_exact") or ctx.get("rag_text") or ctx.get("content") or "")[:20000]
            image_paths = record_image_paths(ctx)
            public_images = [public_asset_path(path) for path in image_paths]
            matched_rows = ctx.get("matched_table_rows") or []
            row_text = ""
            if matched_rows:
                row_text = "\nDòng bảng khớp:\n" + "\n".join(
                    " | ".join(str(cell or "") for cell in row)
                    for row in matched_rows[:12]
                    if isinstance(row, list)
                )
            
            contents.append(
                f"\n[NGUỒN {idx}]\n"
                f"Căn cứ: {ref}\n"
                f"Nhánh truy vấn: {ctx.get('retrieval_slot_facet') or 'general'} | {ctx.get('retrieval_slot_query') or ''}\n"
                f"Ảnh/căn cứ trực quan: {', '.join(public_images) if public_images else 'Không có'}\n"
                f"Nội dung văn bản gốc:\n{text}\n"
                f"{row_text}\n"
            )
            for image_path in image_paths[:2]:
                if loaded_prompt_images >= max_prompt_images:
                    break
                image = self._load_image(image_path)
                if image is not None:
                    contents.append(image)
                    loaded_prompt_images += 1

        contents.append(f"\n[CÂU HỎI TỔNG HỢP]\n{query}\n")
        contents.append(self._answer_requirements())
        
        try:
            res, _model = generate_content_with_fallback(
                self.client,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=8192),
                env_names=("RAG_ANSWER_MODEL",),
                logger=logger,
                label="Answer generation",
            )
            return res.text or self._extractive_answer(query, contexts)
        except Exception:
            pass

        return self._extractive_answer(query, contexts)

    def format_references(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        references = []
        for ctx in contexts:
            images = [public_asset_path(path) for path in record_image_paths(ctx)]
            references.append({
                "source_chunk_id": ctx.get("source_chunk_id"),
                "reference_text": format_reference(ctx),
                "image_path": images[0] if images else "",
                "image_paths": images,
                "retrieval_slot_id": ctx.get("retrieval_slot_id"),
                "retrieval_slot_facet": ctx.get("retrieval_slot_facet"),
            })
        return references

    def _context_images(self, contexts: List[Dict[str, Any]], *, limit: int = 30) -> List[str]:
        images: List[str] = []
        seen = set()
        for ctx in contexts:
            for path in record_image_paths(ctx):
                public = public_asset_path(path)
                if public and public not in seen:
                    seen.add(public)
                    images.append(public)
                if len(images) >= limit:
                    return images
        return images

    def _system_prompt(self) -> str:
        return (
            "Bạn là chuyên gia pháp lý về Luật Giao Thông Việt Nam. "
            "Hãy tổng hợp câu trả lời đầy đủ, chính xác, mạch lạc và tự nhiên. "
            "Áp dụng zero-shot legal reasoning và few-shot pattern matching nội bộ, "
            "nhưng KHÔNG trình bày chain-of-thought. Chỉ nêu kết luận, căn cứ, điều kiện áp dụng, "
            "và các bước kiểm chứng ngắn gọn khi cần. Chỉ kết luận khi có căn cứ trong nguồn được cung cấp."
        )

    def _answer_requirements(self) -> str:
        return (
            "\n[YÊU CẦU]\n"
            "1. Trích dẫn đúng Điều/Khoản/Điểm và tên văn bản khi nguồn có metadata.\n"
            "2. Nếu là tình huống thực tế, tách theo: dữ kiện -> quy tắc áp dụng -> kết luận -> rủi ro/mức phạt.\n"
            "3. Nếu là xe/trường hợp ưu tiên, nêu rõ điều kiện được ưu tiên và nghĩa vụ nhường đường.\n"
            "4. Nếu có bảng, dùng đúng dòng/cột khớp; không suy diễn ngoài bảng.\n"
            "5. Nếu có biển báo/hình ảnh, mô tả mã biển, hình dạng, ý nghĩa và nhắc ảnh/căn cứ trực quan nếu có.\n"
            "6. Nếu thiếu căn cứ cho một nhánh, nói rõ nhánh đó chưa đủ căn cứ thay vì đoán.\n"
            "7. Cuối câu trả lời nêu 'TỔNG HẬU QUẢ' cho trường hợp người dùng hỏi.\n"
            "Few-shot format nội bộ: 'Biển + hành vi' => ý nghĩa biển trước, hành vi sau, xử phạt cuối; "
            "'Bảng/phụ lục' => nêu dòng/cột; 'Tình huống nhiều bước' => kết luận từng bước."
        )

    def _extractive_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        parts = ["Căn cứ trích xuất trực tiếp:"]
        for idx, ctx in enumerate(contexts[:12], start=1):
            image = public_asset_path(ctx.get("image_path")) if ctx.get("image_path") else ""
            image_text = f"\nẢnh/căn cứ trực quan: {image}" if image else ""
            slot = ctx.get("retrieval_slot_facet") or "general"
            parts.append(f"{idx}. [{slot}] {format_reference(ctx)}:{image_text}\n{source_text(ctx)[:1600]}")
        return "\n\n".join(parts)

    def _load_image(self, path: Optional[str]):
        if not path: return None
        img_path = Path(path)
        if not img_path.is_absolute(): img_path = self.project_root / img_path
        if not img_path.exists(): return None
        try:
            return PIL.Image.open(img_path)
        except Exception: return None
