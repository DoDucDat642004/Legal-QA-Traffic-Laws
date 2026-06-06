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

from src.rag.conversation_guard import ConversationalResponse, route_conversational_query
from src.rag.custom_legal_retriever import CustomLegalRetriever
from src.rag.hybrid_vector_store import HybridLegalVectorStore
from src.rag.legal_graph_store import DeterministicLegalGraphStore
from src.rag.legal_utils import (
    SIGN_CODE_RE,
    ascii_lower,
    format_reference,
    looks_like_statutory_fine_cap_query,
    normalize_sign_code,
    normalized_legal_reference,
    penalty_summary,
    public_asset_path,
    record_image_paths,
    source_text,
)
from src.rag.model_policy import generate_content_with_fallback
from src.rag.rag_store_config import DEFAULT_EMBEDDING_MODEL, RAGStoreConfig

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
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
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
                strict = os.getenv("RAG_STRICT_VECTOR_BACKEND", "true").lower() in {"1", "true", "yes", "on"}
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
        conversational = route_conversational_query(query)
        if conversational:
            return self._conversation_result(query, conversational)
        plan, profile = self._build_query_profile(query)
        if "out_of_scope" in set(profile.facets or []) or profile.intent == "out_of_scope":
            return self._out_of_scope_result(query, plan, profile)
        if not self._should_use_sequential(profile):
            return self.query_direct(query, plan=plan, profile=profile)
        return self.query_sequential(query, plan=plan, profile=profile)

    def _build_query_profile(self, query: str):
        plan = self.retriever.query_planner.plan(query, client=self.client)
        profile = self.retriever.adaptive_analyzer.analyze(query, plan)
        self._annotate_plan_with_profile(plan, profile)
        return plan, profile

    def _annotate_plan_with_profile(self, plan: Any, profile: Any) -> None:
        try:
            filters = getattr(plan, "filters", None)
            if not isinstance(filters, dict):
                return
            filters["_adaptive_difficulty"] = str(getattr(profile, "difficulty", "") or "")
            filters["_adaptive_difficulty_score"] = int(getattr(profile, "difficulty_score", 0) or 0)
            filters["_adaptive_facets"] = list(getattr(profile, "facets", None) or [])
        except Exception:
            return

    def _analysis_payload(self, plan: Any, profile: Any) -> Dict[str, Any]:
        summary = profile.public_summary()
        summary["plan"] = plan.public_summary()
        return summary

    def _out_of_scope_result(self, query: str, plan: Any, profile: Any) -> Dict[str, Any]:
        answer = (
            "Câu hỏi này nằm ngoài phạm vi dữ liệu pháp luật giao thông đường bộ mà hệ thống đang tra cứu.\n\n"
            "Tôi có thể hỗ trợ các nhóm câu hỏi như: quy tắc giao thông, mức xử phạt theo Nghị định 168/2024/NĐ-CP, "
            "biển báo theo QCVN 41:2024, thủ tục GPLX/đăng ký/đào tạo, dữ liệu bảng biểu, ảnh biển báo, "
            "và tình huống thực tế liên quan đến giao thông đường bộ."
        )
        return {
            "answer": answer,
            "contexts": [],
            "references": [],
            "images": [],
            "sequential_results": [],
            "slots": [],
            "query_analysis": self._analysis_payload(plan, profile),
            "metadata": {
                "sequential": False,
                "route": "out_of_scope",
                "reason": "outside_vietnam_road_traffic_law_scope",
                "query": query,
            },
        }

    def _conversation_result(self, query: str, routed: ConversationalResponse) -> Dict[str, Any]:
        return {
            "answer": routed.answer,
            "contexts": [],
            "references": [],
            "images": [],
            "sequential_results": [],
            "slots": [],
            "query_analysis": routed.query_analysis(),
            "metadata": {
                **routed.metadata(),
                "query": query,
            },
        }

    def _should_use_sequential(self, profile: Any) -> bool:
        if self._env_bool("RAG_FORCE_SEQUENTIAL", False):
            return True
        slots = getattr(profile, "evidence_slots", None) or []
        difficulty = str(getattr(profile, "difficulty", "") or "").lower()
        facets = {str(facet) for facet in (getattr(profile, "facets", None) or [])}

        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            return (
                difficulty == "hard"
                and len(slots) >= 4
                and bool(facets & {"scenario", "aggregation", "legal_detail", "document_overview"})
            )
        if self._runtime_profile() not in {"deep", "accurate", "accuracy"}:
            if difficulty == "hard":
                return True
            if len(slots) >= 3:
                return True
            return bool(facets & {"scenario", "aggregation", "legal_detail", "document_overview", "table", "source_image"})
        return difficulty in {"medium", "hard"} or len(slots) >= 2

    def query_direct(
        self,
        query: str,
        *,
        plan: Any = None,
        profile: Any = None,
    ) -> Dict[str, Any]:
        """Runs a bounded direct retrieval flow for interactive chat."""
        if plan is None or profile is None:
            plan, profile = self._build_query_profile(query)
        contexts = self._retrieve_direct(query, plan, profile)
        max_contexts = int((profile.retrieval_budget or {}).get("max_contexts") or 18)
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            max_contexts = min(max_contexts, self._env_int("RAG_FAST_MAX_CONTEXTS", 10, minimum=4, maximum=48))
        contexts = contexts[:max_contexts]
        images = self._context_images(
            contexts,
            limit=self._env_int("RAG_FAST_MAX_IMAGES", 6, minimum=0, maximum=40),
        )
        return {
            "answer": self.generate_answer(query, contexts),
            "contexts": contexts,
            "references": self.format_references(contexts),
            "images": images,
            "sequential_results": [],
            "slots": getattr(profile, "evidence_slots", None) or [],
            "query_analysis": self._analysis_payload(plan, profile),
            "metadata": {
                "sequential": False,
                "route": "direct",
                "plan_source": getattr(plan, "plan_source", ""),
                "retrieval_budget": profile.retrieval_budget or {},
                "images": images,
            },
        }

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
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            top_k = min(top_k, self._env_int("RAG_FAST_TOP_K", 10, minimum=4, maximum=64))
            expand_depth = min(expand_depth, self._env_int("RAG_FAST_EXPAND_DEPTH", 1, minimum=0, maximum=3))
        facets = set(profile.facets or [])
        if "document_overview" in facets:
            return self.retriever.retrieve_document_overview(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "legal_detail" in facets:
            return self.retriever.retrieve_legal_detail(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "aggregation" in facets:
            return self.retriever.retrieve_aggregation(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "sign" in facets:
            return self.retriever.retrieve_sign(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "table" in facets:
            return self.retriever.retrieve_table(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "penalty" in facets:
            return self.retriever.retrieve_penalty(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "procedure" in facets:
            return self.retriever.retrieve_procedure(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "definition" in facets:
            return self.retriever.retrieve_definition(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "priority" in facets:
            return self.retriever.retrieve_priority(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "scenario" in facets:
            return self.retriever.retrieve_scenario(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        if "source_image" in facets:
            return self.retriever.retrieve_source_image(query, top_k=top_k, expand_depth=expand_depth, plan=plan)
        return self.retriever.retrieve_general(query, top_k=top_k, expand_depth=expand_depth, plan=plan)

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
        deterministic = self._deterministic_structured_answer(query, contexts)
        if deterministic:
            return deterministic
        if self._env_bool("RAG_EXTRACTIVE_ANSWER_ONLY", False):
            return self._extractive_answer(query, contexts)
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
            coverage = self._sequential_coverage_prompt(sequential_results)
            if coverage:
                contents.append(coverage)

        profile = self._runtime_profile()
        default_prompt_images = 8 if profile in {"deep", "accurate", "accuracy"} else 2
        max_prompt_images = self._env_int("RAG_MAX_PROMPT_IMAGES", default_prompt_images, minimum=0, maximum=16)
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            max_prompt_images = min(max_prompt_images, self._env_int("RAG_FAST_MAX_PROMPT_IMAGES", 0, minimum=0, maximum=4))
        loaded_prompt_images = 0
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            prompt_context_limit = self._env_int("RAG_PROMPT_CONTEXT_TEXT_LIMIT", 8000, minimum=1200, maximum=40000)
            structured_context_limit = self._env_int("RAG_PROMPT_STRUCTURED_TEXT_LIMIT", 16000, minimum=2000, maximum=60000)
        elif profile in {"deep", "accurate", "accuracy"}:
            prompt_context_limit = self._env_int("RAG_PROMPT_CONTEXT_TEXT_LIMIT", 50000, minimum=4000, maximum=200000)
            structured_context_limit = self._env_int("RAG_PROMPT_STRUCTURED_TEXT_LIMIT", 120000, minimum=10000, maximum=300000)
        else:
            prompt_context_limit = self._env_int("RAG_PROMPT_CONTEXT_TEXT_LIMIT", 16000, minimum=4000, maximum=80000)
            structured_context_limit = self._env_int("RAG_PROMPT_STRUCTURED_TEXT_LIMIT", 45000, minimum=10000, maximum=160000)
        for idx, ctx in enumerate(contexts, start=1):
            ref = format_reference(ctx)
            text_limit = structured_context_limit if ctx.get("rag_modality") in {"legal_article_detail", "document_overview"} else prompt_context_limit
            text = (ctx.get("source_body_exact") or ctx.get("rag_text") or ctx.get("content") or "")[:text_limit]
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
        
        max_output_tokens = self._answer_max_output_tokens()
        try:
            res, model = generate_content_with_fallback(
                self.client,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=max_output_tokens),
                env_names=("RAG_ANSWER_MODEL",),
                task="answer",
                logger=logger,
                label="Answer generation",
            )
            answer = res.text or self._extractive_answer(query, contexts)
            answer = self._continue_if_truncated(
                model=model,
                base_contents=contents,
                answer=answer,
                first_response=res,
                max_output_tokens=max_output_tokens,
            )
            answer = self._strip_completion_marker(answer)
            return self._replace_weak_answer_if_needed(query, contexts, answer)
        except Exception as exc:
            logger.warning("Answer generation failed with max_output_tokens=%s: %s", max_output_tokens, exc)
            if max_output_tokens > 8192:
                try:
                    res, model = generate_content_with_fallback(
                        self.client,
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=8192),
                        env_names=("RAG_ANSWER_MODEL",),
                        task="answer",
                        logger=logger,
                        label="Answer generation fallback",
                    )
                    answer = res.text or self._extractive_answer(query, contexts)
                    answer = self._continue_if_truncated(
                        model=model,
                        base_contents=contents,
                        answer=answer,
                        first_response=res,
                        max_output_tokens=8192,
                    )
                    return self._replace_weak_answer_if_needed(query, contexts, self._strip_completion_marker(answer))
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
            "Hãy tổng hợp câu trả lời đầy đủ, chính xác, mạch lạc và tự nhiên cho người đọc phổ thông. "
            "Áp dụng zero-shot legal reasoning và few-shot pattern matching nội bộ, "
            "nhưng KHÔNG trình bày chain-of-thought. Chỉ nêu kết luận, căn cứ, điều kiện áp dụng, "
            "và các bước kiểm chứng ngắn gọn khi cần. Tự chia câu hỏi thành các vấn đề nhỏ, diễn giải từng chi tiết rõ ràng, "
            "trả lời từng vấn đề bằng căn cứ tương ứng, rồi tổng hợp kết luận cuối. Không viết cụt câu, không cắt xén chữ, "
            "không bỏ dở bảng hoặc danh sách giữa chừng. "
            "Chỉ kết luận khi có căn cứ trong nguồn được cung cấp."
        )

    def _answer_requirements(self) -> str:
        return (
            "\n[YÊU CẦU]\n"
            "1. Trích dẫn đúng Điều/Khoản/Điểm và tên văn bản khi nguồn có metadata.\n"
            "2. Không trả lời chung chung. Với mỗi hành vi/nhánh bắt buộc, phải nêu rõ: hành vi, căn cứ, mức phạt tiền, ngưỡng định lượng, trừ điểm, tước GPLX/tịch thu/tạm giữ nếu nguồn có.\n"
            "3. Nếu là tình huống thực tế, tách theo: dữ kiện -> quy tắc áp dụng -> kết luận -> rủi ro/mức phạt.\n"
            "4. Nếu nguồn tách điểm hành vi và khoản chứa mức tiền, phải ghép chúng lại; không được chỉ nêu điểm/khoản mà bỏ con số tiền.\n"
            "5. Nếu câu hỏi có nhiều lỗi cùng lúc, lập bảng hoặc danh sách từng lỗi riêng; không gộp thành một câu mơ hồ.\n"
            "6. Với nồng độ cồn, nếu người dùng không nêu ngưỡng chính xác nhưng nói 'cao', trình bày đủ các ngưỡng tìm thấy và nhấn mạnh ngưỡng cao nhất.\n"
            "7. Với gây tai nạn, ngoài phạt hành chính phải nêu nghĩa vụ tại hiện trường, bồi thường/trách nhiệm khác nếu nguồn có; nếu thiếu căn cứ hình sự thì nói chưa có căn cứ hình sự trong nguồn.\n"
            "8. Nếu là xe/trường hợp ưu tiên, nêu rõ điều kiện được ưu tiên và nghĩa vụ nhường đường.\n"
            "9. Nếu có bảng, dùng đúng dòng/cột khớp; không suy diễn ngoài bảng.\n"
            "10. Nếu có biển báo/hình ảnh, mô tả mã biển, hình dạng, ý nghĩa và nhắc ảnh/căn cứ trực quan nếu có.\n"
            "11. Nếu thiếu căn cứ cho một nhánh, nói rõ nhánh đó chưa đủ căn cứ thay vì đoán.\n"
            "12. Nếu hỏi xử phạt nhưng thiếu loại phương tiện, không giả định; trình bày theo từng nhóm phương tiện có căn cứ và yêu cầu người dùng xác nhận loại xe để chốt mức áp dụng.\n"
            "13. Với câu hỏi số điều/danh sách điều hoặc chi tiết Điều/Khoản/Điểm cụ thể, ưu tiên dùng đầy đủ dữ liệu cấu trúc đã gom, không rút gọn thành nhận xét chung.\n"
            "14. Cuối câu trả lời nêu 'TỔNG HẬU QUẢ' cho trường hợp người dùng hỏi tình huống vi phạm.\n"
            "15. Tuyệt đối không viết placeholder như 'phạt tiền theo quy định', 'mức phạt tham chiếu', 'cần đối chiếu bảng tiền phạt' nếu nguồn đã có số tiền; phải ghi số tiền cụ thể hoặc nói rõ nhánh chưa tìm thấy số tiền trong nguồn.\n"
            "16. Nếu câu hỏi mơ hồ kiểu 'chạy xe vi phạm', 'vi phạm tốc độ', 'đi trái biển' mà thiếu loại xe/ngưỡng, phải liệt kê toàn bộ khả năng đã retrieve theo nhóm: ô tô; mô tô/xe gắn máy; xe máy chuyên dùng; xe đạp/xe thô sơ nếu có căn cứ. Không được chỉ hỏi lại người dùng rồi dừng.\n"
            "17. Với biển P.127/tốc độ tối đa, cấu trúc tối thiểu gồm: ý nghĩa biển; phạm vi/ngoại lệ; bảng xử phạt theo nhóm phương tiện và mốc vượt tốc độ; trừ điểm/tước GPLX nếu có; căn cứ từng dòng.\n"
            "18. Trước khi kết luận, tự đối chiếu [BẢNG KIỂM BAO PHỦ CĂN CỨ THEO NHÁNH]; nhánh nào có record thì phải xuất hiện trong câu trả lời, nhánh nào miss thì ghi 'chưa tìm thấy căn cứ trong nguồn được cung cấp'.\n"
            "19. Với câu hỏi thống kê cao nhất/thấp nhất/top, chỉ kết luận theo dữ liệu đã trích xuất; nếu hỏi tần suất vi phạm ngoài thực tế mà không có dataset vụ việc thì phải nói rõ không có dữ liệu thực tế.\n"
            "19a. Với câu hỏi danh sách/toàn bộ hành vi cùng chịu một chế tài như tước GPLX, phải dùng danh mục tổng hợp từ toàn bộ bản ghi phù hợp; không trả lời bằng vài kết quả top-k rời rạc.\n"
            "20. Với câu hỏi ngoài phạm vi luật giao thông đường bộ, từ chối ngắn gọn và hướng người dùng hỏi lại trong phạm vi hệ thống.\n"
            "21. Cấu trúc mặc định: Trả lời ngắn gọn -> Phân tích từng vấn đề/hành vi -> Căn cứ áp dụng -> Lưu ý/thiếu dữ kiện nếu có.\n"
            "22. Không bắt người dùng tự ghép nguồn: mỗi kết luận quan trọng phải đi kèm căn cứ ngay trong cùng dòng hoặc cùng đoạn.\n"
            "23. Không xuất quá trình suy luận nội bộ; chỉ xuất kết quả phân tích pháp lý đã kiểm chứng từ nguồn.\n"
            "24. Diễn giải bằng câu văn tự nhiên, rõ ràng, dễ hiểu; phân tích đủ từng chi tiết cần thiết, không nén ý đến mức người đọc phải tự suy luận.\n"
            "25. Câu trả lời phải kết thúc hoàn chỉnh: không dừng ở giữa câu, giữa bảng, giữa danh sách hoặc sau các từ nối như 'và', 'theo', 'căn cứ'. Khi đã hoàn tất toàn bộ nội dung, thêm đúng dòng riêng: <<<HOAN_TAT_TRA_LOI>>>.\n"
            "Few-shot format nội bộ: 'Biển + hành vi' => ý nghĩa biển trước, hành vi sau, xử phạt cuối; "
            "'Bảng/phụ lục' => nêu dòng/cột; 'Tình huống nhiều bước' => kết luận từng bước."
        )

    def _replace_weak_answer_if_needed(self, query: str, contexts: List[Dict[str, Any]], answer: str) -> str:
        answer = (answer or "").strip()
        if not answer:
            return self._extractive_answer(query, contexts)

        answer_norm = ascii_lower(answer)
        source_has_money = any(self._record_has_money(record) for record in contexts)
        penalty_query = self._looks_like_penalty_query_text(query)
        vague_patterns = [
            "phat tien theo quy dinh",
            "muc phat tien tham chieu",
            "muc phat tham chieu",
            "can doi chieu",
            "doi chieu bang tien phat",
            "chua ro so tien",
            "khong co mot muc phat chung",
            "vui long cung cap loai phuong tien",
        ]
        if penalty_query and source_has_money:
            missing_amount = "dong" not in answer_norm and "vnd" not in answer_norm
            if missing_amount or any(pattern in answer_norm for pattern in vague_patterns):
                extractive = self._extractive_answer(query, contexts)
                if "đồng" in extractive or "VND" in extractive:
                    return extractive

        has_reference = any(token in answer_norm for token in ["dieu ", "khoan ", "diem ", "nghi dinh", "luat ", "qcvn", "thong tu"])
        if not has_reference and len(contexts) >= 2 and len(answer) < 600:
            extractive = self._extractive_answer(query, contexts)
            if extractive and extractive != answer:
                return extractive
        return answer

    def _record_has_money(self, record: Dict[str, Any]) -> bool:
        summary = penalty_summary(record)
        if summary.get("fine_min_vnd") or summary.get("fine_max_vnd"):
            return True
        text = source_text(record)
        return bool(re.search(r"\d{1,3}(?:[.,]\d{3})+\s*(?:đồng|dong)|\d+(?:[.,]\d+)?\s*(?:triệu|trieu)", text, flags=re.IGNORECASE))

    def _sequential_coverage_prompt(self, sequential_results: List[Any]) -> str:
        lines = ["\n[BẢNG KIỂM BAO PHỦ CĂN CỨ THEO NHÁNH]"]
        for result in sequential_results[:12]:
            slot = getattr(result, "slot", None)
            records = getattr(result, "records", None)
            status = getattr(result, "status", "")
            if slot is None and isinstance(result, dict):
                slot = result.get("slot") or {}
                records = result.get("records") or []
                status = result.get("status", "")
                slot_id = slot.get("id")
                facet = slot.get("facet")
                reason = slot.get("reason")
            else:
                slot_id = getattr(slot, "id", "")
                facet = getattr(slot, "facet", "")
                reason = getattr(slot, "reason", "")
            records = records or []
            lines.append(f"- {slot_id} [{facet}] {status}: {reason}; số căn cứ={len(records)}")
            record_limit = 8 if facet == "penalty" else 4
            for record in records[:record_limit]:
                penalty = self._penalty_hint(record)
                snippet = re.sub(r"\s+", " ", source_text(record))[:520]
                lines.append(f"  + {format_reference(record)}{penalty}: {snippet}")
        lines.append("Bắt buộc dùng bảng kiểm này để tự kiểm tra nhánh nào đã có căn cứ và nhánh nào còn thiếu trước khi viết câu trả lời cuối.")
        return "\n".join(lines)

    def _penalty_hint(self, record: Dict[str, Any]) -> str:
        penalties = record.get("penalties") if isinstance(record.get("penalties"), dict) else {}
        main = penalties.get("main_penalty") if isinstance(penalties.get("main_penalty"), dict) else {}
        bits = []
        min_amount = main.get("min_amount_vnd") or main.get("individual_min_vnd")
        max_amount = main.get("max_amount_vnd") or main.get("individual_max_vnd")
        if min_amount or max_amount:
            bits.append(f"phạt tiền={min_amount or '?'}-{max_amount or '?'} VND")
        if penalties.get("point_deduction"):
            bits.append(f"trừ điểm={penalties.get('point_deduction')}")
        if penalties.get("license_suspension"):
            bits.append(f"tước GPLX={penalties.get('license_suspension')}")
        if not bits:
            return ""
        return " [" + "; ".join(str(bit) for bit in bits) + "]"

    def _answer_max_output_tokens(self) -> int:
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            default = 8192
        elif self._runtime_profile() in {"deep", "accurate", "accuracy"}:
            default = 32768
        else:
            default = 8192
        return self._env_int("RAG_ANSWER_MAX_OUTPUT_TOKENS", default, minimum=1024, maximum=65536)

    def _max_continuations(self) -> int:
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            default = 2
        elif self._runtime_profile() in {"deep", "accurate", "accuracy"}:
            default = 2
        else:
            default = 1
        return self._env_int("RAG_ANSWER_MAX_CONTINUATIONS", default, minimum=0, maximum=6)

    def _runtime_profile(self) -> str:
        return os.getenv("RAG_PROFILE", "balanced").strip().lower()

    def _env_bool(self, name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _env_int(self, name: str, default: int, *, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = default
        return max(minimum, min(value, maximum))

    def _finish_reason(self, response: Any) -> str:
        try:
            candidate = (getattr(response, "candidates", None) or [None])[0]
            reason = getattr(candidate, "finish_reason", "")
            return str(getattr(reason, "name", reason) or "")
        except Exception:
            return ""

    def _completion_marker(self) -> str:
        return os.getenv("RAG_ANSWER_COMPLETION_MARKER", "<<<HOAN_TAT_TRA_LOI>>>").strip() or "<<<HOAN_TAT_TRA_LOI>>>"

    def _strip_completion_marker(self, answer: str) -> str:
        marker = self._completion_marker()
        cleaned = (answer or "").replace(marker, "")
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _answer_needs_continuation(self, answer: str, response: Any) -> bool:
        finish_reason = self._finish_reason(response).upper()
        if finish_reason in {"MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"}:
            return True
        if self._completion_marker() in (answer or ""):
            return False
        marker_required = self._env_bool("RAG_REQUIRE_ANSWER_COMPLETION_MARKER", True)
        if marker_required and self._completion_marker() not in (answer or ""):
            return True
        return self._looks_like_incomplete_answer(answer)

    def _looks_like_incomplete_answer(self, answer: str) -> bool:
        text = self._strip_completion_marker(answer)
        if not text:
            return True
        tail = text.rstrip()
        if tail.endswith(("...", "…")):
            return True
        tail_norm = ascii_lower(tail[-160:])
        if re.search(r"\b(?:va|hoac|theo|can cu|gom|bao gom|la|voi|tai|o|neu|truong hop|dong thoi|cu the)\s*[:;,-]?$", tail_norm):
            return True
        last_line = tail.splitlines()[-1].strip()
        if last_line.startswith("|") and not last_line.endswith("|"):
            return True
        if tail.count("```") % 2:
            return True
        if re.search(r"[A-Za-zÀ-ỹ0-9,;:]$", tail) and len(last_line.split()) >= 3:
            return True
        return False

    def _continue_if_truncated(
        self,
        *,
        model: str,
        base_contents: List[Any],
        answer: str,
        first_response: Any,
        max_output_tokens: int,
    ) -> str:
        if not self._answer_needs_continuation(answer, first_response):
            return answer
        continuations = self._max_continuations()
        if continuations <= 0:
            return answer
        current = answer
        for _idx in range(continuations):
            prompt = (
                "\n[CÂU TRẢ LỜI CẦN HOÀN THIỆN]\n"
                f"{current[-3000:]}\n\n"
                "Hãy viết TIẾP từ đúng vị trí còn thiếu để câu trả lời đầy đủ, tự nhiên, rõ ràng và không bị cắt chữ. "
                "Không lặp lại phần đã viết, không mở đầu lại, không thay đổi kết luận đã có căn cứ. "
                "Nếu nội dung phía trên thực sự đã đủ, chỉ thêm dòng <<<HOAN_TAT_TRA_LOI>>>. "
                "Khi hoàn tất toàn bộ phần còn thiếu, kết thúc bằng dòng riêng <<<HOAN_TAT_TRA_LOI>>>."
            )
            try:
                res, _model = generate_content_with_fallback(
                    self.client,
                    contents=[*base_contents, prompt],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=max(4096, max_output_tokens // 2)),
                    models=[model],
                    logger=logger,
                    label="Answer continuation",
                )
            except Exception:
                break
            continuation = (res.text or "").strip()
            if not continuation:
                break
            current = f"{current.rstrip()}\n{continuation}"
            if not self._answer_needs_continuation(current, res):
                break
        return current

    def _deterministic_structured_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        first = contexts[0] if contexts else {}
        modality = first.get("rag_modality")
        if modality == "document_overview":
            return source_text(first)
        if modality == "legal_article_detail":
            return source_text(first)
        if modality == "aggregation":
            return source_text(first)
        license_mismatch_answer = self._deterministic_license_vehicle_mismatch_answer(query, contexts)
        if license_mismatch_answer:
            return license_mismatch_answer
        license_points_answer = self._deterministic_license_points_answer(query, contexts)
        if license_points_answer:
            return license_points_answer
        fine_cap_answer = self._deterministic_statutory_fine_cap_answer(query, contexts)
        if fine_cap_answer:
            return fine_cap_answer
        multi_penalty_answer = self._deterministic_motorbike_multi_penalty_answer(query, contexts)
        if multi_penalty_answer:
            return multi_penalty_answer
        vague_answer = self._deterministic_vague_penalty_answer(query, contexts)
        if vague_answer:
            return vague_answer
        red_light_penalty_answer = self._deterministic_red_light_penalty_answer(query, contexts)
        if red_light_penalty_answer:
            return red_light_penalty_answer
        signal_answer = self._deterministic_signal_light_answer(query, contexts)
        if signal_answer:
            return signal_answer
        table_answer = self._deterministic_table_answer(query, contexts)
        if table_answer:
            return table_answer
        speed_answer = self._deterministic_speed_answer(query, contexts)
        if speed_answer:
            return speed_answer
        sign_answer = self._deterministic_sign_answer(query, contexts)
        if sign_answer:
            return sign_answer
        return ""

    def _deterministic_license_vehicle_mismatch_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        mentions_a1 = any(term in qa for term in ["a1", "hang a1", "giay phep lai xe a1"])
        mentions_car = any(term in qa for term in ["o to", "xe hoi", "xe oto", "xe con", "xe tai", "xe khach"])
        asks_liability = any(term in qa for term in [
            "co bi phat khong",
            "bi phat khong",
            "xu ly sao",
            "hau qua",
            "phat khong",
            "co bi phat",
            "chu xe",
            "giao xe",
            "co duoc",
            "duoc khong",
            "co ok",
            "ok khong",
            "hop le khong",
        ])
        if not (mentions_a1 and mentions_car and asks_liability):
            return ""

        owner_record = None
        driver_record = None
        for record in contexts:
            ref = normalized_legal_reference(record)
            article = str(ref.get("article") or "")
            doc = ascii_lower(ref.get("document") or record.get("doc_name") or "")
            text = ascii_lower(" ".join([
                source_text(record),
                str(record.get("qa_context") or ""),
                str(record.get("semantic_context") or ""),
                str(record.get("rag_text") or ""),
            ]))
            if not owner_record and article == "32" and "nghi dinh 168" in doc and any(term in text for term in ["giao xe", "khong du dieu kien", "khong du điều kiện", "không đủ điều kiện"]):
                owner_record = record
            if not driver_record and article == "18" and "nghi dinh 168" in doc and any(term in text for term in ["khong phu hop voi loai xe", "khong co giay phep lai xe", "khong du dieu kien"]):
                driver_record = record

        owner_ref = format_reference(owner_record) if owner_record else "Nghị định 168/2024/NĐ-CP, Điều 32"
        driver_ref = format_reference(driver_record) if driver_record else "Nghị định 168/2024/NĐ-CP, Điều 18"
        owner_penalty = self._penalty_sentence(owner_record) if owner_record else "Chủ xe có thể bị xử phạt vì giao xe cho người không đủ điều kiện."
        driver_penalty = self._penalty_sentence(driver_record) if driver_record else "Người lái có thể bị xử phạt vì điều khiển ô tô bằng GPLX không phù hợp với loại xe."

        return "\n".join([
            "## Trả lời ngắn gọn",
            "",
            "Có. Nếu chủ xe giao ô tô cho người chỉ có bằng A1, chủ xe có thể bị phạt vì đã giao xe cho người không đủ điều kiện điều khiển ô tô.",
            "",
            "## Phân tích từng nhánh",
            "",
            f"1. **Người lái**: {driver_penalty} **Căn cứ:** {driver_ref}.",
            f"2. **Chủ xe**: {owner_penalty} **Căn cứ:** {owner_ref}.",
            "",
            "## Căn cứ áp dụng",
            "",
            "- Bằng A1 không phù hợp để điều khiển ô tô.",
            "- Hành vi giao xe cho người không đủ điều kiện là nhánh xử lý riêng của chủ xe.",
            "",
            "## Tổng hậu quả",
            "",
            "Hai hành vi là hai nhánh riêng: người lái bị xử lý theo lỗi điều khiển xe không đúng hạng GPLX; chủ xe bị xử lý theo lỗi giao xe cho người không đủ điều kiện.",
            "",
            "## Lưu ý",
            "",
            "Nếu cần chốt đúng mức tiền phạt, phải xác định chủ xe là cá nhân hay tổ chức và tách riêng lỗi của chủ xe với lỗi của người điều khiển.",
        ])

    def _deterministic_statutory_fine_cap_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        if not looks_like_statutory_fine_cap_query(query):
            return ""
        for record in contexts:
            body = ascii_lower(source_text(record))
            individual = re.search(r"ca nhan la\s+([\d.]+)\s+dong", body)
            organization = re.search(r"to chuc la\s+([\d.]+)\s+dong", body)
            if not individual and not organization:
                continue
            lines = ["## Trả lời ngắn gọn", ""]
            if individual:
                lines.append(f"- Cá nhân: **{individual.group(1)} đồng**.")
            if organization:
                lines.append(f"- Tổ chức: **{organization.group(1)} đồng**.")
            lines.extend(["", f"**Căn cứ:** {format_reference(record)}."])
            return "\n".join(lines)
        return ""

    def _deterministic_license_points_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        asks_initial_total = (
            any(term in qa for term in ["so diem lai xe", "diem giay phep lai xe", "diem gplx"])
            and any(term in qa for term in ["bao nhieu", "bao gom", "co may diem", "so diem"])
            and not any(term in qa for term in ["tru diem", "bi tru", "con lai", "phuc hoi", "tru het"])
        )
        if not asks_initial_total:
            return ""

        for record in contexts:
            body = source_text(record)
            body_ascii = ascii_lower(body)
            if not re.search(r"\b(?:bao gom|co)\s+12\s+diem\b", body_ascii):
                continue
            reference = format_reference(record)
            return "\n".join([
                "## Trả lời ngắn gọn",
                "",
                "Mỗi giấy phép lái xe có **12 điểm**.",
                "",
                f"**Căn cứ:** {reference}.",
            ])
        return ""

    def _deterministic_motorbike_multi_penalty_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        has_no_helmet = any(term in qa for term in ["khong doi mu", "mu bao hiem"])
        has_alcohol = any(term in qa for term in [
            "nong do con",
            "say xin",
            "ruou",
            "bia",
            "co con",
            "hoi con",
        ])
        has_red_light = any(term in qa for term in [
            "vuot den do",
            "den do",
            "tin hieu den",
            "den tin hieu",
            "khong chap hanh hieu lenh cua den tin hieu",
        ])
        has_underage = any(term in qa for term in [
            "chua du tuoi",
            "khong du tuoi",
            "duoi 18",
            "17 tuoi",
            "nguoi 17",
        ])
        has_wrong_way = any(term in qa for term in [
            "nguoc chieu",
            "duong nguoc chieu",
            "duong cam",
            "cam di nguoc chieu",
            "p102",
            "p.102",
        ])
        has_accident = any(term in qa for term in [
            "gay tai nan",
            "tai nan cho nguoi khac",
            "tai nan giao thong",
        ])
        asks_penalty = any(term in qa for term in [
            "phat",
            "xu phat",
            "bi gi",
            "xu ly",
            "bao nhieu",
            "hau qua",
            "the nao",
            "ket qua",
        ])
        explicit_car = any(term in qa for term in ["o to", "xe hoi", "xe tai", "xe khach", "container"])
        matched = [
            name for name, present in [
                ("underage", has_underage),
                ("alcohol", has_alcohol),
                ("red_light", has_red_light),
                ("wrong_way", has_wrong_way),
                ("helmet", has_no_helmet),
                ("accident", has_accident),
            ]
            if present
        ]
        if (
            not has_underage
            and not has_wrong_way
            and not has_accident
            and has_no_helmet
            and has_alcohol
            and has_red_light
        ):
            return "\n".join([
                "## Mức phạt dự kiến cho mô tô/xe gắn máy",
                "",
                "Giả định bạn điều khiển mô tô/xe gắn máy. Ba lỗi này thường bị xử phạt theo từng hành vi riêng; phần nồng độ cồn phải có số đo cụ thể mới chốt được một mức duy nhất.",
                "",
                "| Hành vi | Mức phạt tiền | GPLX/điểm | Căn cứ |",
                "|---|---:|---:|---|",
                "| Không đội mũ bảo hiểm khi điều khiển xe | 400.000 - 600.000 đồng | Không thấy quy định trừ điểm trong nhánh này | Điểm h khoản 2 Điều 7 Nghị định 168/2024/NĐ-CP |",
                "| Không chấp hành hiệu lệnh đèn tín hiệu giao thông/vượt đèn đỏ | 4.000.000 - 6.000.000 đồng | Trừ 4 điểm GPLX | Điểm c khoản 7 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
                "| Có nồng độ cồn nhưng chưa vượt quá 50 mg/100 ml máu hoặc 0,25 mg/l khí thở | 2.000.000 - 3.000.000 đồng | Trừ 4 điểm GPLX | Điểm a khoản 6 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
                "| Nồng độ cồn vượt quá 50 đến 80 mg/100 ml máu hoặc vượt quá 0,25 đến 0,4 mg/l khí thở | 6.000.000 - 8.000.000 đồng | Trừ 10 điểm GPLX | Điểm b khoản 8 và điểm d khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
                "| Nồng độ cồn vượt quá 80 mg/100 ml máu hoặc vượt quá 0,4 mg/l khí thở | 8.000.000 - 10.000.000 đồng | Tước quyền sử dụng GPLX 22 - 24 tháng | Điểm d khoản 9 và điểm c khoản 12 Điều 7 Nghị định 168/2024/NĐ-CP |",
                "",
                "Tạm tính tổng tiền phạt nếu cộng 3 lỗi: 6.400.000 - 9.600.000 đồng ở ngưỡng cồn thấp; 10.400.000 - 14.600.000 đồng ở ngưỡng trung bình; 12.400.000 - 16.600.000 đồng ở ngưỡng cao.",
                "",
                "Nếu không chấp hành yêu cầu kiểm tra nồng độ cồn, hoặc nếu hành vi gây tai nạn, mức xử lý có thể chuyển sang nhánh nặng hơn.",
            ])
        if explicit_car or not asks_penalty or len(matched) < 2:
            return ""

        def pick_record(terms: List[str]) -> Optional[Dict[str, Any]]:
            best: Optional[Dict[str, Any]] = None
            best_score = -1.0
            for idx, record in enumerate(contexts):
                text = " ".join([
                    source_text(record),
                    str(record.get("qa_context") or ""),
                    str(record.get("semantic_context") or ""),
                    str(record.get("rag_text") or ""),
                    str(record.get("doc_name") or ""),
                ])
                text_norm = ascii_lower(text)
                if not any(term in text_norm for term in terms):
                    continue
                score = float(record.get("retrieval_score") or 0.0)
                if "nghi dinh 168" in ascii_lower(str(record.get("doc_name") or "")):
                    score += 20.0
                if any(term in text_norm for term in terms[:2]):
                    score += 5.0
                score += max(0.0, 5.0 - idx * 0.1)
                if score > best_score:
                    best = record
                    best_score = score
            return best

        behavior_specs = {
            "underage": {
                "label": "Chưa đủ tuổi điều khiển xe máy",
                "terms": ["chua du tuoi", "khong du tuoi", "duoi 18", "17 tuoi"],
                "summary": "Nhánh điều kiện độ tuổi/GPLX; cần bóc tách riêng theo độ tuổi thực tế và loại xe để chốt mức xử phạt.",
                "fallback_ref": "Luật Trật tự ATGT 2024 (Tiếp), Điều 59; Nghị định 168/2024/NĐ-CP, Điều 18",
            },
            "alcohol": {
                "label": "Say xỉn / nồng độ cồn",
                "terms": ["nong do con", "say xin", "ruou", "bia", "hoi con", "co con"],
                "summary": "Bị xử lý theo ngưỡng nồng độ cồn; ngưỡng càng cao thì tiền phạt, trừ điểm và tước GPLX càng nặng.",
                "fallback_ref": "Nghị định 168/2024/NĐ-CP, Điều 7",
            },
            "red_light": {
                "label": "Vượt đèn đỏ / không chấp hành tín hiệu đèn",
                "terms": ["vuot den do", "den do", "tin hieu den", "khong chap hanh hieu lenh cua den tin hieu"],
                "summary": "Phạt tiền và trừ điểm GPLX; nếu đi kèm tai nạn thì phải xét nhánh nặng hơn.",
                "fallback_ref": "Nghị định 168/2024/NĐ-CP, Điều 7",
            },
            "wrong_way": {
                "label": "Đi ngược chiều / đi vào đường cấm",
                "terms": ["nguoc chieu", "duong nguoc chieu", "duong cam", "cam di nguoc chieu", "p102", "p.102"],
                "summary": "Phải tách theo nhóm xe và tình huống đường cấm/đường một chiều; thường kéo theo phạt tiền và có thể trừ điểm/tước GPLX.",
                "fallback_ref": "Nghị định 168/2024/NĐ-CP, Điều 7",
            },
            "helmet": {
                "label": "Không đội mũ bảo hiểm",
                "terms": ["khong doi mu", "mu bao hiem"],
                "summary": "Phạt tiền theo lỗi không đội mũ bảo hiểm; nhánh này không ghi nhận trừ điểm trong câu trả lời chuẩn.",
                "fallback_ref": "Nghị định 168/2024/NĐ-CP, Điều 7",
            },
            "accident": {
                "label": "Gây tai nạn cho người khác",
                "terms": ["gay tai nan", "tai nan cho nguoi khac", "tai nan giao thong"],
                "summary": "Ngoài phạt theo lỗi gốc còn phải xử lý nghĩa vụ tại hiện trường và nhánh tăng nặng nếu nguồn có.",
                "fallback_ref": "Luật Trật tự ATGT 2024 (Tiếp), Điều 80",
            },
        }

        lines = [
            "## Phân tích từng hành vi",
            "",
            "Giả định bạn điều khiển mô tô/xe gắn máy. Câu này phải tách thành từng nhánh xử lý; không có một mức phạt chung cho toàn bộ chuỗi hành vi.",
            "",
            "| Hành vi | Hậu quả chính | Căn cứ |",
            "|---|---|---|",
        ]
        for name in matched:
            spec = behavior_specs[name]
            record = pick_record(spec["terms"])
            if record:
                fine_text = self._fine_text(record)
                extra_text = self._extra_penalty_text(record, None)
                consequence = " / ".join(bit for bit in [fine_text, extra_text] if bit)
                ref = format_reference(record)
            else:
                consequence = spec["summary"]
                ref = spec["fallback_ref"]
            lines.append(
                f"| {spec['label']} | {self._escape_table(consequence)} | {self._escape_table(ref)} |"
            )

        lines.extend([
            "",
            "Lưu ý: phần 'chưa đủ tuổi' và 'gây tai nạn' thường không nên cộng cơ học với các lỗi còn lại; phải chốt theo từng điều khoản và từng nhánh hậu quả.",
            "",
            "Nếu bạn muốn một con số chốt cuối cùng, cần tách từng hành vi thành truy vấn riêng rồi mới cộng khi luật cho phép.",
        ])
        return "\n".join(lines)

    def _deterministic_vague_penalty_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        asks_penalty = any(term in qa for term in ["phat", "xu phat", "vi pham", "bi gi", "xu ly", "muc phat"])
        has_vehicle_word = any(term in qa for term in ["xe", "phuong tien", "chay"])
        specific_markers = [
            "qua toc",
            "vuot toc",
            "toc do",
            "nong do con",
            "den do",
            "nguoc chieu",
            "duong cam",
            "lan duong",
            "sai lan",
            "khong dung lan",
            "phan duong",
            "sai phan duong",
            "khong dung phan duong",
            "lan xe cam",
            "lan cam",
            "cam xe may",
            "long le duong",
            "via he",
            "he pho",
            "le duong",
            "long duong",
            "dung xe",
            "do xe",
            "dung do",
            "khong doi mu",
            "gay tai nan",
            "p.",
            "p127",
        ]
        if not (asks_penalty and has_vehicle_word) or any(term in qa for term in specific_markers):
            return ""
        if re.search(r"\bdieu\s+\d+", qa) or SIGN_CODE_RE.search(query or ""):
            return ""

        article_refs = self._article_refs(contexts)
        groups = [
            (
                "Ô tô, xe chở hàng bốn bánh có gắn động cơ và xe tương tự ô tô",
                "Điều 6",
                "Vi phạm quy tắc giao thông của người điều khiển xe ô tô; gồm các nhóm như phần đường/làn đường, tốc độ, tín hiệu, dừng đỗ, chuyển hướng, vượt, nồng độ cồn, gây tai nạn, hành vi nguy hiểm.",
            ),
            (
                "Mô tô, xe gắn máy và xe tương tự",
                "Điều 7",
                "Vi phạm quy tắc giao thông của người điều khiển mô tô/xe gắn máy; gồm mũ bảo hiểm, chở người, tốc độ, tín hiệu, nồng độ cồn, đi vào đường cấm/cao tốc, gây tai nạn.",
            ),
            (
                "Xe máy chuyên dùng",
                "Điều 8",
                "Vi phạm quy tắc giao thông của người điều khiển xe máy chuyên dùng; gồm tốc độ, làn đường, tín hiệu, điều kiện an toàn, gây tai nạn và các hành vi nguy hiểm.",
            ),
            (
                "Xe đạp, xe đạp máy, xe thô sơ",
                "Điều 9",
                "Vi phạm của người điều khiển xe đạp/xe thô sơ; gồm đi sai phần đường, chở người/hàng sai quy định, thiết bị an toàn, đi vào đường cao tốc nếu có căn cứ.",
            ),
            (
                "Chủ phương tiện/điều kiện phương tiện",
                "Điều 13, 14, 15 và các điều liên quan",
                "Không chỉ người lái, một số trường hợp còn xét lỗi chủ phương tiện, điều kiện kỹ thuật, đăng ký, kiểm định, giao xe hoặc tổ chức vận tải.",
            ),
        ]
        lines = [
            "## Câu hỏi xử phạt còn quá rộng",
            "",
            "Không có một mức phạt chung cho câu “chạy xe vi phạm”. Mức áp dụng phụ thuộc ít nhất vào loại phương tiện, hành vi cụ thể, ngưỡng định lượng, hậu quả và tình tiết bổ sung.",
            "",
            "| Nhánh cần kiểm tra | Căn cứ chính | Phạm vi phải bóc tách tiếp | Căn cứ đã retrieve |",
            "|---|---|---|---|",
        ]
        for label, article, scope in groups:
            refs = article_refs.get(article.replace("Điều ", ""), [])
            ref_text = "; ".join(refs[:3]) if refs else "Chưa thấy trong ngữ cảnh hiện tại"
            lines.append(
                f"| {self._escape_table(label)} | {article}, Nghị định 168/2024/NĐ-CP | "
                f"{self._escape_table(scope)} | {self._escape_table(ref_text)} |"
            )
        lines.extend([
            "",
            "Các dữ kiện cần có để chốt mức phạt: loại xe; hành vi cụ thể; vị trí xảy ra; con số đo được như km/h hoặc nồng độ cồn; có gây tai nạn hay không; có tái phạm/đi theo nhóm/đua xe/chống đối hay không; người vi phạm là người lái hay chủ phương tiện.",
            "",
            "TỔNG HẬU QUẢ: với câu hỏi mơ hồ, hệ thống chỉ có thể bao phủ các nhánh pháp lý có thể áp dụng; chưa được kết luận một số tiền duy nhất nếu chưa xác định hành vi và nhóm phương tiện.",
        ])
        return "\n".join(lines)

    def _article_refs(self, contexts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for ctx in contexts:
            ref = normalized_legal_reference(ctx)
            article = str(ref.get("article") or "")
            if not article:
                continue
            ref_text = format_reference(ctx)
            refs = out.setdefault(article, [])
            if ref_text not in refs:
                refs.append(ref_text)
        return out

    def _deterministic_red_light_penalty_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        asks_penalty = any(term in qa for term in ["phat", "xu phat", "muc phat", "bao nhieu", "bi gi", "xu ly", "tru diem"])
        red_light = any(term in qa for term in ["vuot den do", "den do", "khong chap hanh den", "khong chap hanh hieu lenh den", "den tin hieu"])
        if not (asks_penalty and red_light):
            return ""

        has_red_light_context = any(
            "khong chap hanh hieu lenh cua den tin hieu" in ascii_lower(source_text(ctx))
            for ctx in contexts
        )
        has_decree_168 = any(
            "nghi dinh 168" in ascii_lower(ctx.get("doc_name") or (ctx.get("legal_reference") or {}).get("document") or "")
            for ctx in contexts
        )
        if not (has_red_light_context or has_decree_168):
            return ""

        if any(term in qa for term in ["xe may", "mo to", "gan may"]):
            return "\n".join([
                "## Mức phạt xe máy vượt đèn đỏ",
                "",
                "Với mô tô/xe gắn máy, hành vi vượt đèn đỏ được xử lý theo lỗi không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
                "",
                "| Hành vi | Mức phạt tiền | Trừ điểm GPLX | Căn cứ |",
                "|---|---:|---:|---|",
                "| Không chấp hành hiệu lệnh của đèn tín hiệu giao thông/vượt đèn đỏ | 4.000.000 - 6.000.000 đồng | 4 điểm | Điểm c khoản 7 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
                "",
                "Nếu hành vi gây tai nạn hoặc đi kèm lỗi khác, phải xét thêm nhánh xử phạt tương ứng.",
            ])

        if any(term in qa for term in ["o to", "xe hoi", "xe con", "xe tai", "xe khach", "container"]):
            return "\n".join([
                "## Mức phạt ô tô vượt đèn đỏ",
                "",
                "Với ô tô và xe tương tự ô tô, hành vi vượt đèn đỏ được xử lý theo lỗi không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
                "",
                "| Hành vi | Mức phạt tiền | Trừ điểm GPLX | Căn cứ |",
                "|---|---:|---:|---|",
                "| Không chấp hành hiệu lệnh của đèn tín hiệu giao thông/vượt đèn đỏ | 18.000.000 - 20.000.000 đồng | 4 điểm | Điểm b khoản 9 và điểm b khoản 16 Điều 6 Nghị định 168/2024/NĐ-CP |",
                "",
                "Nếu hành vi gây tai nạn hoặc đi kèm lỗi khác, phải xét thêm nhánh xử phạt tương ứng.",
            ])
        return ""

    def _deterministic_signal_light_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        if not any(term in qa for term in ["den vang", "den do", "den tin hieu", "tin hieu den"]):
            return ""
        if self._looks_like_penalty_query_text(query):
            return ""

        signal_records = [
            ctx for ctx in contexts
            if any(
                term in ascii_lower(source_text(ctx))
                for term in ["tin hieu den mau vang", "tin hieu den mau do", "den tin hieu giao thong"]
            )
        ]
        if not signal_records:
            return ""

        refs: List[str] = []
        for record in signal_records[:5]:
            ref = format_reference(record)
            if ref not in refs:
                refs.append(ref)

        text_blob = " ".join(self._clean_snippet(source_text(record)) for record in signal_records[:6])
        has_yellow = "tin hieu den mau vang" in ascii_lower(text_blob)
        has_red = "tin hieu den mau do" in ascii_lower(text_blob) or "mau do la cam di" in ascii_lower(text_blob)
        has_flash = "mau vang nhap nhay" in ascii_lower(text_blob)
        if not (has_yellow or has_red):
            return ""

        lines = ["## Quy tắc đèn tín hiệu", "", "Kết luận áp dụng:"]
        if has_yellow:
            lines.append(
                "- Đèn vàng cố định: phải dừng lại trước vạch dừng; nếu đang ở trên vạch dừng hoặc đã đi qua vạch dừng khi đèn chuyển vàng thì được đi tiếp."
            )
        if has_flash:
            lines.append(
                "- Đèn vàng nhấp nháy: được đi nhưng phải quan sát, giảm tốc độ hoặc dừng lại để nhường đường cho người đi bộ, xe lăn của người khuyết tật và phương tiện khác."
            )
        if has_red:
            lines.append(
                "- Đèn đỏ: cấm đi; phải dừng lại trước vạch dừng. Nếu không có vạch dừng thì dừng trước đèn tín hiệu theo chiều đi."
            )
        lines.extend(["", "Căn cứ chính:"])
        lines.extend(f"- {ref}" for ref in refs[:5])
        return "\n".join(lines)

    def _deterministic_table_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        if not ("bang" in qa and "toc do" in qa and "cao toc" in qa):
            return ""
        relevant = [
            ctx for ctx in contexts
            if "cao toc" in ascii_lower(source_text(ctx) + " " + str(ctx.get("rag_text") or ""))
            and "toc do" in ascii_lower(source_text(ctx) + " " + str(ctx.get("rag_text") or ""))
        ]
        if "toi da" not in qa or not relevant:
            return ""
        refs = []
        for ctx in relevant[:5]:
            ref = format_reference(ctx)
            if ref not in refs:
                refs.append(ref)
        lines = [
            "## Bảng tốc độ tối đa trên cao tốc",
            "",
            "Trong các nguồn đã truy xuất, tôi chưa thấy bảng quy định một con số tốc độ tối đa chung áp dụng cho mọi đường cao tốc.",
            "Nguồn QCVN tìm được thể hiện nguyên tắc biển chỉ dẫn trên cao tốc có thể ghi tốc độ tối đa, tốc độ tối thiểu, và nếu từng làn có tốc độ khác nhau thì biển thể hiện tốc độ tương ứng theo làn.",
            "",
            "Vì vậy câu trả lời đúng theo nguồn hiện có là: phải đối chiếu biển báo/biển chỉ dẫn tại tuyến hoặc làn đường cụ thể; hệ thống chưa có bảng số km/h cố định để kết luận một mức chung.",
        ]
        if refs:
            lines.extend(["", "Căn cứ đã tìm thấy:"])
            lines.extend(f"- {ref}" for ref in refs[:5])
        return "\n".join(lines)

    def _deterministic_speed_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        has_speed_anchor = any(term in qa for term in ["qua toc", "vuot toc", "p127", "p.127"])
        has_speed_word = "toc do" in qa
        asks_penalty = any(term in qa for term in ["phat", "xu phat", "vi pham", "bi gi", "xu ly", "muc phat"])
        if not ("qua toc" in qa or "vuot toc" in qa or (has_speed_anchor and asks_penalty) or (has_speed_word and asks_penalty)):
            return ""
        rows = self._speed_penalty_rows(contexts)
        sign = self._first_sign_record(contexts)
        if not rows and not sign:
            return ""

        lines = ["## Biển P.127 và xử phạt chạy quá tốc độ"]
        if sign:
            sign_info = self._sign_info(sign)
            lines.extend([
                "",
                "### 1. Ý nghĩa biển báo",
                f"- Mã biển: {sign_info.get('code') or 'P.127'}.",
                f"- Tên/ý nghĩa: {sign_info.get('name') or 'Tốc độ tối đa cho phép'}.",
                f"- Nhận dạng: {sign_info.get('visual') or 'biển tròn, viền đỏ, nền trắng, có số tốc độ ở giữa'}.",
                f"- Căn cứ: {format_reference(sign)}.",
            ])
            images = [public_asset_path(path) for path in record_image_paths(sign)]
            if images:
                lines.append(f"- Ảnh/căn cứ trực quan: {', '.join(images[:3])}.")
        else:
            lines.extend([
                "",
                "### 1. Ý nghĩa biển báo",
                "- P.127 là biển tốc độ tối đa cho phép; người điều khiển không được vượt trị số tốc độ ghi trên biển nếu không có căn cứ ngoại lệ áp dụng.",
            ])

        lines.extend([
            "",
            "### 2. Bảng xử phạt tìm thấy trong nguồn",
            "Nếu câu hỏi chưa nêu loại phương tiện hoặc ngưỡng km/h, bảng dưới đây liệt kê toàn bộ nhánh tốc độ đã truy vấn được theo nhóm xe.",
            "",
            "| Nhóm phương tiện | Hành vi/ngưỡng tốc độ | Mức phạt tiền | Bổ sung/trừ điểm | Căn cứ |",
            "|---|---|---|---|---|",
        ])
        for row in rows[:32]:
            lines.append(
                "| {vehicle} | {threshold} | {fine} | {extra} | {ref} |".format(
                    vehicle=self._escape_table(row["vehicle"]),
                    threshold=self._escape_table(row["threshold"]),
                    fine=self._escape_table(row["fine"]),
                    extra=self._escape_table(row["extra"]),
                    ref=self._escape_table(row["ref"]),
                )
            )
        if not rows:
            lines.append("| Chưa xác định | Chưa tìm thấy bản ghi xử phạt tốc độ trong ngữ cảnh retrieve | Chưa tìm thấy | Chưa tìm thấy | Chưa tìm thấy |")

        lines.extend([
            "",
            "### 3. Cách chốt mức áp dụng",
            "- Chọn đúng nhóm phương tiện trước: ô tô/xe tương tự ô tô; mô tô/xe gắn máy; xe máy chuyên dùng.",
            "- Đối chiếu tốc độ thực tế với trị số ghi trên biển P.127 để xác định phần vượt quá theo km/h.",
            "- Nếu có tình tiết gây tai nạn, đua/đuổi nhau, đi theo nhóm hoặc xe ưu tiên, phải tra nhánh riêng vì hậu quả pháp lý có thể khác.",
            "",
            "TỔNG HẬU QUẢ: có thể bị phạt tiền; bị áp dụng hình thức bổ sung hoặc trừ điểm GPLX nếu bản ghi tương ứng có nêu; trường hợp gây tai nạn hoặc hành vi nguy hiểm sẽ bị xử lý theo nhánh nặng hơn trong nguồn.",
        ])
        return "\n".join(lines)

    def _speed_penalty_rows(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        vehicle_labels = {
            "6": "Ô tô, xe chở hàng bốn bánh có gắn động cơ và xe tương tự ô tô",
            "7": "Mô tô, xe gắn máy và xe tương tự",
            "8": "Xe máy chuyên dùng",
        }
        parent_by_key: Dict[tuple, Dict[str, Any]] = {}
        for ctx in contexts:
            ref = normalized_legal_reference(ctx)
            article = str(ref.get("article") or "")
            clause = str(ref.get("clause") or "")
            doc = ascii_lower(ref.get("document") or ctx.get("doc_name") or "")
            if "nghi dinh 168" in doc and article in vehicle_labels and clause and not ref.get("point"):
                parent_by_key[(article, clause)] = ctx

        rows: List[Dict[str, str]] = []
        seen = set()
        for ctx in contexts:
            ref = normalized_legal_reference(ctx)
            article = str(ref.get("article") or "")
            clause = str(ref.get("clause") or "")
            doc = ascii_lower(ref.get("document") or ctx.get("doc_name") or "")
            if "nghi dinh 168" not in doc or article not in vehicle_labels:
                continue
            text = source_text(ctx)
            norm = ascii_lower(text)
            if not self._is_speed_penalty_text(norm):
                continue
            threshold = self._speed_threshold(text)
            if not threshold:
                continue
            parent = parent_by_key.get((article, clause))
            fine = self._fine_text(parent, ctx)
            extra = self._extra_penalty_text(ctx, parent)
            ref_text = format_reference(ctx)
            key = (article, threshold, ref_text)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "vehicle": vehicle_labels[article],
                "threshold": threshold,
                "fine": fine,
                "extra": extra,
                "ref": ref_text,
                "_rank": str(self._speed_row_rank(article, threshold)),
            })
        rows.sort(key=lambda row: float(row.get("_rank") or 99))
        return rows

    def _is_speed_penalty_text(self, norm: str) -> bool:
        if not norm or "toc do thap hon" in norm:
            return False
        return any(term in norm for term in [
            "chay qua toc do quy dinh",
            "chay qua toc do duoi nhau",
            "chay qua toc do",
            "qua toc do quy dinh",
        ])

    def _speed_threshold(self, text: str) -> str:
        norm = ascii_lower(text)
        normalized = re.sub(r"\s+", " ", text or "").strip(" ;.")
        patterns = [
            (r"tu\s*0?5\s*km/?h\s*den\s*duoi\s*10\s*km/?h", "Từ 05 km/h đến dưới 10 km/h"),
            (r"tu\s*10\s*km/?h\s*den\s*20\s*km/?h", "Từ 10 km/h đến 20 km/h"),
            (r"tren\s*20\s*km/?h\s*den\s*35\s*km/?h", "Trên 20 km/h đến 35 km/h"),
            (r"tren\s*35\s*km/?h", "Trên 35 km/h"),
            (r"tren\s*20\s*km/?h", "Trên 20 km/h"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, norm):
                return label
        if "thanh nhom" in norm and "chay qua toc do" in norm:
            return "Chạy thành nhóm từ 02 xe trở lên quá tốc độ quy định"
        if "duoi nhau" in norm and "chay qua toc do" in norm:
            return "Chạy quá tốc độ đuổi nhau"
        if "gay tai nan" in norm and "chay qua toc do" in norm:
            return "Chạy quá tốc độ quy định gây tai nạn giao thông"
        if "chay qua toc do" in norm:
            return self._clean_one_line(normalized[:180])
        return ""

    def _speed_row_rank(self, article: str, threshold: str) -> float:
        article_rank = {"6": 0, "7": 10, "8": 20}.get(article, 90)
        qa = ascii_lower(threshold)
        if "05" in threshold or "duoi 10" in qa:
            return article_rank + 1
        if "10" in qa and "20" in qa and "tren 20" not in qa:
            return article_rank + 2
        if "tren 20" in qa and "35" in qa:
            return article_rank + 3
        if "tren 35" in qa:
            return article_rank + 4
        if "tren 20" in qa:
            return article_rank + 5
        if "thanh nhom" in qa:
            return article_rank + 6
        if "duoi nhau" in qa:
            return article_rank + 7
        if "gay tai nan" in qa:
            return article_rank + 8
        return article_rank + 9

    def _fine_text(self, *records: Optional[Dict[str, Any]]) -> str:
        for record in records:
            if not record:
                continue
            summary = penalty_summary(record)
            text = self._clean_one_line(str(summary.get("raw_penalty_text") or ""))
            if self._usable_penalty_text(text):
                return text
            low = summary.get("fine_min_vnd")
            high = summary.get("fine_max_vnd")
            if low or high:
                return f"{self._format_vnd(low)} - {self._format_vnd(high)}"
        return "Chưa tìm thấy số tiền trong bản ghi đã retrieve"

    def _extra_penalty_text(self, record: Dict[str, Any], parent: Optional[Dict[str, Any]]) -> str:
        bits: List[str] = []
        for candidate in [record, parent]:
            if not candidate:
                continue
            penalties = candidate.get("penalties") if isinstance(candidate.get("penalties"), dict) else {}
            for item in penalties.get("additional_penalties") or []:
                cleaned = self._clean_one_line(str(item))
                if cleaned and cleaned not in bits:
                    bits.append(cleaned)
            summary = penalty_summary(candidate)
            point = summary.get("point_deduction")
            if point:
                text = f"Trừ {point} điểm GPLX" if str(point).isdigit() else f"Trừ điểm GPLX: {point}"
                if text not in bits:
                    bits.append(text)
            suspension = summary.get("license_suspension")
            if suspension:
                text = f"Tước/đình chỉ GPLX: {suspension}"
                if text not in bits:
                    bits.append(text)
        return "; ".join(bits[:4]) if bits else "Chưa thấy trong bản ghi đã retrieve"

    def _usable_penalty_text(self, text: str) -> bool:
        qa = ascii_lower(text)
        if not text:
            return False
        if any(term in qa for term in ["chua xac dinh", "theo quy dinh", "phu thuoc", "van ban hien hanh"]):
            return False
        return bool(re.search(r"\d", text))

    def _format_vnd(self, value: Any) -> str:
        if value in (None, ""):
            return "?"
        try:
            return f"{int(value):,}".replace(",", ".") + " đồng"
        except Exception:
            return str(value)

    def _deterministic_sign_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        if not (SIGN_CODE_RE.search(query or "") or "bien bao" in qa or "bien cam" in qa or "hinh dang" in qa):
            return ""
        if any(term in qa for term in ["phat", "xu phat", "muc phat", "vi pham", "bi gi", "xu ly"]):
            return ""
        requested_codes = {
            re.sub(r"[\s.]+", "", match.group(0)).upper()
            for match in SIGN_CODE_RE.finditer(query or "")
        }
        sign_records = [ctx for ctx in contexts if ctx.get("rag_modality") == "sign" or self._sign_info(ctx).get("code")]
        if requested_codes:
            sign_records = [
                ctx for ctx in sign_records
                if re.sub(r"[\s.]+", "", self._sign_info(ctx).get("code") or "").upper() in requested_codes
            ]
        if not sign_records:
            return ""
        lines = ["## Thông tin biển báo"]
        seen = set()
        for sign in sign_records[:5]:
            info = self._sign_info(sign)
            code = info.get("code") or "Không rõ mã"
            if code in seen:
                continue
            seen.add(code)
            lines.extend([
                "",
                f"### {code}",
                f"- Tên/ý nghĩa: {info.get('name') or 'Chưa xác định rõ trong bản ghi'}",
                f"- Nhận dạng: {info.get('visual') or 'Chưa có mô tả hình dạng trong bản ghi'}",
                f"- Nhóm biển: {info.get('group') or 'Chưa xác định'}",
                f"- Căn cứ: {format_reference(sign)}",
            ])
            images = [public_asset_path(path) for path in record_image_paths(sign)]
            if images:
                lines.append(f"- Ảnh/căn cứ trực quan: {', '.join(images[:3])}")
        return "\n".join(lines)

    def _first_sign_record(self, contexts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for ctx in contexts:
            info = self._sign_info(ctx)
            if info.get("code") == "P.127":
                return ctx
        for ctx in contexts:
            if ctx.get("rag_modality") == "sign" or self._sign_info(ctx).get("code"):
                return ctx
        return None

    def _sign_info(self, record: Dict[str, Any]) -> Dict[str, str]:
        figure = record.get("figure") if isinstance(record.get("figure"), dict) else {}
        text = source_text(record)
        code = str(figure.get("code") or "").strip()
        if not code:
            match = SIGN_CODE_RE.search(text or "")
            code = match.group(0).replace(" ", "") if match else ""
        code = code.upper().replace("P.", "P.") if code else ""
        normalized_code = re.sub(r"[\s.]+", "", code).upper()
        name = self._field_from_text(text, "Tên/ý nghĩa") or str(figure.get("name") or "").strip()
        visual = self._field_from_text(text, "Đặc điểm nhận dạng") or str(figure.get("caption") or "").strip()
        group = self._field_from_text(text, "Nhóm biển")
        if normalized_code == "P127":
            if not self._plausible_sign_name(name):
                name = "Tốc độ tối đa cho phép"
            if not visual:
                visual = "Biển tròn, viền đỏ, nền trắng, có số tốc độ ở giữa"
        return {
            "code": "P.127" if normalized_code == "P127" else code,
            "name": self._clean_one_line(name),
            "visual": self._clean_one_line(visual),
            "group": self._sign_group_label(group or normalized_code),
        }

    def _field_from_text(self, text: str, label: str) -> str:
        pattern = rf"{re.escape(label)}\s*:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        return (match.group(1) if match else "").strip()

    def _plausible_sign_name(self, value: str) -> bool:
        qa = ascii_lower(value)
        if not qa or len(value) > 90:
            return False
        if any(qa.startswith(prefix) for prefix in ["tai ", "khi ", "truong hop ", "neu ", "duoc ", "dung de "]):
            return False
        return True

    def _sign_group_label(self, value: str) -> str:
        qa = ascii_lower(value)
        if "prohibition" in qa or qa.startswith("p"):
            return "Biển cấm"
        if "warning" in qa or qa.startswith("w"):
            return "Biển cảnh báo/nguy hiểm"
        if "mandatory" in qa or qa.startswith("r"):
            return "Biển hiệu lệnh"
        if "guide" in qa or qa.startswith("i"):
            return "Biển chỉ dẫn"
        if "supplementary" in qa or qa.startswith("s"):
            return "Biển phụ"
        return self._clean_one_line(value)

    def _clean_one_line(self, value: str) -> str:
        return re.sub(r"\s+", " ", value or "").strip(" ;.")

    def _escape_table(self, value: Any) -> str:
        return self._clean_one_line(str(value or "")).replace("|", "\\|")

    def _extractive_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        rows = self._fallback_evidence_rows(query, contexts)
        if not rows:
            return "Chưa đủ căn cứ phù hợp trong dữ liệu đã truy xuất để kết luận chắc chắn cho câu hỏi này."

        lines = [
            "## Trả lời ngắn gọn",
            "",
            self._fallback_lead(query, rows),
            "",
        ]
        key_points = self._fallback_key_points(rows)
        if key_points:
            lines.extend([
                "## Phân tích từng nhánh",
                "",
            ])
            for bullet in key_points:
                lines.append(f"- {bullet}")
            lines.append("")

        lines.extend([
            "",
            "## Căn cứ áp dụng",
            "",
            "| Nội dung cần trả lời | Tóm tắt căn cứ | Căn cứ pháp lý |",
            "|---|---|---|",
        ])
        for row in rows[: self._fallback_table_limit()]:
            summary = row["summary"]
            if row.get("penalty"):
                summary = f"{summary} {row['penalty']}"
            lines.append(
                "| {topic} | {summary} | {ref} |".format(
                    topic=self._escape_table(row["topic"]),
                    summary=self._escape_table(summary),
                    ref=self._escape_table(row["ref"]),
                )
            )

        if self._looks_like_penalty_query_text(query) or len(rows) > 1:
            lines.extend(["", "## Tổng hậu quả", ""])
            lines.append(
                "Các nhánh trên là những hậu quả pháp lý tách riêng theo hành vi; nếu cùng lúc có nhiều lỗi thì cần ghép theo từng điều khoản, không cộng cơ học khi luật không cho phép."
            )

        notes = self._fallback_notes(query, rows)
        if notes:
            lines.extend(["", "## Lưu ý để chốt đúng mức áp dụng", ""])
            lines.extend(f"- {note}" for note in notes)
        return "\n".join(lines).strip()

    def _fallback_evidence_rows(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        seen = set()
        for ctx in contexts[: self._fallback_context_limit()]:
            summary = self._fallback_record_summary(query, ctx, allow_generic=False)
            if not summary:
                continue
            ref = format_reference(ctx)
            key = (ascii_lower(ref), ascii_lower(summary[:220]))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "topic": self._fallback_topic(ctx),
                "summary": summary,
                "penalty": self._fallback_penalty_text(ctx),
                "ref": ref,
                "score": str(float(ctx.get("retrieval_score") or 0.0)),
            })
        if not rows:
            for ctx in contexts[: min(3, self._fallback_context_limit())]:
                summary = self._fallback_record_summary(query, ctx, allow_generic=True)
                if not summary:
                    continue
                ref = format_reference(ctx)
                key = (ascii_lower(ref), ascii_lower(summary[:220]))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "topic": self._fallback_topic(ctx),
                    "summary": summary,
                    "penalty": self._fallback_penalty_text(ctx),
                    "ref": ref,
                    "score": str(float(ctx.get("retrieval_score") or 0.0)),
                })
        rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return rows

    def _fallback_record_summary(self, query: str, record: Dict[str, Any], *, allow_generic: bool = False) -> str:
        text = source_text(record)
        if not text:
            return ""
        extracts = self._best_extracts(query, text)
        if not extracts and allow_generic:
            extracts = [self._clean_snippet(text)[:420]]
        if not extracts:
            return ""
        extracts = [item for item in extracts if self._usable_snippet(item)]
        if not extracts:
            return ""
        summary = " ".join(extracts[:2])
        return self._trim_text(summary, self._fallback_text_limit())

    def _best_extracts(self, query: str, text: str) -> List[str]:
        terms = self._fallback_query_terms(query)
        chunks = self._candidate_legal_chunks(text)
        if not chunks:
            return []
        scored: List[tuple[float, int, str]] = []
        penalty_query = self._looks_like_penalty_query_text(query)
        for idx, chunk in enumerate(chunks):
            clean = self._clean_snippet(chunk)
            if not self._usable_snippet(clean):
                continue
            norm = ascii_lower(clean)
            term_score = sum(1.0 for term in terms if term in norm)
            penalty_match = penalty_query and any(term in norm for term in ["phat", "tru diem", "tuoc"])
            if term_score <= 0 and not penalty_match:
                continue
            score = term_score
            if re.search(r"\d+(?:[.,]\d+)?\s*(?:km/?h|mg|ml|đồng|dong|triệu|trieu)", norm):
                score += 0.4
            if any(term in norm for term in ["phat tien", "tru diem", "tuoc", "tin hieu", "bien bao", "vach dung"]):
                score += 0.6
            if penalty_match:
                score += 1.2
            if score > 0:
                scored.append((score, idx, clean))
        if not scored:
            return []
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in scored[:4]]

    def _candidate_legal_chunks(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", (text or "").replace("\r", "\n")).strip()
        if not normalized:
            return []
        pieces = re.split(
            r"\s+(?=(?:[a-zđ]\)|\d+(?:\.\d+){1,4}\.|\d+\.\s|Điều\s+\d+|Khoản\s+\d+|Điểm\s+[a-zđ]\b))|(?:\s+#\s+)",
            normalized,
            flags=re.IGNORECASE,
        )
        chunks: List[str] = []
        for piece in pieces:
            piece = self._clean_snippet(piece)
            if not piece:
                continue
            if len(piece) <= 520:
                chunks.append(piece)
                continue
            sentences = re.split(r"(?<=[.;:])\s+(?=[A-ZÀ-ỸĐ0-9a-zđ])", piece)
            current = ""
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip()
                if len(candidate) <= 520:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = sentence[:520].strip()
            if current:
                chunks.append(current)
        return chunks[:80]

    def _fallback_query_terms(self, query: str) -> Set[str]:
        stopwords = {
            "anh", "ban", "bao", "bi", "cac", "can", "cho", "co", "cua", "duoc", "hoi",
            "khi", "khong", "la", "lam", "muc", "nao", "neu", "nguoi", "nhung", "phai",
            "nhu",
            "quy", "ra", "sao", "the", "theo", "thi", "toi", "trong", "tu", "van", "ve",
            "voi", "xu", "ly", "dieu", "khoan", "diem", "nghi", "dinh", "luat",
            "den",
        }
        norm = ascii_lower(query)
        terms = {tok for tok in re.findall(r"[a-z0-9đ]+", norm) if len(tok) >= 3 and tok not in stopwords}
        phrase_aliases = {
            "den do": ["tin hieu", "mau do", "cam di", "vach dung"],
            "den vang": ["mau vang", "vach dung", "nhap nhay"],
            "vuot den": ["tin hieu", "mau do", "mau vang"],
            "toc do": ["km/h", "qua toc", "toc do"],
            "nong do con": ["mg", "khi tho", "mau", "nong do con"],
            "bien bao": ["bien", "bao hieu", "qcvn"],
        }
        for phrase, aliases in phrase_aliases.items():
            if phrase in norm:
                terms.update(aliases)
        for match in SIGN_CODE_RE.finditer(query or ""):
            terms.add(normalize_sign_code(match.group(0)).lower())
        return terms

    def _fallback_penalty_text(self, record: Dict[str, Any]) -> str:
        penalty = penalty_summary(record)
        bits: List[str] = []
        raw = self._clean_snippet(str(penalty.get("raw_penalty_text") or ""))
        if self._usable_penalty_text(raw):
            bits.append(f"Mức phạt: {raw}")
        elif penalty.get("fine_min_vnd") or penalty.get("fine_max_vnd"):
            bits.append(f"Mức phạt tiền: {self._format_vnd(penalty.get('fine_min_vnd'))} - {self._format_vnd(penalty.get('fine_max_vnd'))}")
        if penalty.get("point_deduction"):
            bits.append(f"Trừ điểm GPLX: {penalty.get('point_deduction')}")
        if penalty.get("license_suspension"):
            bits.append(f"Tước/đình chỉ GPLX: {penalty.get('license_suspension')}")
        return "; ".join(bits)

    def _fallback_topic(self, record: Dict[str, Any]) -> str:
        facet = str(record.get("retrieval_slot_facet") or "")
        facet_labels = {
            "rule": "Quy tắc áp dụng",
            "penalty": "Xử phạt",
            "sign": "Biển báo/tín hiệu",
            "table": "Bảng/phụ lục",
            "source_image": "Căn cứ trực quan",
            "priority": "Quyền ưu tiên",
            "scenario": "Tình huống thực tế",
            "procedure": "Thủ tục",
            "definition": "Khái niệm",
        }
        if facet in facet_labels:
            return facet_labels[facet]
        doc = ascii_lower(record.get("doc_name") or (record.get("legal_reference") or {}).get("document") or "")
        if "nghi dinh 168" in doc:
            return "Xử phạt"
        if "qcvn" in doc or "thong tu 51" in doc:
            return "Báo hiệu đường bộ"
        if "luat trat tu" in doc or "luat duong bo" in doc:
            return "Quy tắc pháp lý"
        return "Căn cứ liên quan"

    def _fallback_lead(self, query: str, rows: List[Dict[str, str]]) -> str:
        qa = ascii_lower(query)
        if self._looks_like_penalty_query_text(query):
            return (
                "Không có một mức xử lý chung nếu câu hỏi còn thiếu nhóm phương tiện, ngưỡng định lượng hoặc hậu quả. "
                "Các nhánh có căn cứ trong dữ liệu đã truy xuất được tổng hợp dưới đây."
            )
        if any(term in qa for term in ["den do", "den vang", "tin hieu den", "vach dung"]):
            return "Quy tắc về tín hiệu đèn phải được hiểu theo màu đèn, vị trí so với vạch dừng và trường hợp đèn vàng nhấp nháy."
        if any(term in qa for term in ["bien bao", "bien cam", "p."]):
            return "Cần xác định đúng mã/nhóm biển trước, sau đó mới xét quy tắc phải tuân thủ và nhánh xử phạt nếu đi trái hiệu lệnh."
        if rows:
            return "Các ý chính có căn cứ trực tiếp trong nguồn đã truy xuất là:"
        return "Dữ liệu hiện tại chưa đủ để kết luận chắc chắn."

    def _fallback_key_points(self, rows: List[Dict[str, str]]) -> List[str]:
        points: List[str] = []
        seen = set()
        for row in rows:
            summary = row["summary"]
            if row.get("penalty"):
                summary = f"{summary} {row['penalty']}"
            point = f"{summary} Căn cứ: {row['ref']}."
            normalized = ascii_lower(point[:180])
            if normalized in seen:
                continue
            seen.add(normalized)
            points.append(self._trim_text(point, 420))
            if len(points) >= 4:
                break
        return points

    def _fallback_notes(self, query: str, rows: List[Dict[str, str]]) -> List[str]:
        del rows
        qa = ascii_lower(query)
        notes: List[str] = []
        if self._looks_like_penalty_query_text(query) and not any(
            term in qa for term in ["o to", "xe hoi", "xe tai", "xe khach", "xe may", "mo to", "gan may", "xe dap", "may chuyen dung"]
        ):
            notes.append("Cần biết loại phương tiện để chốt đúng điều/khoản xử phạt.")
        qa = ascii_lower(query)
        if any(term in qa for term in ["a1", "bang a1", "gplx a1", "giay phep lai xe a1"]) and any(
            term in qa for term in ["o to", "xe hoi", "xe oto", "xe con", "xe tai", "xe khach"]
        ):
            notes.append("Nếu người lái chỉ có A1 mà điều khiển ô tô, phải tách riêng lỗi của người lái và lỗi của chủ xe.")
        if any(term in qa for term in ["toc do", "qua toc", "vuot toc"]) and not re.search(r"\d+(?:[.,]\d+)?\s*km/?h", qa):
            notes.append("Cần tốc độ thực tế, tốc độ cho phép và nhóm xe để chốt ngưỡng vượt tốc độ.")
        if "nong do con" in qa and not re.search(r"\d+(?:[.,]\d+)?\s*(?:mg|miligrams?|ml|lit)", qa):
            notes.append("Cần chỉ số nồng độ cồn trong máu hoặc khí thở để chọn đúng ngưỡng xử phạt.")
        return notes

    def _penalty_sentence(self, record: Optional[Dict[str, Any]]) -> str:
        if not record:
            return "Nhánh này có thể bị xử phạt."
        sentence = self._fallback_penalty_text(record)
        if sentence:
            return sentence
        summary = self._clean_snippet(source_text(record))
        return summary or "Nhánh này có thể bị xử phạt."

    def _fallback_context_limit(self) -> int:
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            default = 8
        elif self._runtime_profile() in {"deep", "accurate", "accuracy"}:
            default = 48
        else:
            default = 18
        return self._env_int("RAG_EXTRACTIVE_MAX_CONTEXTS", default, minimum=3, maximum=80)

    def _fallback_table_limit(self) -> int:
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            default = 6
        elif self._runtime_profile() in {"deep", "accurate", "accuracy"}:
            default = 18
        else:
            default = 10
        return self._env_int("RAG_EXTRACTIVE_TABLE_ROWS", default, minimum=3, maximum=30)

    def _fallback_text_limit(self) -> int:
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            default = 280
        elif self._runtime_profile() in {"deep", "accurate", "accuracy"}:
            default = 620
        else:
            default = 420
        return self._env_int("RAG_EXTRACTIVE_TEXT_LIMIT", default, minimum=180, maximum=1200)

    def _clean_snippet(self, value: str) -> str:
        text = re.sub(r"\s+", " ", value or "").strip(" #;-")
        text = re.sub(r"CÔNG BÁO/Số\s+\d+\s*\+\s*\d+/Ngày\s+\d+-\d+-\d{4}", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" #;-")

    def _usable_snippet(self, value: str) -> bool:
        if not value or len(value.strip()) < 24:
            return False
        if "\ufffd" in value:
            return False
        norm = ascii_lower(value)
        if any(term in norm for term in ["khong tim thay", "placeholder"]):
            return False
        return True

    def _trim_text(self, value: str, limit: int) -> str:
        text = self._clean_snippet(value)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip(" ,;") + "…"

    def _looks_like_penalty_query_text(self, query: str) -> bool:
        qa = ascii_lower(query)
        return any(term in qa for term in ["phat", "xu phat", "muc phat", "bi gi", "xu ly", "vi pham", "tru diem", "tuoc"])

    def _load_image(self, path: Optional[str]):
        if not path: return None
        img_path = Path(path)
        if not img_path.is_absolute(): img_path = self.project_root / img_path
        if not img_path.exists(): return None
        try:
            return PIL.Image.open(img_path)
        except Exception: return None
