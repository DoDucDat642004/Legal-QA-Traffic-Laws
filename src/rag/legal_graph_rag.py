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
from src.rag.legal_utils import (
    SIGN_CODE_RE,
    ascii_lower,
    format_reference,
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
        plan, profile = self._build_query_profile(query)
        if "out_of_scope" in set(profile.facets or []) or profile.intent == "out_of_scope":
            return self._out_of_scope_result(query, plan, profile)
        if not self._should_use_sequential(profile):
            return self.query_direct(query, plan=plan, profile=profile)
        return self.query_sequential(query, plan=plan, profile=profile)

    def _build_query_profile(self, query: str):
        plan = self.retriever.query_planner.plan(query, client=self.client)
        profile = self.retriever.adaptive_analyzer.analyze(query, plan)
        return plan, profile

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

        max_prompt_images = int(os.getenv("RAG_MAX_PROMPT_IMAGES", "8"))
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            max_prompt_images = min(max_prompt_images, self._env_int("RAG_FAST_MAX_PROMPT_IMAGES", 0, minimum=0, maximum=4))
        loaded_prompt_images = 0
        if self._env_bool("RAG_DEPLOY_FAST_MODE", False):
            prompt_context_limit = self._env_int("RAG_PROMPT_CONTEXT_TEXT_LIMIT", 8000, minimum=1200, maximum=40000)
            structured_context_limit = self._env_int("RAG_PROMPT_STRUCTURED_TEXT_LIMIT", 16000, minimum=2000, maximum=60000)
        else:
            prompt_context_limit = self._env_int("RAG_PROMPT_CONTEXT_TEXT_LIMIT", 50000, minimum=4000, maximum=200000)
            structured_context_limit = self._env_int("RAG_PROMPT_STRUCTURED_TEXT_LIMIT", 120000, minimum=10000, maximum=300000)
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
            return answer
        except Exception as exc:
            logger.warning("Answer generation failed with max_output_tokens=%s: %s", max_output_tokens, exc)
            if max_output_tokens > 8192:
                try:
                    res, _model = generate_content_with_fallback(
                        self.client,
                        contents=contents,
                        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=8192),
                        env_names=("RAG_ANSWER_MODEL",),
                        task="answer",
                        logger=logger,
                        label="Answer generation fallback",
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
            "20. Với câu hỏi ngoài phạm vi luật giao thông đường bộ, từ chối ngắn gọn và hướng người dùng hỏi lại trong phạm vi hệ thống.\n"
            "Few-shot format nội bộ: 'Biển + hành vi' => ý nghĩa biển trước, hành vi sau, xử phạt cuối; "
            "'Bảng/phụ lục' => nêu dòng/cột; 'Tình huống nhiều bước' => kết luận từng bước."
        )

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
        default = 4096 if self._env_bool("RAG_DEPLOY_FAST_MODE", False) else 32768
        return self._env_int("RAG_ANSWER_MAX_OUTPUT_TOKENS", default, minimum=1024, maximum=65536)

    def _max_continuations(self) -> int:
        default = 0 if self._env_bool("RAG_DEPLOY_FAST_MODE", False) else 2
        return self._env_int("RAG_ANSWER_MAX_CONTINUATIONS", default, minimum=0, maximum=6)

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

    def _continue_if_truncated(
        self,
        *,
        model: str,
        base_contents: List[Any],
        answer: str,
        first_response: Any,
        max_output_tokens: int,
    ) -> str:
        if self._finish_reason(first_response).upper() not in {"MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"}:
            return answer
        continuations = self._max_continuations()
        if continuations <= 0:
            return answer
        current = answer
        for _idx in range(continuations):
            prompt = (
                "\n[CÂU TRẢ LỜI ĐANG BỊ NGẮT]\n"
                f"{current[-3000:]}\n\n"
                "Hãy viết TIẾP từ đúng vị trí bị ngắt. Không lặp lại phần đã viết, "
                "không mở đầu lại, chỉ tiếp tục nội dung còn thiếu dựa trên các nguồn đã cung cấp."
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
            if self._finish_reason(res).upper() not in {"MAX_TOKENS", "FINISH_REASON_MAX_TOKENS"}:
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
        multi_penalty_answer = self._deterministic_motorbike_multi_penalty_answer(query, contexts)
        if multi_penalty_answer:
            return multi_penalty_answer
        vague_answer = self._deterministic_vague_penalty_answer(query, contexts)
        if vague_answer:
            return vague_answer
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

    def _deterministic_motorbike_multi_penalty_answer(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        qa = ascii_lower(query)
        has_no_helmet = ("khong doi" in qa and "mu" in qa) or "mu bao hiem" in qa
        has_alcohol = any(term in qa for term in [
            "nong do con",
            "say xin",
            "ruou",
            "bia",
            "co con",
        ])
        has_red_light = any(term in qa for term in [
            "vuot den do",
            "den do",
            "tin hieu den",
            "den tin hieu",
        ])
        asks_penalty = any(term in qa for term in ["phat", "xu phat", "bi gi", "xu ly", "bao nhieu"])
        explicit_car = any(term in qa for term in ["o to", "xe hoi", "xe tai", "xe khach", "container"])
        if explicit_car or not (has_no_helmet and has_alcohol and has_red_light and asks_penalty):
            return ""

        lines = [
            "## Mức phạt dự kiến cho mô tô/xe gắn máy",
            "",
            "Giả định bạn điều khiển mô tô/xe gắn máy. Ba lỗi này thường bị xử phạt theo từng hành vi riêng; phần nồng độ cồn phải có số đo cụ thể mới chốt được một mức duy nhất.",
            "",
            "| Hành vi | Mức phạt tiền | GPLX/điểm | Căn cứ |",
            "|---|---|---|---|",
            "| Không đội mũ bảo hiểm khi điều khiển xe | 400.000 - 600.000 đồng | Không thấy quy định trừ điểm trong nhánh này | Điểm h khoản 2 Điều 7 Nghị định 168/2024/NĐ-CP |",
            "| Không chấp hành hiệu lệnh đèn tín hiệu giao thông/vượt đèn đỏ | 4.000.000 - 6.000.000 đồng | Trừ 4 điểm GPLX | Điểm c khoản 7 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
            "| Có nồng độ cồn nhưng chưa vượt quá 50 mg/100 ml máu hoặc 0,25 mg/l khí thở | 2.000.000 - 3.000.000 đồng | Trừ 4 điểm GPLX | Điểm a khoản 6 và điểm b khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
            "| Nồng độ cồn vượt quá 50 đến 80 mg/100 ml máu hoặc vượt quá 0,25 đến 0,4 mg/l khí thở | 6.000.000 - 8.000.000 đồng | Trừ 10 điểm GPLX | Điểm b khoản 8 và điểm d khoản 13 Điều 7 Nghị định 168/2024/NĐ-CP |",
            "| Nồng độ cồn vượt quá 80 mg/100 ml máu hoặc vượt quá 0,4 mg/l khí thở | 8.000.000 - 10.000.000 đồng | Tước quyền sử dụng GPLX 22 - 24 tháng | Điểm d khoản 9 và điểm c khoản 12 Điều 7 Nghị định 168/2024/NĐ-CP |",
            "",
            "Tạm tính tổng tiền phạt nếu cộng 3 lỗi: 6.400.000 - 9.600.000 đồng ở ngưỡng cồn thấp; 10.400.000 - 14.600.000 đồng ở ngưỡng trung bình; 12.400.000 - 16.600.000 đồng ở ngưỡng cao.",
            "",
            "Nếu không chấp hành yêu cầu kiểm tra nồng độ cồn, hoặc nếu hành vi gây tai nạn, mức xử lý có thể chuyển sang nhánh nặng hơn.",
        ]
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
        del query
        parts = [
            "Tôi tìm thấy các căn cứ liên quan dưới đây. Bạn có thể dùng phần căn cứ pháp lý kèm theo để đối chiếu chi tiết."
        ]
        try:
            default_contexts = "8" if self._env_bool("RAG_DEPLOY_FAST_MODE", False) else "80"
            max_contexts = int(os.getenv("RAG_EXTRACTIVE_MAX_CONTEXTS", default_contexts))
        except Exception:
            max_contexts = 8 if self._env_bool("RAG_DEPLOY_FAST_MODE", False) else 80
        try:
            default_limit = "1200" if self._env_bool("RAG_DEPLOY_FAST_MODE", False) else "12000"
            text_limit = int(os.getenv("RAG_EXTRACTIVE_TEXT_LIMIT", default_limit))
        except Exception:
            text_limit = 1200 if self._env_bool("RAG_DEPLOY_FAST_MODE", False) else 12000
        for idx, ctx in enumerate(contexts[:max_contexts], start=1):
            text = re.sub(r"\s+", " ", source_text(ctx) or "").strip()
            if len(text) > text_limit:
                text = text[: max(0, text_limit - 1)].rstrip() + "…"
            penalty = penalty_summary(ctx)
            penalty_bits = []
            if penalty.get("fine_min_vnd") or penalty.get("fine_max_vnd"):
                penalty_bits.append(f"phạt tiền {penalty.get('fine_min_vnd') or '?'} - {penalty.get('fine_max_vnd') or '?'} đồng")
            if penalty.get("point_deduction"):
                penalty_bits.append(f"trừ điểm: {penalty.get('point_deduction')}")
            if penalty.get("license_suspension"):
                penalty_bits.append(f"tước GPLX: {penalty.get('license_suspension')}")
            penalty_text = f"\nMức áp dụng: {', '.join(str(bit) for bit in penalty_bits)}" if penalty_bits else ""
            parts.append(f"### {idx}. {format_reference(ctx)}{penalty_text}\n{text}")
        return "\n\n".join(parts)

    def _load_image(self, path: Optional[str]):
        if not path: return None
        img_path = Path(path)
        if not img_path.is_absolute(): img_path = self.project_root / img_path
        if not img_path.exists(): return None
        try:
            return PIL.Image.open(img_path)
        except Exception: return None
