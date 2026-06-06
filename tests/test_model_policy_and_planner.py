import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from src.rag.hybrid_vector_store import _should_load_embedder
from src.rag.adaptive_query import AdaptiveQuestionAnalyzer
from src.rag.conversation_guard import route_conversational_query
from src.rag.custom_legal_retriever import CustomLegalRetriever
from src.data_pipeline.qa_generator import deterministic_fallback_qa, question_practicality_issue
from src.rag.legal_graph_rag import LegalGraphRAG
from src.rag.legal_utils import looks_like_table_query, public_asset_path
from src.rag.model_policy import generate_content_with_fallback, model_candidates
from src.rag.query_preprocessor import missing_data_hints, prepare_chat_query
from src.rag.query_planner import LegalQueryPlanner, QueryPlan
from src.rag.sequential_retrieval import SequentialRetrievalOrchestrator
from frontend.asset_utils import image_source
from frontend.render_utils import split_markdown_sections, vision_display_rows
from api.main import _image_query_needs_penalty, _image_sign_fast_answer


@contextmanager
def patched_env(**updates):
    old_values = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class FakeModels:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate_content(self, **kwargs):
        model = kwargs["model"]
        self.calls.append(model)
        response = self.responses.get(model)
        if isinstance(response, list):
            response = response.pop(0) if response else ""
        if isinstance(response, Exception):
            raise response
        if isinstance(response, tuple):
            text, finish_reason = response
            return SimpleNamespace(
                text=text,
                candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))],
            )
        return SimpleNamespace(text=response)


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class ModelPolicyTest(unittest.TestCase):
    def test_planner_task_prefers_fast_models_and_filters_unknown_env_values(self):
        with patched_env(RAG_PLANNER_MODEL="unknown-model,gemini-2.5-flash-lite"):
            candidates = model_candidates("RAG_PLANNER_MODEL", task="planner")

        self.assertEqual(candidates[0], "gemini-2.5-flash-lite")
        self.assertIn("gemini-3.1-flash-lite", candidates)
        self.assertNotIn("unknown-model", candidates)

    def test_vision_candidates_exclude_non_vision_models(self):
        with patched_env(RAG_VISION_MODEL="gemma-4-31b-it,gemini-2.5-flash-lite"):
            candidates = model_candidates("RAG_VISION_MODEL", vision=True, task="vision")

        self.assertEqual(candidates[0], "gemini-2.5-flash-lite")
        self.assertNotIn("gemma-4-31b-it", candidates)

    def test_generate_content_falls_back_when_first_candidate_returns_empty_text(self):
        client = FakeClient({
            "gemini-3.1-flash-lite": "",
            "gemini-2.5-flash-lite": "ok",
        })

        with patched_env(RAG_PLANNER_MODEL=None):
            response, model = generate_content_with_fallback(
                client,
                contents=["plan this"],
                task="planner",
                env_names=("RAG_PLANNER_MODEL",),
            )

        self.assertEqual(response.text, "ok")
        self.assertEqual(model, "gemini-2.5-flash-lite")
        self.assertEqual(client.models.calls[:2], ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite"])


class QueryPlannerCoverageTest(unittest.TestCase):
    def test_conversation_guard_handles_greeting_without_retrieval(self):
        routed = route_conversational_query("Hello")

        self.assertIsNotNone(routed)
        self.assertEqual(routed.intent, "greeting")
        self.assertIn("pháp luật giao thông đường bộ", routed.answer)
        self.assertTrue(routed.metadata()["retrieval_skipped"])

    def test_conversation_guard_introduces_capabilities_and_sources(self):
        routed = route_conversational_query("Bạn có thể giúp gì cho tôi và truy xuất từ đâu?")

        self.assertIsNotNone(routed)
        self.assertEqual(routed.intent, "capability_intro")
        self.assertIn("Nghị định 168/2024/NĐ-CP", routed.answer)
        self.assertIn("QCVN 41:2024", routed.answer)
        self.assertIn("mức phạt", routed.answer)

    def test_conversation_guard_rejects_clear_off_topic_questions(self):
        routed = route_conversational_query("Hôm nay thời tiết ở Hà Nội thế nào?")

        self.assertIsNotNone(routed)
        self.assertEqual(routed.intent, "out_of_scope")
        self.assertIn("chỉ hỗ trợ", routed.answer)
        self.assertIn("giao thông đường bộ", routed.answer)

    def test_conversation_guard_does_not_block_traffic_law_questions(self):
        self.assertIsNone(route_conversational_query("Xe máy vượt đèn đỏ bị phạt bao nhiêu?"))
        self.assertIsNone(route_conversational_query("Hello, xe máy vượt đèn đỏ bị phạt bao nhiêu?"))
        self.assertIsNone(route_conversational_query("Tôi bị phạt bao nhiêu?"))

    def test_colloquial_parking_sidewalk_question_routes_to_penalty(self):
        query = "Hey tôi đỗ xe tải ở vỉa hè có ổn không?"
        planner = LegalQueryPlanner()
        plan = planner.rule_plan(query)
        profile = AdaptiveQuestionAnalyzer().analyze(query, plan)
        facets = [slot["facet"] for slot in plan.subquestions]
        profile_queries = "\n".join(slot["query"] for slot in profile.evidence_slots)

        self.assertIsNone(route_conversational_query(query))
        self.assertEqual(plan.intent.value, "penalty")
        self.assertIn("penalty", facets)
        self.assertIn("rule", facets)
        self.assertIn("vỉa hè", profile_queries)
        self.assertIn("Nghị định 168/2024/NĐ-CP", profile_queries)

    def test_llm_query_understanding_converts_colloquial_legality_to_retrieval_slots(self):
        payload = json.dumps({
            "in_scope": True,
            "intent": "penalty",
            "confidence": 0.86,
            "difficulty_hint": "easy",
            "user_tone": "colloquial",
            "facets": ["rule", "penalty"],
            "entities": {
                "vehicle": "xe tải",
                "action": "để xe trên vỉa hè",
                "location": "vỉa hè",
                "sign_codes": [],
                "asks_legality": True,
                "asks_penalty": True,
                "missing_facts": [],
            },
            "retrieval_queries": [
                {
                    "facet": "rule",
                    "query": "quy định xe tải dừng đỗ trên vỉa hè, hè phố hoặc lòng đường",
                    "priority": 1,
                    "reason": "Người dùng hỏi có được/có sao không.",
                    "must_answer": True,
                },
                {
                    "facet": "penalty",
                    "query": "mức phạt xe tải dừng đỗ trên vỉa hè theo Nghị định 168/2024/NĐ-CP",
                    "priority": 2,
                    "reason": "Cần truy xuất chế tài tương ứng.",
                    "must_answer": True,
                },
            ],
            "notes": ["Câu có lời chào nhưng vẫn là truy vấn pháp luật giao thông."],
        }, ensure_ascii=False)
        client = FakeClient({"gemini-3.1-flash-lite": payload})

        with patched_env(RAG_ENABLE_LLM_QUERY_UNDERSTANDING="true", RAG_ENABLE_AI_PLANNER="false"):
            plan = LegalQueryPlanner().plan("Alo tôi để xe tải lên vỉa hè có sao không?", client=client)

        self.assertEqual(plan.plan_source, "llm_understanding")
        self.assertEqual(plan.intent.value, "penalty")
        self.assertEqual(client.models.calls[0], "gemini-3.1-flash-lite")
        queries = "\n".join(slot["query"] for slot in plan.subquestions)
        self.assertIn("vỉa hè", queries)
        self.assertIn("Nghị định 168/2024/NĐ-CP", queries)
        self.assertIn("llm_understanding_entities", plan.filters)

    def test_llm_query_understanding_invalid_payload_keeps_rule_fallback(self):
        client = FakeClient({"gemini-3.1-flash-lite": "not json"})

        with patched_env(RAG_ENABLE_LLM_QUERY_UNDERSTANDING="true", RAG_ENABLE_AI_PLANNER="true"):
            plan = LegalQueryPlanner().plan("Alo tôi để xe tải lên vỉa hè có sao không?", client=client)

        self.assertEqual(plan.plan_source, "rule")

    def test_generated_question_practicality_filter_rejects_lan_man_prompts(self):
        self.assertEqual(question_practicality_issue("Top 10 hành vi hay vi phạm nhất là gì?"), "broad_or_statistical_question")
        self.assertEqual(question_practicality_issue("Quy định này nói gì theo cách dễ hiểu?"), "abstract_heading_like_question")
        self.assertEqual(question_practicality_issue("Xe máy vượt đèn đỏ bị phạt bao nhiêu?"), "")

    def test_deterministic_qa_fallback_uses_real_world_question(self):
        qa_pairs = deterministic_fallback_qa({
            "source_chunk_id": "test_chunk",
            "doc_name": "Nghị định 168/2024/NĐ-CP",
            "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "4"},
            "source_body_exact": (
                "Người điều khiển xe không chấp hành hiệu lệnh của đèn tín hiệu giao thông "
                "bị phạt tiền và bị trừ điểm giấy phép lái xe."
            ),
        })

        self.assertTrue(qa_pairs)
        question = qa_pairs[0]["question"]
        self.assertIn("đèn", question.lower())
        self.assertNotIn("Quy định này", question)
        self.assertEqual(question_practicality_issue(question), "")

    def test_query_preprocessor_compacts_long_query_without_llm(self):
        long_background = " ".join(["thong tin ngoai le khong lien quan"] * 240)
        query = (
            f"{long_background}. Tinh huong chinh: xe may vuot den do tai nga tu, "
            "sau do khong doi mu bao hiem va hoi muc phat, tru diem GPLX."
        )

        with patched_env(RAG_PREPARED_QUERY_MAX_CHARS="900", RAG_LONG_QUERY_WORDS="120"):
            prepared = prepare_chat_query(None, query, [])

        self.assertTrue(prepared.was_preprocessed)
        self.assertLessEqual(len(prepared.effective_query), 900)
        self.assertIn("xe may", prepared.effective_query)
        self.assertIn("vuot den do", prepared.effective_query)
        self.assertIn("khong doi mu bao hiem", prepared.effective_query)

    def test_query_preprocessor_uses_llm_json_for_history_condense(self):
        json_payload = (
            '{"standalone_query":"Xe máy trong tình huống trước vượt đèn đỏ; hỏi mức phạt tiền, trừ điểm GPLX và căn cứ.",'
            '"history_summary":"Tình huống trước là xe máy ở ngã tư.",'
            '"missing_data_hints":["Thiếu hậu quả tai nạn nếu có."],'
            '"warnings":[]}'
        )
        client = FakeClient({"gemini-3.1-flash-lite": json_payload})
        history = [
            {"role": "user", "content": "Tôi đi xe máy qua ngã tư khi đèn đỏ."},
            {"role": "assistant", "content": "Cần tra cứu tín hiệu đèn và mức phạt tương ứng."},
        ]

        with patched_env(RAG_ENABLE_QUERY_PREPROCESSOR_LLM="true"):
            prepared = prepare_chat_query(client, "Vậy bị phạt bao nhiêu?", history)

        self.assertTrue(prepared.used_llm)
        self.assertIn("Xe máy", prepared.effective_query)
        self.assertIn("vượt đèn đỏ", prepared.effective_query)
        self.assertIn("Thiếu hậu quả tai nạn nếu có.", prepared.missing_data_hints)
        self.assertEqual(client.models.calls[0], "gemini-3.1-flash-lite")
        self.assertTrue(json_payload)

    def test_missing_data_hints_cover_ambiguous_speed_penalty(self):
        hints = missing_data_hints("Chạy xe quá tốc độ bị phạt sao?")
        joined = "\n".join(hints)

        self.assertIn("Loại phương tiện", joined)
        self.assertIn("ngưỡng tốc độ", joined)

    def test_rule_planner_handles_interwoven_table_image_sign_penalty_scenario(self):
        query = (
            "Tinh huong: toi thay bien P.127 trong phu luc/bang QCVN, "
            "can anh trang goc va muc phat neu xe may chay qua toc do "
            "dong thoi khong doi mu bao hiem."
        )

        plan = LegalQueryPlanner().rule_plan(query)
        facets = [slot["facet"] for slot in plan.subquestions]

        self.assertIn("P127", plan.sign_codes)
        self.assertIn("scenario", facets)
        self.assertIn("sign", facets)
        self.assertIn("table", facets)
        self.assertIn("penalty", facets)
        self.assertIn("source_image", facets)
        self.assertIn("sign", plan.expected_modalities)
        self.assertIn("table", plan.expected_modalities)
        self.assertIn("image", plan.expected_modalities)

    def test_adaptive_analyzer_expands_ambiguous_penalty_by_vehicle_group(self):
        analyzer = AdaptiveQuestionAnalyzer()
        plan = SimpleNamespace(
            subquestions=[
                {
                    "facet": "penalty",
                    "query": "Tra cứu mức phạt cho câu hỏi: Chạy xe quá tốc độ bị phạt sao?",
                    "priority": 1,
                    "reason": "Planner AI chỉ tạo một nhánh chung.",
                    "must_answer": True,
                }
            ],
            intent=SimpleNamespace(value="penalty"),
            plan_source="ai",
            difficulty_hint="medium",
        )

        profile = analyzer.analyze("Chạy xe quá tốc độ bị phạt sao?", plan)
        queries = "\n".join(slot["query"] for slot in profile.evidence_slots)

        self.assertIn("ô tô", queries)
        self.assertIn("mô tô", queries)
        self.assertIn("xe máy chuyên dùng", queries)
        self.assertEqual(profile.difficulty, "hard")
        self.assertGreaterEqual(len(profile.evidence_slots), 4)

    def test_adaptive_analyzer_keeps_single_vehicle_single_penalty_easy(self):
        analyzer = AdaptiveQuestionAnalyzer()
        planner = LegalQueryPlanner()
        query = "Xe máy vượt đèn đỏ bị phạt bao nhiêu?"

        profile = analyzer.analyze(query, planner.rule_plan(query))

        self.assertEqual(profile.difficulty, "easy")
        self.assertLessEqual(profile.difficulty_score, 2)
        self.assertIn("Câu hỏi xử phạt một hành vi", profile.difficulty_reason)

    def test_conditional_reranker_only_triggers_for_hard_queries(self):
        retriever = CustomLegalRetriever.__new__(CustomLegalRetriever)
        easy_plan = QueryPlan(filters={
            "_adaptive_difficulty": "easy",
            "_adaptive_difficulty_score": 2,
            "_adaptive_facets": ["penalty"],
        })
        hard_plan = QueryPlan(filters={
            "_adaptive_difficulty": "hard",
            "_adaptive_difficulty_score": 7,
            "_adaptive_facets": ["scenario", "penalty"],
        })

        with patched_env(RAG_ENABLE_RERANKER="false", RAG_ENABLE_RERANKER_FOR_HARD="true"):
            self.assertFalse(retriever._should_model_rerank("Xe máy vượt đèn đỏ bị phạt bao nhiêu?", [{}], easy_plan))
            self.assertTrue(retriever._should_model_rerank("Tình huống nhiều lỗi cần tổng hợp mức phạt", [{}], hard_plan))

    def test_general_legal_questions_use_structured_routes_without_false_facets(self):
        planner = LegalQueryPlanner()
        analyzer = AdaptiveQuestionAnalyzer()
        expected = {
            "336-2025 nghị định chính phủ có bao nhiêu điều luật?": "document_overview",
            "Một người thì có số điểm lái xe là bao nhiêu?": "definition",
            "vi phạm hành vi nào thì bị mức phạt cao nhất?": "aggregation",
            "Các hành vi vi phạm nào sẽ bị tước bằng lái xe": "aggregation",
            "điều 40 trong nghị định 168 nói về cái gì": "legal_detail",
        }

        for query, expected_intent in expected.items():
            with self.subTest(query=query):
                plan = planner.rule_plan(query)
                profile = analyzer.analyze(query, plan)
                self.assertEqual(plan.intent.value, expected_intent)
                self.assertEqual(profile.intent, expected_intent)
                self.assertNotIn("source_image", profile.facets)
                self.assertNotIn("table", profile.facets)

        overview = analyzer.analyze(
            "336-2025 nghị định chính phủ có bao nhiêu điều luật?",
            planner.rule_plan("336-2025 nghị định chính phủ có bao nhiêu điều luật?"),
        )
        self.assertNotIn("penalty", overview.facets)

    def test_license_word_bang_does_not_trigger_table_route(self):
        planner = LegalQueryPlanner()
        analyzer = AdaptiveQuestionAnalyzer()
        queries = [
            "Tôi có bằng A1 chạy xe hơi thì có ok không?",
            "Tôi có bằng lái xe A2 nhưng muốn chạy xe con có được không?",
            "Chủ xe giao ô tô cho người chỉ có bằng A1 thì bị xử lý sao?",
        ]

        for query in queries:
            with self.subTest(query=query):
                plan = planner.rule_plan(query)
                profile = analyzer.analyze(query, plan)

                self.assertFalse(looks_like_table_query(query))
                self.assertNotEqual(plan.intent.value, "table")
                self.assertNotIn("table", profile.facets)

    def test_sanction_catalog_does_not_request_vehicle_clarification(self):
        hints = missing_data_hints("Các hành vi vi phạm nào sẽ bị tước bằng lái xe?")
        self.assertFalse(any("Loại phương tiện" in hint for hint in hints))

    def test_statutory_fine_cap_is_definition_not_aggregation(self):
        query = (
            "Theo Nghị định 336/2025/NĐ-CP, mức phạt tiền tối đa trong hoạt động đường bộ "
            "đối với cá nhân và tổ chức được quy định là bao nhiêu?"
        )
        planner = LegalQueryPlanner()
        plan = planner.rule_plan(query)
        profile = AdaptiveQuestionAnalyzer().analyze(query, plan)

        self.assertEqual(plan.intent.value, "definition")
        self.assertEqual(profile.intent, "definition")
        self.assertIn("definition", profile.facets)
        self.assertNotIn("aggregation", profile.facets)
        self.assertNotIn("penalty", profile.facets)
        self.assertEqual(len(profile.evidence_slots), 1)
        self.assertEqual(profile.difficulty, "easy")
        self.assertLessEqual(profile.difficulty_score, 2)

    def test_aggregation_profile_skips_penalty_followups_and_answer_repair(self):
        orchestrator = SequentialRetrievalOrchestrator(None, lambda *_args, **_kwargs: "")
        profile = SimpleNamespace(facets=["aggregation"])
        query = "Các hành vi vi phạm nào sẽ bị tước bằng lái xe?"

        followups = orchestrator._coverage_followup_slots(
            query=query,
            profile=profile,
            plan=QueryPlan(),
            results=[],
            records=[],
            existing_slots=[],
            round_idx=0,
        )
        repairs = orchestrator._answer_repair_slots(
            answer="Danh mục hành vi bị tước GPLX.",
            query=query,
            profile=profile,
            plan=QueryPlan(),
            results=[],
            records=[],
            existing_slots=[],
            repair_idx=0,
        )

        self.assertEqual(followups, [])
        self.assertEqual(repairs, [])
        self.assertFalse(
            orchestrator._answer_has_unresolved_ambiguity(
                "Danh mục hành vi bị tước GPLX.",
                query,
                profile,
            )
        )


class StructuredGeneralRetrievalTest(unittest.TestCase):
    def test_direct_retrieval_honors_structured_facets(self):
        class RouteSpy:
            def __getattr__(self, name):
                if name.startswith("retrieve_"):
                    return lambda *_args, **_kwargs: [{"route": name}]
                raise AttributeError(name)

        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        rag.retriever = RouteSpy()
        plan = QueryPlan()

        for facet, route in [
            ("document_overview", "retrieve_document_overview"),
            ("legal_detail", "retrieve_legal_detail"),
            ("aggregation", "retrieve_aggregation"),
        ]:
            with self.subTest(facet=facet):
                profile = SimpleNamespace(
                    retrieval_budget={"top_k": 10, "expand_depth": 1},
                    facets=[facet],
                )
                records = rag._retrieve_direct("query", plan, profile)
                self.assertEqual(records[0]["route"], route)

    def test_document_overview_counts_only_direct_article_headings(self):
        retriever = CustomLegalRetriever.__new__(CustomLegalRetriever)
        retriever.vector_store = SimpleNamespace(records=[
            {
                "source_chunk_id": "d1",
                "doc_name": "Nghị định 336/2025/NĐ-CP",
                "legal_reference": {"document": "Nghị định 336/2025/NĐ-CP", "article": "1"},
                "source_body_exact": "Điều 1. Phạm vi điều chỉnh",
            },
            {
                "source_chunk_id": "d2",
                "doc_name": "Nghị định 336/2025/NĐ-CP",
                "legal_reference": {"document": "Nghị định 336/2025/NĐ-CP", "article": "2"},
                "source_body_exact": "# Điều 2. Đối tượng áp dụng",
            },
            {
                "source_chunk_id": "legacy_80",
                "doc_name": "Nghị định 336/2025/NĐ-CP",
                "legal_reference": {"document": "Nghị định 336/2025/NĐ-CP", "article": "80"},
                "source_body_exact": "Bãi bỏ điểm i khoản 1 Điều 80 của văn bản khác.",
            },
        ])

        rows = retriever._document_article_rows(["Nghị định 336/2025/NĐ-CP"])
        self.assertEqual([row["article"] for row in rows], ["1", "2"])
        self.assertIn("**2 tiêu đề điều", retriever._format_document_overview("Nghị định 336/2025/NĐ-CP", rows))

    def test_license_suspension_catalog_aggregates_all_matching_actions(self):
        retriever = CustomLegalRetriever.__new__(CustomLegalRetriever)
        records = [
            {
                "source_chunk_id": "s1",
                "doc_name": "Nghị định 168/2024/NĐ-CP",
                "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "6", "clause": "7", "point": "a"},
                "violation_content": "Điều khiển xe chạy quá tốc độ quy định trên 35 km/h",
                "qa_context": "Hành vi điều khiển xe quá tốc độ trên 35 km/h bị tước GPLX từ 2-4 tháng.",
                "source_body_exact": "Điều khiển xe chạy quá tốc độ quy định trên 35 km/h bị tước GPLX từ 2-4 tháng.",
            },
            {
                "source_chunk_id": "s2",
                "doc_name": "Nghị định 168/2024/NĐ-CP",
                "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "9", "point": "d"},
                "violation_content": "Điều khiển xe khi có nồng độ cồn vượt ngưỡng cao nhất",
                "qa_context": "Hành vi này bị tước quyền sử dụng giấy phép lái xe từ 22 đến 24 tháng.",
                "source_body_exact": "Điều khiển xe khi có nồng độ cồn vượt ngưỡng cao nhất bị tước quyền sử dụng giấy phép lái xe từ 22 đến 24 tháng.",
            },
        ]

        result = retriever._license_suspension_aggregation_records(
            "Các hành vi nào bị tước GPLX?",
            records,
            top_k=8,
        )
        summary = result[0]["source_body_exact"]
        self.assertEqual(result[0]["rag_modality"], "aggregation")
        self.assertIn("quá tốc độ", summary)
        self.assertIn("nồng độ cồn", summary)
        self.assertIn("22 đến 24 tháng", summary)
        self.assertEqual(len(result), 3)

    def test_point_aggregation_returns_highest_deduction_table(self):
        retriever = CustomLegalRetriever.__new__(CustomLegalRetriever)
        records = [
            {
                "source_chunk_id": "p1",
                "doc_name": "Nghị định 168/2024/NĐ-CP",
                "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "8", "point": "d"},
                "source_body_exact": "Đây là một hành vi vi phạm có hình thức xử phạt: trừ 10 điểm GPLX.",
                "qa_context": "Hành vi vi phạm bị trừ 10 điểm GPLX.",
                "penalties": {"point_deduction": 10},
            },
            {
                "source_chunk_id": "p2",
                "doc_name": "Nghị định 168/2024/NĐ-CP",
                "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "6", "clause": "7", "point": "b"},
                "source_body_exact": "Hành vi vi phạm khác bị trừ 4 điểm GPLX.",
                "qa_context": "Hành vi vi phạm bị trừ 4 điểm GPLX.",
                "penalties": {"point_deduction": 4},
            },
        ]

        result = retriever._point_aggregation_records("Hành vi nào bị trừ điểm GPLX cao nhất?", records, top_k=5)
        summary = result[0]["source_body_exact"]
        self.assertEqual(result[0]["rag_modality"], "aggregation")
        self.assertIn("## Thống kê mức trừ điểm cao nhất", summary)
        self.assertIn("10", summary)
        self.assertIn("4", summary)
        self.assertEqual(len(result), 3)

    def test_license_point_total_has_exact_article_anchor(self):
        retriever = CustomLegalRetriever.__new__(CustomLegalRetriever)
        retriever.vector_store = SimpleNamespace(records=[{
            "source_chunk_id": "points",
            "doc_name": "Luật Trật tự ATGT 2024 (Tiếp)",
            "legal_reference": {
                "document": "Luật Trật tự ATGT 2024 (Tiếp)",
                "article": "58",
                "clause": "1",
            },
            "source_body_exact": "Điểm của giấy phép lái xe bao gồm 12 điểm.",
        }])

        records = retriever._topic_anchor_matches("Một người thì có số điểm lái xe là bao nhiêu?")
        self.assertEqual(records[0]["source_chunk_id"], "points")
        self.assertIn("topic_license_points_total", records[0]["retrieval_reasons"])

    def test_license_vehicle_mismatch_detector_is_scoped(self):
        retriever = CustomLegalRetriever.__new__(CustomLegalRetriever)

        self.assertTrue(retriever._looks_like_license_vehicle_mismatch_query("Tôi có bằng A1 chạy xe hơi có được không?"))
        self.assertTrue(retriever._looks_like_license_vehicle_mismatch_query("GPLX A2 lái ô tô thì sao?"))
        self.assertFalse(retriever._looks_like_license_vehicle_mismatch_query("Tôi có bằng A1 chạy xe máy có được không?"))
        self.assertFalse(retriever._looks_like_license_vehicle_mismatch_query("Tra bảng thông số xe ô tô trong phụ lục."))


class HybridVectorStoreConfigTest(unittest.TestCase):
    def test_disabled_embeddings_skip_existing_local_model_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_model_path = Path(tmp_dir)

            self.assertFalse(
                _should_load_embedder(
                    enable_embeddings=False,
                    local_model_path=local_model_path,
                    force_reindex=False,
                    allow_model_download=False,
                )
            )
            self.assertTrue(
                _should_load_embedder(
                    enable_embeddings=True,
                    local_model_path=local_model_path,
                    force_reindex=False,
                    allow_model_download=False,
                )
            )


class AssetPathTest(unittest.TestCase):
    def test_public_asset_path_is_idempotent_for_processed_assets(self):
        self.assertEqual(
            public_asset_path("data/processed/images/doc/page_0.png"),
            "/processed/images/doc/page_0.png",
        )
        self.assertEqual(
            public_asset_path("/processed/images/doc/page_0.png"),
            "/processed/images/doc/page_0.png",
        )
        self.assertEqual(
            public_asset_path("processed/images/doc/page_0.png"),
            "/processed/images/doc/page_0.png",
        )
        self.assertEqual(
            public_asset_path("/app/data/processed/sign_assets/P_127.png"),
            "/processed/sign_assets/P_127.png",
        )
        self.assertEqual(
            public_asset_path("https://example.test/asset.png"),
            "https://example.test/asset.png",
        )

    def test_image_source_prefers_local_processed_file_for_streamlit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            processed_dir = Path(tmp_dir) / "processed"
            image_path = processed_dir / "images" / "doc" / "page_0.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"png")

            self.assertEqual(
                image_source(
                    "/processed/images/doc/page_0.png",
                    api_url="http://127.0.0.1:8002",
                    processed_dir=processed_dir,
                ),
                str(image_path.resolve()),
            )
            self.assertEqual(
                image_source(
                    "data/processed/images/doc/missing.png",
                    api_url="http://127.0.0.1:8002",
                    processed_dir=processed_dir,
                ),
                "http://127.0.0.1:8002/processed/images/doc/missing.png",
            )


class RenderUtilsTest(unittest.TestCase):
    def test_split_markdown_sections_preserves_section_order(self):
        answer = (
            "## Trả lời ngắn gọn\n"
            "Có.\n\n"
            "## Phân tích từng nhánh\n"
            "- Nhánh 1\n\n"
            "## Căn cứ áp dụng\n"
            "| a | b |\n"
        )

        sections = split_markdown_sections(answer)

        self.assertEqual([item["title"] for item in sections], [
            "Trả lời ngắn gọn",
            "Phân tích từng nhánh",
            "Căn cứ áp dụng",
        ])
        self.assertIn("Có.", sections[0]["body"])
        self.assertIn("- Nhánh 1", sections[1]["body"])
        self.assertIn("| a | b |", sections[2]["body"])

    def test_vision_display_rows_includes_confidence_and_codes(self):
        rows = vision_display_rows({
            "is_traffic_sign": True,
            "trusted_codes": ["P.102"],
            "confidence": 0.87,
            "sign_group": "Biển cấm",
            "dominant_colors": ["đỏ", "trắng"],
        })

        labels = [label for label, _value in rows]
        self.assertIn("Nhận diện", labels)
        self.assertIn("Mã tin cậy", labels)
        self.assertIn("Độ tin cậy", labels)
        self.assertIn("Nhóm biển", labels)


class ImageSignAnswerTest(unittest.TestCase):
    def test_image_query_penalty_detector(self):
        self.assertTrue(_image_query_needs_penalty("Biển này bị phạt thế nào?"))
        self.assertFalse(_image_query_needs_penalty("Biển này có nghĩa gì?"))

    def test_image_fast_answer_uses_sign_structure(self):
        answer = _image_sign_fast_answer(
            vision={
                "is_traffic_sign": True,
                "confidence": 0.92,
                "sign_group": "Biển cấm",
                "symbol": "vạch trắng ngang",
                "text": "",
                "alternatives": [{"code": "P.101", "reason": "gần giống biển cấm"}],
            },
            trusted_codes=["P.102"],
            query="Biển này nghĩa là gì?",
            docs=[
                {
                    "rag_modality": "sign",
                    "figure": {"code": "P.102", "name": "Cấm đi ngược chiều", "caption": "Biển tròn nền đỏ, có vạch trắng ngang"},
                    "legal_reference": {"document": "QCVN 41:2024 (Thông tư 51/2024)", "article": "", "clause": ""},
                    "source_body_exact": "Biển số P.102: Cấm đi ngược chiều",
                }
            ],
        )

        self.assertIn("## Trả lời ngắn gọn", answer)
        self.assertIn("## Nhận diện ảnh", answer)
        self.assertIn("## Phân tích ảnh", answer)
        self.assertIn("P.102", answer)
        self.assertIn("Cấm đi ngược chiều", answer)


class DeterministicPenaltyAnswerTest(unittest.TestCase):
    def test_statutory_fine_cap_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Theo Nghị định 336/2025/NĐ-CP, mức phạt tiền tối đa đối với cá nhân và tổ chức là bao nhiêu?",
            [{
                "doc_name": "Nghị định 336/2025/NĐ-CP",
                "legal_reference": {
                    "document": "Nghị định 336/2025/NĐ-CP",
                    "article": "3",
                    "clause": "1",
                },
                "source_body_exact": (
                    "Mức phạt tiền tối đa đối với cá nhân là 75.000.000 đồng "
                    "và đối với tổ chức là 150.000.000 đồng."
                ),
            }],
        )

        self.assertIn("**75.000.000 đồng**", answer)
        self.assertIn("**150.000.000 đồng**", answer)
        self.assertIn("Điều 3", answer)

    def test_license_point_total_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Một người thì có số điểm lái xe là bao nhiêu?",
            [{
                "doc_name": "Luật Trật tự ATGT 2024 (Tiếp)",
                "legal_reference": {
                    "document": "Luật Trật tự ATGT 2024 (Tiếp)",
                    "article": "58",
                    "clause": "1",
                },
                "source_body_exact": "Điểm của giấy phép lái xe bao gồm 12 điểm.",
            }],
        )

        self.assertIn("**12 điểm**", answer)
        self.assertIn("Khoản 1", answer)
        self.assertIn("Điều 58", answer)

    def test_owner_gives_car_to_a1_driver_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Chủ xe giao ô tô cho người chỉ có bằng A1 lái thì chủ xe có bị phạt không?",
            [],
        )

        self.assertIn("Có. Nếu chủ xe giao ô tô", answer)
        self.assertIn("## Phân tích từng nhánh", answer)
        self.assertIn("## Tổng hậu quả", answer)
        self.assertIn("Điều 32", answer)
        self.assertIn("Điều 18", answer)

    def test_answer_continuation_strips_completion_marker(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        rag.client = FakeClient({
            "gemini-3.1-flash-lite": [
                (" phần còn lại rõ ràng.\n<<<HOAN_TAT_TRA_LOI>>>", "STOP"),
            ]
        })
        first_response = SimpleNamespace(
            text="Trả lời đang bị cắt giữa",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
        )

        with patched_env(RAG_ANSWER_MAX_CONTINUATIONS="2", RAG_REQUIRE_ANSWER_COMPLETION_MARKER="true"):
            answer = rag._continue_if_truncated(
                model="gemini-3.1-flash-lite",
                base_contents=["system", "context"],
                answer="Trả lời đang bị cắt giữa",
                first_response=first_response,
                max_output_tokens=4096,
            )

        self.assertIn("Trả lời đang bị cắt giữa", answer)
        self.assertIn("phần còn lại rõ ràng", answer)
        self.assertNotIn("<<<HOAN_TAT_TRA_LOI>>>", rag._strip_completion_marker(answer))

    def test_completion_marker_prevents_unneeded_continuation(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        rag.client = FakeClient({"gemini-3.1-flash-lite": [("không nên gọi", "STOP")]})
        response = SimpleNamespace(
            text="Đã trả lời đầy đủ.\n<<<HOAN_TAT_TRA_LOI>>>",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )

        with patched_env(RAG_ANSWER_MAX_CONTINUATIONS="2", RAG_REQUIRE_ANSWER_COMPLETION_MARKER="true"):
            answer = rag._continue_if_truncated(
                model="gemini-3.1-flash-lite",
                base_contents=["system"],
                answer=response.text,
                first_response=response,
                max_output_tokens=4096,
            )

        self.assertEqual(answer, response.text)
        self.assertEqual(rag.client.models.calls, [])

    def test_motorbike_helmet_alcohol_red_light_question_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Tôi chạy xe không đội mũ bảo hiểm, say xỉn, vượt đèn đỏ thì bị phạt như thế nào?",
            [{"rag_modality": "text"}],
        )

        self.assertIn("400.000 - 600.000 đồng", answer)
        self.assertIn("4.000.000 - 6.000.000 đồng", answer)
        self.assertIn("2.000.000 - 3.000.000 đồng", answer)
        self.assertIn("8.000.000 - 10.000.000 đồng", answer)
        self.assertIn("12.400.000 - 16.600.000 đồng", answer)
        self.assertIn("Nghị định 168/2024/NĐ-CP", answer)

    def test_motorbike_compound_penalty_question_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Tôi chưa đủ tuổi chạy xe máy, say xỉn, vượt đèn đỏ, đi ngược chiều, không đội mũ bảo hiểm, gây tai nạn cho người khác thì hậu quả như thế nào?",
            [
                {
                    "rag_modality": "text",
                    "doc_name": "Nghị định 168/2024/NĐ-CP",
                    "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "7", "point": "c"},
                    "source_body_exact": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông; vượt đèn đỏ.",
                    "penalties": {
                        "main_penalty": {"individual_min_vnd": 4000000, "individual_max_vnd": 6000000},
                        "point_deduction": 4,
                    },
                    "retrieval_score": 10.0,
                },
                {
                    "rag_modality": "text",
                    "doc_name": "Luật Trật tự ATGT 2024 (Tiếp)",
                    "legal_reference": {"document": "Luật Trật tự ATGT 2024 (Tiếp)", "article": "59", "clause": "1"},
                    "source_body_exact": "Người điều khiển xe mô tô phải đủ tuổi theo quy định.",
                    "retrieval_score": 9.0,
                },
            ],
        )

        self.assertIn("Phân tích từng hành vi", answer)
        self.assertIn("Chưa đủ tuổi điều khiển xe máy", answer)
        self.assertIn("Say xỉn / nồng độ cồn", answer)
        self.assertIn("Vượt đèn đỏ / không chấp hành tín hiệu đèn", answer)
        self.assertIn("Đi ngược chiều / đi vào đường cấm", answer)
        self.assertIn("Không đội mũ bảo hiểm", answer)
        self.assertIn("Gây tai nạn cho người khác", answer)
        self.assertIn("Nghị định 168/2024/NĐ-CP", answer)

    def test_extractive_penalty_answer_uses_standard_sections(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._extractive_answer(
            "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
            [
                {
                    "doc_name": "Nghị định 168/2024/NĐ-CP",
                    "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "7", "point": "c"},
                    "source_body_exact": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
                    "penalties": {
                        "main_penalty": {"individual_min_vnd": 4000000, "individual_max_vnd": 6000000},
                        "point_deduction": 4,
                    },
                    "retrieval_score": 10.0,
                }
            ],
        )

        self.assertIn("## Trả lời ngắn gọn", answer)
        self.assertIn("## Phân tích từng nhánh", answer)
        self.assertIn("## Căn cứ áp dụng", answer)
        self.assertIn("## Tổng hậu quả", answer)
        self.assertIn("Trừ điểm GPLX: 4", answer)

    def test_signal_light_question_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Đèn vàng và đèn đỏ thì phải đi như thế nào?",
            [
                {
                    "doc_name": "QCVN 41:2024 (Thông tư 51/2024)",
                    "legal_reference": {"document": "QCVN 41:2024 (Thông tư 51/2024)", "section": "6.3.2 - 6.3.5"},
                    "source_body_exact": (
                        "6.3.2. Tín hiệu đèn màu vàng phải dừng lại trước vạch dừng; "
                        "trường hợp đang đi trên vạch dừng hoặc đã đi qua vạch dừng mà tín hiệu đèn màu vàng thì được đi tiếp. "
                        "Trường hợp tín hiệu đèn màu vàng nhấp nháy, người Điều khiển phương tiện tham gia giao thông đường bộ được đi "
                        "nhưng phải quan sát, giảm tốc độ hoặc dừng lại nhường đường cho người đi bộ, xe lăn của người khuyết tật qua đường hoặc các phương tiện khác. "
                        "6.3.3. Tín hiệu đèn màu đỏ là cấm đi: báo hiệu phải dừng lại trước vạch dừng. "
                        "Nếu không có vạch dừng thì phải dừng trước đèn tín hiệu theo chiều đi."
                    ),
                }
            ],
        )

        self.assertIn("Đèn vàng cố định", answer)
        self.assertIn("Đèn vàng nhấp nháy", answer)
        self.assertIn("Đèn đỏ", answer)
        self.assertIn("QCVN 41:2024", answer)

    def test_motorbike_red_light_penalty_answers_without_model(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._deterministic_structured_answer(
            "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
            [
                {
                    "doc_name": "Nghị định 168/2024/NĐ-CP",
                    "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "7", "point": "c"},
                    "source_body_exact": "c) Không chấp hành hiệu lệnh của đèn tín hiệu giao thông;",
                }
            ],
        )

        self.assertIn("4.000.000 - 6.000.000 đồng", answer)
        self.assertIn("4 điểm", answer)
        self.assertIn("Điểm c khoản 7", answer)

    def test_extractive_fallback_synthesizes_relevant_rules(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._extractive_answer(
            "Đèn vàng và đèn đỏ thì phải đi như thế nào?",
            [
                {
                    "doc_name": "QCVN 41:2024 (Thông tư 51/2024)",
                    "legal_reference": {"document": "QCVN 41:2024 (Thông tư 51/2024)", "section": "6.3"},
                    "source_body_exact": (
                        "6.3.2. Tín hiệu đèn màu vàng phải dừng lại trước vạch dừng; "
                        "trường hợp đang đi trên vạch dừng hoặc đã đi qua vạch dừng mà tín hiệu đèn màu vàng thì được đi tiếp. "
                        "# 6.3.3. Tín hiệu đèn màu đỏ là cấm đi: báo hiệu phải dừng lại trước vạch dừng. "
                        "Nếu không có vạch dừng thì phải dừng trước đèn tín hiệu theo chiều đi."
                    ),
                    "retrieval_score": 9.0,
                },
                {
                    "doc_name": "Nghị định 168/2024/NĐ-CP",
                    "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "3", "point": "a"},
                    "source_body_exact": "a) Chuyển hướng không quan sát hoặc không gi�7, Điều 7, Chương II.",
                    "retrieval_score": 8.0,
                },
            ],
        )

        self.assertNotIn("Tôi tìm thấy các căn cứ", answer)
        self.assertIn("Trả lời ngắn gọn", answer)
        self.assertIn("đèn màu vàng phải dừng", answer)
        self.assertIn("đèn màu đỏ là cấm đi", answer)
        self.assertNotIn("gi�7", answer)

    def test_extractive_fallback_uses_penalty_metadata(self):
        rag = LegalGraphRAG.__new__(LegalGraphRAG)
        answer = rag._extractive_answer(
            "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
            [
                {
                    "doc_name": "Nghị định 168/2024/NĐ-CP",
                    "legal_reference": {"document": "Nghị định 168/2024/NĐ-CP", "article": "7", "clause": "7", "point": "c"},
                    "source_body_exact": "Không chấp hành hiệu lệnh của đèn tín hiệu giao thông.",
                    "penalties": {
                        "main_penalty": {"individual_min_vnd": 4000000, "individual_max_vnd": 6000000},
                        "point_deduction": 4,
                    },
                    "retrieval_score": 10.0,
                }
            ],
        )

        self.assertIn("4.000.000 đồng - 6.000.000 đồng", answer)
        self.assertIn("Trừ điểm GPLX: 4", answer)
        self.assertIn("Điểm c, Khoản 7, Điều 7", answer)


if __name__ == "__main__":
    unittest.main()
