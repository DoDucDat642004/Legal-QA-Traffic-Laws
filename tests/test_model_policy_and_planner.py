import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from src.rag.hybrid_vector_store import _should_load_embedder
from src.rag.legal_graph_rag import LegalGraphRAG
from src.rag.model_policy import generate_content_with_fallback, model_candidates
from src.rag.query_planner import LegalQueryPlanner


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
        if isinstance(response, Exception):
            raise response
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


class DeterministicPenaltyAnswerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
