#!/usr/bin/env python3
"""Build-time smoke checks for the Hugging Face Docker Space."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EMBEDDING_DIR = ROOT / "data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder"
RERANKER_DIR = ROOT / "data/models/openvino/BAAI_bge-reranker-v2-m3"


def _force_offline_local_env() -> None:
    os.environ["HOME"] = "/tmp"
    os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
    os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
    os.environ["MPLCONFIGDIR"] = "/tmp/.cache/matplotlib"
    os.environ["OPENVINO_TELEMETRY_DISABLE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["RAG_GRAPH_BACKEND"] = "local"
    os.environ["RAG_CANONICAL_BACKEND"] = "local"
    os.environ["RAG_OBJECT_BACKEND"] = "local"
    os.environ["RAG_DEPLOY_FAST_MODE"] = "true"
    os.environ["RAG_ENABLE_EMBEDDINGS"] = "false"
    os.environ["RAG_ENABLE_RERANKER"] = "false"
    os.environ["RAG_ENABLE_AI_PLANNER"] = "false"
    os.environ["RAG_RETRIEVAL_MAX_ROUNDS"] = "1"
    os.environ["RAG_RETRIEVAL_MAX_SLOTS"] = "8"
    os.environ["RAG_FAST_MAX_CONTEXTS"] = "10"
    os.environ["RAG_FAST_TOP_K"] = "10"
    os.environ["RAG_FAST_EXPAND_DEPTH"] = "1"
    os.environ["RAG_FAST_MAX_IMAGES"] = "6"
    os.environ["RAG_INCLUDE_GRAPH_TRACE"] = "false"
    os.environ["RAG_API_IMAGE_LIMIT"] = "6"
    os.environ["RAG_EXTRACTIVE_ANSWER_ONLY"] = "true"
    os.environ["RAG_ENABLE_SIGN_AI_PROBE"] = "false"
    os.environ["RAG_ANSWER_MAX_OUTPUT_TOKENS"] = "4096"
    os.environ["RAG_ANSWER_MAX_CONTINUATIONS"] = "0"
    os.environ["RAG_ALLOW_MODEL_DOWNLOAD"] = "false"
    os.environ["RAG_ALLOW_RERANKER_DOWNLOAD"] = "false"
    os.environ["RAG_OPENVINO_EXPORT"] = "false"
    os.environ["RAG_OPENVINO_RERANKER_EXPORT"] = "false"
    os.environ["RAG_EMBEDDING_BACKEND"] = "openvino"
    os.environ["RAG_OPENVINO_MODEL_DIR"] = str(EMBEDDING_DIR)
    os.environ["RAG_EMBEDDING_MODEL"] = str(EMBEDDING_DIR)
    os.environ["RAG_RERANKER_BACKEND"] = "openvino"
    os.environ["RAG_OPENVINO_RERANKER_MODEL_DIR"] = str(RERANKER_DIR)
    os.environ["RAG_RERANKER_MODEL"] = str(RERANKER_DIR)


def _assert_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Required deploy artifact is missing or empty: {path}")
    with path.open("rb") as handle:
        if handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(f"Required deploy artifact is still a Git LFS pointer: {path}")


def _check_artifacts() -> None:
    for path in [
        ROOT / "data/graph/legal_graph.json",
        EMBEDDING_DIR / "config.json",
        EMBEDDING_DIR / "openvino_model.xml",
        EMBEDDING_DIR / "openvino_model.bin",
        EMBEDDING_DIR / "tokenizer_config.json",
        EMBEDDING_DIR / "vocab.txt",
        EMBEDDING_DIR / "bpe.codes",
        RERANKER_DIR / "config.json",
        RERANKER_DIR / "openvino_model.xml",
        RERANKER_DIR / "openvino_model.bin",
        RERANKER_DIR / "tokenizer_config.json",
    ]:
        _assert_file(path)
    print("Deploy smoke: required data and model artifacts are present.", flush=True)


def _check_records() -> None:
    from src.rag.record_expander import load_processed_records
    from src.rag.legal_utils import record_image_paths
    from frontend.asset_utils import image_source

    records = load_processed_records(ROOT / "data/processed")
    if not records:
        raise RuntimeError("No processed legal records found for deploy.")
    print(f"Deploy smoke: loaded {len(records)} processed records.")

    for record in records:
        paths = record_image_paths(record)
        if not paths:
            continue
        source = image_source(
            paths[0],
            api_url="http://127.0.0.1:8002",
            processed_dir=ROOT / "data/processed",
        )
        if not Path(source).is_file():
            raise RuntimeError(f"Processed image asset is not locally readable: {paths[0]} -> {source}")
        print(f"Deploy smoke: source image asset resolved locally: {source}", flush=True)
        break
    else:
        raise RuntimeError("No source image assets found in processed records.")


def _check_fast_queries() -> None:
    from src.rag.legal_graph_rag import LegalGraphRAG

    rag = LegalGraphRAG(
        ROOT / "data/processed",
        graph_path=ROOT / "data/graph/legal_graph.json",
        use_reranker=False,
    )
    queries = [
        "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        "Biển P.102 có ý nghĩa gì?",
        "Người điều khiển xe máy có nồng độ cồn cao bị xử lý thế nào?",
        "Ô tô vượt quá tốc độ 15 km/h bị phạt sao?",
    ]
    for query in queries:
        start = time.perf_counter()
        result = rag.query_adaptive(query)
        elapsed = time.perf_counter() - start
        answer = str(result.get("answer") or "").strip()
        contexts = result.get("contexts") or []
        if not answer or not contexts:
            raise RuntimeError(f"Fast query smoke returned incomplete result for: {query}")
        if elapsed > 60:
            raise RuntimeError(f"Fast query smoke exceeded 60s ({elapsed:.1f}s): {query}")
        print(
            f"Deploy smoke: fast query ok in {elapsed:.2f}s; "
            f"contexts={len(contexts)}; query={query}",
            flush=True,
        )


def main() -> int:
    _force_offline_local_env()
    try:
        _check_artifacts()
        _check_records()
        _check_fast_queries()
        print("Deploy smoke: offline deploy artifacts validated.", flush=True)
    except Exception as exc:
        print(f"Deploy smoke failed: {exc}", file=sys.stderr)
        print(
            "Safe config: "
            f"RAG_VECTOR_BACKEND={os.getenv('RAG_VECTOR_BACKEND')}, "
            f"RAG_ENABLE_EMBEDDINGS={os.getenv('RAG_ENABLE_EMBEDDINGS')}, "
            f"RAG_EMBEDDING_BACKEND={os.getenv('RAG_EMBEDDING_BACKEND')}, "
            f"RAG_OPENVINO_MODEL_DIR={os.getenv('RAG_OPENVINO_MODEL_DIR')}, "
            f"RAG_RERANKER_BACKEND={os.getenv('RAG_RERANKER_BACKEND')}, "
            f"RAG_OPENVINO_RERANKER_MODEL_DIR={os.getenv('RAG_OPENVINO_RERANKER_MODEL_DIR')}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
