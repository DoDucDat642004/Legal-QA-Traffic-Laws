#!/usr/bin/env python3
"""Build-time smoke checks for the Hugging Face Docker Space."""

from __future__ import annotations

import os
import sys
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

    records = load_processed_records(ROOT / "data/processed")
    if not records:
        raise RuntimeError("No processed legal records found for deploy.")
    print(f"Deploy smoke: loaded {len(records)} processed records.")


def main() -> int:
    _force_offline_local_env()
    try:
        _check_artifacts()
        _check_records()
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
