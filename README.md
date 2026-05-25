<!--
title: LuatGiaoThongDuongBoAI
emoji: "🚦"
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
-->

# Luật giao thông đường bộ (RAG + Gemini-3.1-flash-lite + Gemma-4-31b-it)

Vietnamese Traffic Law AI is a personal RAG project for looking up Vietnamese road traffic law. It combines a Streamlit UI, FastAPI backend, Qdrant vector retrieval, OpenVINO-accelerated `bkai-foundation-models/vietnamese-bi-encoder` embeddings, graph expansion, table/sign evidence, and optional Gemini synthesis.

The system is built for practical legal QA: vague questions, traffic-sign questions, penalty lookup, article-level detail, table/image evidence, multi-law reasoning, and out-of-scope detection.

## Main Capabilities

- Text QA for Vietnamese road traffic law with article/clause/point references.
- Query planning and sequential retrieval for complex real-world situations.
- Graph expansion across related legal articles, clauses, traffic signs, tables, and source images.
- Traffic-sign lookup and image-based sign recognition.
- Deterministic fallbacks for key cases when the LLM is unavailable: P.127, speed violations, broad penalty questions, max/min fines, max point deduction, and out-of-scope questions.
- Source search, graph trace, claim verification, and data status panels in the frontend.
- Qdrant-backed semantic retrieval using the exported OpenVINO vietnamese-bi-encoder model under `data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder`.

## Legal Data Included

The prepared dataset currently covers:

- Nghị định 168/2024/NĐ-CP
- Nghị định 336/2025/NĐ-CP
- Luật Đường bộ 2024
- Luật Trật tự, an toàn giao thông đường bộ 2024
- QCVN 41:2024 / Thông tư 51/2024/TT-BGTVT
- Thông tư 35/2024/TT-BGTVT
- Curated handmade QA and sign/table records under `data/processed/handmade`

Large data assets under `data/processed` and `data/graph/legal_graph.json` are tracked with Git LFS. Install Git LFS before cloning if you need the full dataset locally.

## Architecture

```text
Streamlit UI
    |
FastAPI API
    |
LegalGraphRAG
    |-- Query planner and adaptive analyzer
    |-- Sequential retrieval orchestrator
    |-- Qdrant vector store
    |-- OpenVINO vietnamese-bi-encoder embedder
    |-- Local graph JSON or Neo4j
    |-- Table/sign/image retrievers
    |-- Gemini model fallback policy
```

## Requirements

- Python 3.10 or newer
- 8 GB RAM minimum for local Qdrant/OpenVINO retrieval
- Git LFS for full data checkout
- Qdrant service on `QDRANT_URL` for the default runtime
- Optional: Docker, Neo4j, PostgreSQL, MinIO for local full-stack runs

## Local Setup

```bash
cd Legal-QA-Traffic-Laws
git lfs pull

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` and set at least `GEMINI_API_KEY` if you want LLM-generated answers and image sign recognition. Without a Gemini key, the system still serves deterministic/extractive answers for many routes.

Start Qdrant before running the API:

```bash
docker compose -f docker-compose.rag.yml up -d qdrant
```

Default runtime settings are:

```env
RAG_VECTOR_BACKEND=qdrant
RAG_GRAPH_BACKEND=local
RAG_ENABLE_EMBEDDINGS=true
RAG_EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder
RAG_EMBEDDING_BACKEND=openvino
RAG_EMBEDDING_DIMENSION=768
RAG_EMBEDDING_MAX_LENGTH=256
RAG_OPENVINO_MODEL_DIR=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder
RAG_STRICT_VECTOR_BACKEND=true
RAG_ENABLE_RERANKER=true
QDRANT_COLLECTION=legal_traffic_records_vi
```

Run the API and frontend:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8002 --reload
TRAFFIC_LAW_API_URL=http://localhost:8002 streamlit run frontend/app.py --server.port 7860
```

Open `http://localhost:7860`.

## Docker

```bash
docker build -t traffic-law-ai .
docker run --rm -p 7860:7860 --env-file .env traffic-law-ai
```

The container starts FastAPI on port `8002` internally and Streamlit on port `7860`. Provide a reachable `QDRANT_URL` when running the Docker image.

## Hugging Face Spaces

This repository can run on Hugging Face Spaces with `sdk: docker`.

1. Create a new Space with SDK `Docker`.
2. Push the repository, including Git LFS files.
3. Add secrets in **Settings > Secrets**:
   - `GEMINI_API_KEY`
   - `QDRANT_URL` and optional `QDRANT_API_KEY`
4. Use `.env.huggingface.example` as the Space variable template. It keeps the serving path on Qdrant + OpenVINO with 768-dimension vectors and enables AI planner, lexical hybrid search, and reranker.

The OpenVINO model is expected at `data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder`. Set `RAG_OPENVINO_EXPORT=true` only when you intentionally want to export/download a missing model. PostgreSQL is not required for online answering; it is only used by the sync pipeline for canonical record storage.

## Full-Stack Backends

Start local infrastructure:

```bash
docker compose -f docker-compose.rag.yml up -d postgres qdrant neo4j minio
cp .env.rag.example .env
```

Then sync stores:

```bash
python -m src.data_pipeline.rag_store_sync
```

Qdrant is the default vector store. Neo4j, PostgreSQL, and MinIO remain optional for graph persistence, canonical records, and asset storage.

## Rebuild Rules

Rebuild Qdrant whenever `RAG_EMBEDDING_MODEL`, `RAG_EMBEDDING_BACKEND`, `RAG_EMBEDDING_DIMENSION`, or `QDRANT_EMBED_TEXT_MAX_CHARS` changes:

```bash
RAG_VECTOR_BACKEND=qdrant \
RAG_EMBEDDING_BACKEND=openvino \
RAG_EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder \
RAG_EMBEDDING_DIMENSION=768 \
RAG_EMBEDDING_MAX_LENGTH=256 \
python -m src.data_pipeline.rag_store_sync --skip-postgres --skip-neo4j --skip-minio
```

The Qdrant store validates the collection vector size and embedding metadata on startup. A stale collection is recreated automatically. The graph does not depend on embedding vectors; rebuild or resync graph only when `data/processed` or graph export logic changes.

## Useful API Endpoints

- `GET /health`
- `GET /system/status`
- `GET /sources/search`
- `GET /graph/trace`
- `POST /chat/analyze`
- `POST /chat/text`
- `POST /chat/sign`
- `POST /chat/table`
- `POST /chat/image`
- `POST /chat/verify`

## Validation

Syntax check:

```bash
find api frontend scripts src -name '*.py' -print0 | xargs -0 python -m py_compile
bash -n entrypoint.sh
find scripts -name '*.sh' -print0 | xargs -0 bash -n
```

Configuration sanity check:

```bash
python - <<'PY'
from src.rag.rag_store_config import RAGStoreConfig

config = RAGStoreConfig()
print(config.vector_backend)
print(config.embedding_backend)
print(config.embedding_model)
print(config.openvino_model_dir)
PY
```

## Repository Hygiene

- Do not commit `.env`, local caches, `.venv`, or generated reports.
- Large legal assets are managed through Git LFS.
- `data/vector_db` is generated locally and intentionally not required for a fresh local build.
- Keep README, `.env.example`, and `requirements.txt` updated whenever runtime behavior changes.

## Disclaimer

This project supports legal research and traffic-law lookup. It is not a substitute for official legal advice or the original legal documents. For administrative or enforcement decisions, verify against official promulgated texts and competent authorities.
