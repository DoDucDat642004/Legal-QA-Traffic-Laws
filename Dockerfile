FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    RAG_VECTOR_BACKEND=qdrant \
    RAG_GRAPH_BACKEND=local \
    RAG_ENABLE_EMBEDDINGS=true \
    RAG_ALLOW_MODEL_DOWNLOAD=false \
    RAG_ENABLE_RERANKER=true \
    RAG_RERANKER_BACKEND=openvino \
    RAG_ALLOW_RERANKER_DOWNLOAD=false \
    RAG_OPENVINO_RERANKER_MODEL_DIR=data/models/openvino/cross-encoder_mmarco-mMiniLMv2-L12-H384-v1 \
    RAG_EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder \
    RAG_EMBEDDING_BACKEND=openvino \
    RAG_EMBEDDING_DIMENSION=768 \
    RAG_EMBEDDING_MAX_LENGTH=256 \
    RAG_OPENVINO_DEVICE=CPU \
    RAG_OPENVINO_MODEL_DIR=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder \
    RAG_STRICT_VECTOR_BACKEND=true \
    RAG_ENABLE_AI_PLANNER=true \
    RAG_AI_PLANNER_ALWAYS=false \
    RAG_AI_PLANNER_MIN_RULE_CONFIDENCE=0.72 \
    RAG_AI_PLANNER_MAX_QUERIES=2 \
    RAG_MAX_PLANNED_QUERIES=3 \
    RAG_RETRIEVAL_MAX_ROUNDS=3 \
    RAG_RETRIEVAL_MAX_SLOTS=18 \
    RAG_VECTOR_SEARCH_MULTIPLIER=2 \
    RAG_QDRANT_VECTOR_MULTIPLIER=4 \
    RAG_QDRANT_ENABLE_LEXICAL=true \
    RAG_RERANKER_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
    RAG_MODEL_RERANK_LIMIT=32 \
    RAG_RERANKER_DEVICE=cpu \
    OPENVINO_TELEMETRY_DISABLE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    QDRANT_COLLECTION=legal_traffic_records_vi \
    QDRANT_TIMEOUT=300

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ghostscript \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["./entrypoint.sh"]
