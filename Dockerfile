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
    RAG_OPENVINO_RERANKER_MODEL_DIR=data/models/openvino/BAAI_bge-reranker-v2-m3 \
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
    RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    RAG_MODEL_RERANK_LIMIT=32 \
    RAG_RERANKER_DEVICE=cpu \
    OPENVINO_TELEMETRY_DISABLE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    QDRANT_COLLECTION=legal_traffic_records_vi \
    QDRANT_TIMEOUT=300

WORKDIR /app

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
    git \
    git-lfs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN set -eux; \
    needs_models=0; \
    for model_file in \
      data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder/openvino_model.bin \
      data/models/openvino/BAAI_bge-reranker-v2-m3/openvino_model.bin; do \
      if [ ! -s "$model_file" ] || head -c 64 "$model_file" | grep -q "version https://git-lfs.github.com/spec/v1"; then \
        needs_models=1; \
      fi; \
    done; \
    if [ "$needs_models" = "1" ]; then \
      echo "OpenVINO models missing or unresolved; fetching model artifacts from GitHub LFS"; \
      git lfs install --skip-repo; \
      tmp_dir="$(mktemp -d)"; \
      GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse https://github.com/DoDucDat642004/Legal-QA-Traffic-Laws.git "$tmp_dir"; \
      git -C "$tmp_dir" sparse-checkout set \
        data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder \
        data/models/openvino/BAAI_bge-reranker-v2-m3; \
      (while sleep 20; do echo "Still downloading OpenVINO model artifacts from Git LFS..."; done) & \
      progress_pid="$!"; \
      GIT_LFS_PROGRESS=/dev/stderr git -C "$tmp_dir" lfs pull --include="data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder/**,data/models/openvino/BAAI_bge-reranker-v2-m3/**" --exclude=""; \
      kill "$progress_pid" 2>/dev/null || true; \
      wait "$progress_pid" 2>/dev/null || true; \
      mkdir -p data/models/openvino; \
      cp -a "$tmp_dir/data/models/openvino/." data/models/openvino/; \
      rm -rf "$tmp_dir"; \
    fi; \
    for model_file in \
      data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder/openvino_model.bin \
      data/models/openvino/BAAI_bge-reranker-v2-m3/openvino_model.bin; do \
      if [ ! -s "$model_file" ] || head -c 64 "$model_file" | grep -q "version https://git-lfs.github.com/spec/v1"; then \
        echo "OpenVINO model hydration failed: $model_file" >&2; \
        exit 1; \
      fi; \
    done

EXPOSE 7860
ENTRYPOINT ["bash", "entrypoint.sh"]
