FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HOME=/tmp \
    HF_HOME=/tmp/.cache/huggingface \
    XDG_CACHE_HOME=/tmp/.cache \
    MPLCONFIGDIR=/tmp/.cache/matplotlib \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    RAG_DEPLOY_FORCE_LOCAL_MODE=true \
    RAG_DEPLOY_FAST_MODE=true \
    RAG_VECTOR_BACKEND=local \
    RAG_GRAPH_BACKEND=local \
    RAG_ENABLE_EMBEDDINGS=false \
    RAG_ALLOW_MODEL_DOWNLOAD=false \
    RAG_ENABLE_RERANKER=false \
    RAG_RERANKER_BACKEND=openvino \
    RAG_ALLOW_RERANKER_DOWNLOAD=false \
    RAG_OPENVINO_RERANKER_MODEL_DIR=data/models/openvino/BAAI_bge-reranker-v2-m3 \
    RAG_EMBEDDING_MODEL=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder \
    RAG_EMBEDDING_BACKEND=openvino \
    RAG_EMBEDDING_DIMENSION=768 \
    RAG_EMBEDDING_MAX_LENGTH=256 \
    RAG_OPENVINO_DEVICE=CPU \
    RAG_OPENVINO_MODEL_DIR=data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder \
    RAG_STRICT_VECTOR_BACKEND=false \
    RAG_ENABLE_AI_PLANNER=false \
    RAG_AI_PLANNER_ALWAYS=false \
    RAG_AI_PLANNER_MIN_RULE_CONFIDENCE=0.72 \
    RAG_AI_PLANNER_MAX_QUERIES=2 \
    RAG_MAX_PLANNED_QUERIES=3 \
    RAG_RETRIEVAL_MAX_ROUNDS=1 \
    RAG_RETRIEVAL_MAX_SLOTS=8 \
    RAG_FAST_MAX_CONTEXTS=18 \
    RAG_FAST_TOP_K=18 \
    RAG_FAST_EXPAND_DEPTH=1 \
    RAG_FAST_MAX_IMAGES=12 \
    RAG_FAST_MAX_PROMPT_IMAGES=0 \
    RAG_API_IMAGE_LIMIT=12 \
    RAG_INCLUDE_GRAPH_TRACE=false \
    RAG_CHAT_TEXT_DEADLINE_SECONDS=120 \
    RAG_EXTRACTIVE_ANSWER_ONLY=true \
    RAG_ENABLE_SIGN_AI_PROBE=false \
    WARMUP_RAG_ON_START=true \
    RAG_PROMPT_CONTEXT_TEXT_LIMIT=8000 \
    RAG_PROMPT_STRUCTURED_TEXT_LIMIT=16000 \
    RAG_ANSWER_MAX_OUTPUT_TOKENS=4096 \
    RAG_ANSWER_MAX_CONTINUATIONS=0 \
    RAG_EXTRACTIVE_MAX_CONTEXTS=8 \
    RAG_EXTRACTIVE_TEXT_LIMIT=1200 \
    RAG_VECTOR_SEARCH_MULTIPLIER=2 \
    RAG_QDRANT_VECTOR_MULTIPLIER=4 \
    RAG_QDRANT_ENABLE_LEXICAL=true \
    RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    RAG_MODEL_RERANK_LIMIT=32 \
    RAG_RERANKER_DEVICE=CPU \
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

RUN useradd -m -u 1000 user \
    && mkdir -p /tmp/.cache/huggingface /tmp/.cache/matplotlib /tmp/.cache/fontconfig /tmp/intel \
    && chown -R user:user /app /tmp/.cache /tmp/intel \
    && chmod -R 777 /tmp/.cache /tmp/intel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user

RUN set -eux; \
    mkdir -p data/processed data/graph data/models/openvino; \
    needs_lfs=0; \
    for required_file in \
      data/graph/legal_graph.json \
      data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder/openvino_model.bin \
      data/models/openvino/BAAI_bge-reranker-v2-m3/openvino_model.bin; do \
      if [ ! -s "$required_file" ] || head -c 64 "$required_file" | grep -q "version https://git-lfs.github.com/spec/v1"; then \
        needs_lfs=1; \
      fi; \
    done; \
    processed_count="$(find data/processed -maxdepth 1 -type f -name '*.extracted.json' | wc -l)"; \
    if [ "$processed_count" -eq 0 ]; then \
      needs_lfs=1; \
    else \
      for data_file in data/processed/*.extracted.json; do \
        [ -e "$data_file" ] || continue; \
        if [ ! -s "$data_file" ] || head -c 64 "$data_file" | grep -q "version https://git-lfs.github.com/spec/v1"; then \
          needs_lfs=1; \
        fi; \
      done; \
    fi; \
    if [ "$needs_lfs" = "1" ]; then \
      echo "Required LFS data missing or unresolved; fetching data, graph, and OpenVINO model artifacts from GitHub LFS"; \
      git lfs install --skip-repo; \
      tmp_dir="$(mktemp -d)"; \
      GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --sparse https://github.com/DoDucDat642004/Legal-QA-Traffic-Laws.git "$tmp_dir"; \
      git -C "$tmp_dir" sparse-checkout set \
        data/processed \
        data/graph \
        data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder \
        data/models/openvino/BAAI_bge-reranker-v2-m3; \
      (while sleep 20; do echo "Still downloading required LFS artifacts..."; done) & \
      progress_pid="$!"; \
      GIT_LFS_PROGRESS=/dev/stderr git -C "$tmp_dir" lfs pull --include="data/processed/**,data/graph/**,data/models/openvino/bkai-foundation-models_vietnamese-bi-encoder/**,data/models/openvino/BAAI_bge-reranker-v2-m3/**" --exclude=""; \
      kill "$progress_pid" 2>/dev/null || true; \
      wait "$progress_pid" 2>/dev/null || true; \
      mkdir -p data/processed data/graph; \
      cp -a "$tmp_dir/data/processed/." data/processed/; \
      cp -a "$tmp_dir/data/graph/." data/graph/; \
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
    done; \
    python -c "from src.rag.record_expander import load_processed_records; records = load_processed_records('data/processed'); assert records, 'No legal records found at data/processed'; print(f'Loaded {len(records)} processed legal records')"; \
    python scripts/hf_deploy_smoke.py

EXPOSE 7860
ENTRYPOINT ["bash", "entrypoint.sh"]
