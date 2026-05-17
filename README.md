# Legal-QA Traffic Laws

Hệ thống RAG cho tra cứu luật giao thông Việt Nam. Project tập trung vào trích xuất PDF pháp lý, lập catalog biển báo/bảng, lưu vector/graph/canonical records và đánh giá truy xuất bằng ground truth.

## Kiến trúc

1. `src/data_pipeline`: trích xuất PDF, chuẩn hóa Điều/Khoản/Điểm, bảng, hình và quan hệ tham chiếu.
2. `src/rag`: truy xuất hybrid qua Qdrant/FAISS, graph expansion qua Neo4j/JSON, route riêng cho biển báo và bảng.
3. `api`: FastAPI endpoint cho text, ảnh biển báo và bảng.
4. `src/evaluation`: benchmark embedding và đánh giá RAG bằng Recall@k, MRR, ref hit, modality hit, latency.

## Cấu trúc thư mục
```text
├── api/                # FastAPI app
├── data/
│   ├── raw/            # PDF và kết quả parse thô
│   ├── processed/      # Legal records đã chuẩn hóa
│   ├── graph/          # Legal graph JSON
│   ├── qa_pairs/       # QA sinh từ dữ liệu pháp lý
│   └── eval/           # Ground truth và report đánh giá
├── scripts/            # Lệnh vận hành/đánh giá
├── src/
│   ├── data_pipeline/  # PDF extraction, parsers, audits, store sync
│   ├── rag/            # Retriever, vector store, graph store, answer generation
│   └── evaluation/     # Metrics, evaluator, embedding benchmark
└── requirements.txt
```

## Cài đặt
```bash
pip install -r requirements.txt
```

## RAG stores

Mặc định hệ thống vẫn chạy local với FAISS/BM25 và graph JSON. Để chạy với store dịch vụ, dựng hạ tầng sau:

```bash
docker compose -f docker-compose.rag.yml up -d
```

Tạo cấu hình từ mẫu:

```bash
cp .env.rag.example .env
```

Các backend chính:

- `Qdrant`: lưu dense vectors kèm payload filter theo `doc`, `article`, `modality`, `has_table`, `has_sign`, `sign_codes`.
- `Neo4j`: lưu graph pháp lý `document -> article -> clause -> point -> table/figure/sign/reference`.
- `PostgreSQL`: lưu canonical records, bảng đã parse theo rows/headers, sign catalog và bản sao expanded RAG records.
- `MinIO`: lưu ảnh trang, crop biển báo, ảnh bảng.

Nạp dữ liệu vào các store RAG:

```bash
python -m src.data_pipeline.rag_store_sync --dry-run
python -m src.data_pipeline.rag_store_sync
```

Hoặc chạy cùng pipeline:

```bash
python -m src.data_pipeline.run_pipeline --skip-extraction --skip-qa --sync-rag-stores
```

Khi muốn API đọc từ Qdrant/Neo4j, đặt:

```bash
export RAG_VECTOR_BACKEND=qdrant
export RAG_GRAPH_BACKEND=neo4j
export RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
export RAG_EMBEDDING_BACKEND=auto
```

### Tối ưu local Intel Mac

Trên Mac Intel/Core i9, cấu hình mặc định nên ưu tiên CPU ổn định:

```bash
export RAG_EMBEDDING_DEVICE=cpu
export RAG_EMBEDDING_BATCH_SIZE=64
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

Nếu đã cài `openvino` và `optimum-intel[openvino]`, có thể export model OpenVINO một lần:

```bash
RAG_EMBEDDING_BACKEND=openvino \
RAG_OPENVINO_EXPORT=true \
RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
python -m src.evaluation.embedding_benchmark --backend openvino --limit 64
```

Sau đó chạy backend `auto` hoặc `openvino`. Nếu OpenVINO chưa sẵn sàng, `auto` sẽ fallback về SentenceTransformers.

Khi trả lời bằng Gemini, hệ thống sẽ thử `RAG_ANSWER_MODEL` trước, rồi tự fallback sang `RAG_ANSWER_FALLBACK_MODELS`
nếu model đầu bị hết quota/rate limit. Mặc định fallback là `gemma-4-31b-it`.

Chạy đánh giá RAG:

```bash
RETRIEVAL_ONLY=1 NO_RERANKER=1 scripts/run_rag_evaluation.sh
```

Kết quả nằm trong `data/eval/reports/rag_eval_*`, gồm summary, failures, manifest môi trường và embedding benchmark.

## Pipeline trích xuất dữ liệu luật

Pipeline chuẩn hóa dữ liệu theo các tầng:

1. `PDF -> text layer/OCR fallback`
2. `clean -> chapter/article/clause/point`
3. `context-preserving chunks`
4. `cross-reference graph`
5. `deterministic QA pairs`

### Chạy pipeline
```bash
python -m src.data_pipeline.run_pipeline
```

Chỉ trích xuất dữ liệu cấu trúc:
```bash
python -m src.data_pipeline.run_pipeline --skip-qa
```

### Output chính

- `data/processed/*.extracted.json`: legal records đã chuẩn hóa theo từng văn bản.
- `data/chunks/*.chunks.jsonl`: chunk nguồn giữ ngữ cảnh, bảng, hình và tọa độ trang.
- `data/graph/legal_graph.json`: graph tham chiếu giữa văn bản, điều khoản, bảng và biển báo.
- `data/qa_pairs/*.qa.json`: bộ câu hỏi đáp sinh tự động bám đúng điều khoản.

### Ghi chú OCR

- Pipeline tự ưu tiên `tesseract` với ngôn ngữ `vie` nếu máy có sẵn.
- Nếu máy chưa có `vie`, pipeline sẽ fallback sang `eng`; vẫn nhận diện được cấu trúc `Điều/Khoản/Điểm` cho PDF scan nhưng độ chính xác tiếng Việt sẽ thấp hơn.
- Với các PDF scan quan trọng như `168-nd-cp.signed.pdf` và `336nd.signed.pdf`, nên cài thêm `vie.traineddata` để tối đa độ chính xác.
