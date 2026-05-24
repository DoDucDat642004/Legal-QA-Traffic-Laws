# 🚦 Vietnamese Traffic Law AI (Legal-QA-RAG)

Hệ thống hỏi đáp pháp luật giao thông đường bộ Việt Nam sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)** tiên tiến, kết hợp giữa Graph DB và Vector DB để cung cấp câu trả lời chính xác, có căn cứ pháp lý và minh họa bằng hình ảnh.

## 🌟 Tính năng nổi bật

- **Hybrid Retrieval:** Kết hợp tìm kiếm ngữ nghĩa (FAISS) và tìm kiếm cấu trúc (Graph) để truy xuất Điều/Khoản chính xác.
- **Đa phương thức (Multimodal):** Nhận diện biển báo giao thông qua hình ảnh tải lên.
- **Xoay vòng Model thông minh:** Tự động chuyển đổi giữa Gemini 2.0 Flash, Flash-Lite và Gemma khi chạm ngưỡng giới hạn truy vấn (Quota).
- **Căn cứ hình ảnh:** Hiển thị ảnh trích xuất trực tiếp từ QCVN 41:2024 và các Nghị định.
- **Hỗ trợ 7 nguồn tài liệu mới nhất:**
  - Nghị định 168/2024/NĐ-CP (Xử phạt)
  - Luật Đường bộ 2024
  - Luật Trật tự ATGT 2024 (Phần 1 & 2)
  - QCVN 41:2024 (Biển báo & Vạch kẻ đường)
  - Thông tư 35/2024/TT-BGTVT
  - Nghị định 336/2025/NĐ-CP

## 🏗️ Kiến trúc hệ thống

```text
[Người dùng] <-> [Streamlit UI] <-> [FastAPI Backend]
                                          |
        ---------------------------------------------------
        |                 |               |               |
 [Query Planner]   [Vector Store]   [Graph Store]   [LLM Manager]
 (Intent Analysis) (Semantic Search) (Legal Hierarchy) (Model Rotation)
```

## 🚀 Hướng dẫn cài đặt

### 1. Yêu cầu hệ thống
- Python 3.10+
- RAM: Tối thiểu 8GB (để chạy FAISS và Sentence Transformers)

### 2. Cài đặt môi trường
```bash
# Clone dự án
git clone https://github.com/your-username/Legal-QA-Traffic-Laws.git
cd Legal-QA-Traffic-Laws

# Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Cấu hình
Tạo file `.env` tại thư mục gốc:
```env
GEMINI_API_KEY=your_api_key_here
RAG_VECTOR_BACKEND=local
RAG_GRAPH_BACKEND=local
```

### 4. Chạy ứng dụng
```bash
# Chạy Backend (Cổng 8002)
python -m uvicorn api.main:app --port 8002 --reload

# Chạy Frontend (Cổng 7860/8501)
streamlit run frontend/app.py
```

## ☁️ Triển khai (Deployment)

Dự án đã được tối ưu để chạy trên **Hugging Face Spaces** qua Docker:
1. Tạo Space mới trên Hugging Face (SDK: Docker).
2. Thêm `GEMINI_API_KEY` vào mục **Settings > Secrets**.
3. Push toàn bộ code và thư mục `data/processed` lên Space.

## 🛠️ Cấu trúc thư mục

- `api/`: Mã nguồn FastAPI Backend.
- `frontend/`: Giao diện người dùng Streamlit.
- `src/rag/`: Logic cốt lõi của hệ thống RAG (Retriever, Planner, Model Policy).
- `data/processed/`: Dữ liệu pháp luật đã được trích xuất và index.
- `scripts/`: Các script đánh giá và kiểm thử chất lượng câu trả lời.

## ⚖️ Giấy phép & Tuyên bố miễn trừ
Dự án được phát triển cho mục đích tra cứu hỗ trợ. Người dùng nên đối chiếu với văn bản pháp luật gốc trên Cổng Thông tin điện tử Chính phủ khi thực hiện các thủ tục hành chính.
