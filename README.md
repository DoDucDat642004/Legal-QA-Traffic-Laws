# Hệ thống Hỏi - Đáp Luật Giao thông Đường bộ (Legal-QA Traffic Laws)

Dự án xây dựng hệ thống hỏi đáp tự động về Luật Trật tự, An toàn giao thông đường bộ (Luật số 36/2024/QH15) sử dụng phương pháp Retrieval-Augmented Generation (RAG) và so sánh với mô hình Baseline LSTM.

## 📋 Tổng quan Dự án

Đồ án tập trung vào việc giải quyết khó khăn trong tra cứu luật pháp truyền thống bằng cách kết hợp sức mạnh của mô hình ngôn ngữ lớn (LLM) và cơ sở dữ liệu luật thực tế.

- **Dữ liệu mục tiêu:** Luật Trật tự, an toàn giao thông đường bộ 2024.
- **Giải pháp chính:** RAG (Retrieval-Augmented Generation).
- **Mô hình cơ sở:** RNN/LSTM.

## 🏗️ Kiến trúc Hệ thống (Proposed Pipeline)

1.  **Data Pipeline:** Thu thập, làm sạch và chunking dữ liệu luật.
2.  **Retrieval:** Sử dụng `vinai/phobert-base` để chuyển đổi văn bản thành vector, lưu trữ tại `ChromaDB`.
3.  **Generation:** Sử dụng LLM (Gemma-2B/Qwen2-1.5B hoặc Gemini API) với kỹ thuật Few-shot và Chain-of-Thought (CoT).
4.  **Evaluation:** Đánh giá qua Recall@k, BLEU score và Accuracy dựa trên bộ dataset Q|A|Ref.

## 📅 Lộ trình Triển khai (Phases)

### Phase 1: Chuẩn bị Dữ liệu & Khai phá (Tuần 1-2)
*   Xây dựng bộ Dataset (300 mẫu Q|A|Ref).
*   Thực hiện EDA & Text Mining (POS Tagging, NER).
*   Hoàn thiện báo cáo tiến độ và slide.

### Phase 2: Mô hình cơ sở (Tuần 3)
*   Xây dựng và huấn luyện mô hình RNN/LSTM làm Baseline để so sánh hiệu năng.

### Phase 3: Giải pháp RAG "Ăn tiền" (Tuần 4)
*   Thiết lập Vector Database với PhoBERT.
*   Tích hợp LLM (Local/API).
*   Áp dụng Prompt Engineering (Few-shot, CoT).

### Phase 4: Đánh giá & Đóng gói (Tuần 5)
*   Tính toán các chỉ số Metrics (Recall, BLEU, Accuracy).
*   Lập bảng so sánh kết quả giữa các phương pháp.

## 📂 Cấu trúc Thư mục
```text
├── api/                # FastAPI/Flask app
├── data/
│   ├── raw/           # PDF luật gốc
│   ├── processed/     # Dữ liệu sau khi làm sạch
│   └── qa_pair/       # File Excel/CSV Q-A-Ref
├── notebooks/         # EDA và thử nghiệm
├── src/
│   ├── data_pipeline/ # Xử lý parser dữ liệu
│   ├── models/        # Code LSTM & RAG pipeline
│   └── evaluation/    # Script tính toán metrics
└── requirements.txt
```

## 🛠️ Cài đặt
```bash
pip install -r requirements.txt
```
