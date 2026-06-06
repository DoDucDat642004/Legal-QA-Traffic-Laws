import io
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asset_utils import image_source
from render_utils import split_markdown_sections, vision_display_rows
from src.rag.source_catalog import source_catalog_payload, source_info_for_document


API_URL = os.getenv("TRAFFIC_LAW_API_URL", "http://localhost:8002").rstrip("/")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


SHOW_ADVANCED_TOOLS = env_bool("SHOW_ADVANCED_TOOLS", False)
SHOW_RETRIEVAL_DETAILS = env_bool("SHOW_RETRIEVAL_DETAILS", False)
ENABLE_PRE_ANALYSIS = env_bool("ENABLE_PRE_ANALYSIS", False)
CHAT_REQUEST_TIMEOUT_SECONDS = int(os.getenv("CHAT_REQUEST_TIMEOUT_SECONDS", "420"))

st.set_page_config(page_title="Luật Giao Thông AI", layout="wide", page_icon="§")


st.markdown(
    """
    <style>
    :root {
        --bg: #f5f7f9;
        --panel: #ffffff;
        --border: #d8dee6;
        --text: #172033;
        --muted: #667085;
        --accent: #0f766e;
        --accent-dark: #115e59;
        --accent-soft: #e5f3ef;
        --warn-soft: #fff8e8;
    }
    .stApp {
        background: var(--bg);
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
    }
    .app-header {
        padding: 10px 0 14px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 18px;
    }
    .app-title {
        font-size: 26px;
        line-height: 1.2;
        font-weight: 750;
        color: var(--text);
        margin: 0;
    }
    .app-subtitle {
        font-size: 14px;
        color: var(--muted);
        margin-top: 6px;
        max-width: 820px;
    }
    .welcome-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0 14px 0;
    }
    .welcome-title {
        color: var(--text);
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .welcome-note {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
    }
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 10px;
        margin: 8px 0 14px 0;
    }
    .metric-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 11px 12px;
    }
    .metric-label {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 3px;
    }
    .metric-value {
        color: var(--text);
        font-size: 15px;
        font-weight: 650;
    }
    .stepper {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0 14px 0;
    }
    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 7px 0;
        border-bottom: 1px solid #eef1f5;
    }
    .step-row:last-child {
        border-bottom: none;
    }
    .step-dot {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        border: 1px solid var(--border);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        flex: 0 0 auto;
        margin-top: 1px;
    }
    .step-done .step-dot {
        background: var(--accent);
        border-color: var(--accent);
        color: #fff;
    }
    .step-active .step-dot {
        background: var(--accent-soft);
        border-color: var(--accent);
        color: var(--accent);
    }
    .step-wait .step-dot {
        background: #f5f6f8;
        color: var(--muted);
    }
    .step-title {
        font-size: 14px;
        font-weight: 650;
        color: var(--text);
    }
    .step-note {
        font-size: 12px;
        color: var(--muted);
        margin-top: 2px;
    }
    .source-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        margin: 6px 0;
    }
    .source-title {
        color: var(--text);
        font-weight: 650;
        font-size: 14px;
        margin-bottom: 4px;
    }
    .source-meta {
        color: var(--muted);
        font-size: 12px;
    }
    .small-note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
    }
    .gap-list {
        background: var(--warn-soft);
        border: 1px solid #f2d49b;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 8px 0 14px 0;
    }
    .action-band {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
    }
    .status-pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #f5f6f8;
        color: var(--muted);
        font-size: 11px;
        font-weight: 650;
        margin-right: 5px;
    }
    .status-supported {
        background: var(--accent-soft);
        border-color: #8ac8bd;
        color: var(--accent);
    }
    .status-weak {
        background: var(--warn-soft);
        border-color: #f2d49b;
        color: #8a5a00;
    }
    .status-review {
        background: #fff1f2;
        border-color: #fecdd3;
        color: #be123c;
    }
    .trace-list {
        border-left: 2px solid var(--border);
        padding-left: 10px;
        margin: 8px 0 12px 4px;
    }
    .answer-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0 16px 0;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .answer-title {
        color: var(--text);
        font-size: 15px;
        font-weight: 750;
        margin-bottom: 8px;
    }
    .answer-section {
        margin: 12px 0 4px 0;
        padding-top: 10px;
        border-top: 1px solid #eef1f5;
    }
    .answer-section:first-child {
        margin-top: 0;
        padding-top: 0;
        border-top: 0;
    }
    .answer-section-title {
        color: var(--accent-dark);
        font-size: 18px;
        line-height: 1.25;
        font-weight: 750;
        margin: 0 0 8px 0;
        padding-left: 10px;
        border-left: 4px solid var(--accent);
    }
    .answer-subtitle {
        color: var(--muted);
        font-size: 12px;
        margin-top: 2px;
    }
    .vision-panel {
        background: #f7fbff;
        border: 1px solid #cfe3ff;
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0 14px 0;
    }
    .vision-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 8px;
        margin-top: 10px;
    }
    .vision-item {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #d8e8fb;
        border-radius: 10px;
        padding: 8px 10px;
    }
    .vision-label {
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 3px;
    }
    .vision-value {
        color: var(--text);
        font-size: 13px;
        line-height: 1.4;
        font-weight: 600;
        word-break: break-word;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
        color: var(--text);
        line-height: 1.62;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
        margin: 0.35rem 0 0.6rem 0;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h4 {
        color: var(--text);
        margin: 1rem 0 0.55rem 0;
        line-height: 1.25;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2 {
        font-size: 1.18rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e6ebf1;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
        font-size: 1.03rem;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] ol {
        margin: 0.35rem 0 0.7rem 1.2rem;
        padding-left: 0.9rem;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] li {
        margin: 0.2rem 0;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        display: block;
        overflow-x: auto;
        margin: 0.5rem 0 0.9rem 0;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] th,
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] td {
        border: 1px solid #dbe3eb;
        padding: 0.48rem 0.6rem;
        text-align: left;
        vertical-align: top;
        white-space: normal;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] th {
        background: #f5f8fb;
        font-weight: 700;
    }
    div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] blockquote {
        border-left: 4px solid var(--accent);
        background: #f4fbf8;
        padding: 0.6rem 0.8rem;
        margin: 0.65rem 0;
        border-radius: 8px;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--border);
        min-height: 38px;
    }
    .stTextInput > div > div > input,
    .stTextArea textarea {
        border-radius: 8px;
    }
    .stChatInput textarea {
        min-height: 48px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PROCESSING_STEPS = [
    ("Tiếp nhận câu hỏi", "Đọc nội dung và xác định nhóm vấn đề pháp luật giao thông."),
    ("Tìm căn cứ", "Tra cứu văn bản, biển báo, bảng biểu và nguồn liên quan."),
    ("Đối chiếu", "Chọn căn cứ phù hợp và loại bỏ nguồn yếu."),
    ("Trả lời", "Tóm tắt kết luận kèm căn cứ để bạn kiểm tra lại."),
]


QUESTION_BANK = [
    "Xe máy vượt đèn đỏ bị phạt bao nhiêu và có bị trừ điểm GPLX không?",
    "Biển P.102 có ý nghĩa gì, đi vào đường có biển này bị phạt ra sao?",
    "Tôi lái ô tô có nồng độ cồn thì bị phạt, trừ điểm và tước GPLX thế nào?",
    "Xe máy chạy quá tốc độ 15 km/h trong khu dân cư bị xử lý ra sao?",
    "Đi xe máy không đội mũ bảo hiểm thì người lái và người ngồi sau bị phạt thế nào?",
    "Vừa lái xe vừa dùng điện thoại thì ô tô và xe máy bị phạt khác nhau không?",
    "Vượt đèn đỏ nhưng camera phạt nguội ghi lại thì mức xử phạt thế nào?",
    "Xe tôi dừng đỗ sát giao lộ làm cản trở giao thông thì bị phạt ra sao?",
    "Tôi có bằng A1 chạy xe hơi thì có được không?",
    "Chủ xe giao ô tô cho người chỉ có bằng A1 lái thì chủ xe có bị phạt không?",
    "Tôi chưa đủ tuổi chạy xe máy dung tích lớn thì người giao xe có bị phạt không?",
    "Tôi mượn xe người khác rồi vi phạm bị tạm giữ xe thì trách nhiệm thuộc về ai?",
    "Gặp xe cứu thương bật còi ưu tiên ở ngã tư thì xe máy phải nhường đường thế nào?",
    "Cơ sở giáo dục tự tổ chức xe đưa đón học sinh thì phải theo quy định nào?",
    "Tài xế xe công nghệ thao tác điện thoại nhận chuyến khi xe đang chạy có vi phạm không?",
    "Hàng hóa ký gửi theo xe khách là gì, hàng hôi thối có được nhận không?",
    "Tôi muốn cấp đổi GPLX thì cần hồ sơ gì và thời hạn giải quyết bao lâu?",
    "Theo QCVN 41, gặp đèn đỏ hoặc đèn vàng thì phải dừng ở đâu nếu có vạch dừng?",
    "Biển cố định cho phép rẽ phải nhưng biển tạm thời ở công trường lại cấm rẽ phải thì theo biển nào?",
]
SUGGESTED_QUESTION_COUNT = 6


def ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("queued_question", "")
    st.session_state.setdefault("app_mode", "Trò chuyện AI")
    st.session_state.setdefault("inspector_query", "")
    st.session_state.setdefault("saved_cases", [])
    st.session_state.setdefault("source_search_results", [])
    st.session_state.setdefault("source_graph_trace", None)
    if "suggested_questions" not in st.session_state:
        sample_size = min(SUGGESTED_QUESTION_COUNT, len(QUESTION_BANK))
        st.session_state.suggested_questions = random.sample(QUESTION_BANK, sample_size)


def api_post(path: str, *, data: dict[str, Any] | None = None, files: dict[str, Any] | None = None, timeout: int = 600) -> requests.Response:
    return requests.post(f"{API_URL}{path}", data=data or {}, files=files, timeout=timeout)


def api_get(path: str, *, params: dict[str, Any] | None = None, timeout: int = 600) -> requests.Response:
    return requests.get(f"{API_URL}{path}", params=params or {}, timeout=timeout)


def image_url(path: str) -> str:
    return image_source(path, api_url=API_URL, processed_dir=PROCESSED_DIR)


def html_escape(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def ascii_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def render_html(markup: str) -> None:
    compact_markup = "".join(line.strip() for line in markup.splitlines())
    st.markdown(compact_markup, unsafe_allow_html=True)


def source_file_text(source: dict[str, Any] | None) -> str:
    if not source:
        return ""
    files = source.get("raw_files") or source.get("processed_files") or []
    return ", ".join(str(item) for item in files if item)


def source_card_markup(source: dict[str, Any]) -> str:
    title = source.get("name") or "Văn bản pháp luật"
    number = source.get("document_number") or ""
    files = source_file_text(source)
    url = source.get("source_url") or ""
    note = source.get("note") or ""
    url_markup = (
        f' · <a href="{html_escape(url)}" target="_blank" rel="noopener noreferrer">Văn bản gốc</a>'
        if url else ""
    )
    note_markup = f'<div class="small-note">{html_escape(note)}</div>' if note else ""
    return f"""
        <div class="source-card">
            <div class="source-title">{html_escape(title)}</div>
            <div class="source-meta">{html_escape(number)} · Tệp: {html_escape(files)}{url_markup}</div>
            {note_markup}
        </div>
    """


def render_document_sources(sources: list[dict[str, Any]] | None = None, *, compact: bool = False) -> None:
    sources = sources or source_catalog_payload()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        grouped.setdefault(str(source.get("group") or "Nguồn tài liệu"), []).append(source)
    for group, items in grouped.items():
        st.caption(group)
        for source in items[:3 if compact else len(items)]:
            render_html(source_card_markup(source))


def document_source_for_reference(ref: dict[str, Any]) -> dict[str, Any] | None:
    source = ref.get("document_source")
    if isinstance(source, dict) and source.get("name"):
        return source
    legal_ref = ref.get("legal_reference") or {}
    return source_info_for_document(ref.get("doc_name") or legal_ref.get("document"))


def render_header() -> None:
    render_html(
        """
        <div class="app-header">
            <div class="app-title">Trợ lý Luật Giao thông Việt Nam</div>
            <div class="app-subtitle">
                Hỏi về mức phạt, biển báo, giấy phép lái xe, thủ tục và tình huống giao thông đường bộ.
                Câu trả lời luôn kèm căn cứ để bạn tự đối chiếu.
            </div>
        </div>
        """
    )


def render_stepper(active_index: int, done_until: int = -1) -> None:
    rows = []
    for idx, (title, note) in enumerate(PROCESSING_STEPS):
        if idx <= done_until:
            cls = "step-row step-done"
            marker = "✓"
        elif idx == active_index:
            cls = "step-row step-active"
            marker = str(idx + 1)
        else:
            cls = "step-row step-wait"
            marker = str(idx + 1)
        rows.append(
            f"""
            <div class="{cls}">
                <div class="step-dot">{marker}</div>
                <div>
                    <div class="step-title">{html_escape(title)}</div>
                    <div class="step-note">{html_escape(note)}</div>
                </div>
            </div>
            """
        )
    render_html(f'<div class="stepper">{"".join(rows)}</div>')


def render_vision_summary(vision: dict[str, Any] | None) -> None:
    rows = vision_display_rows(vision)
    if not rows:
        return

    trusted_codes = vision.get("trusted_codes") if vision else []
    if isinstance(trusted_codes, list):
        trusted_codes = ", ".join(str(code) for code in trusted_codes if code)
    candidate_codes = vision.get("candidate_codes") if vision else []
    if isinstance(candidate_codes, list):
        candidate_codes = ", ".join(str(code) for code in candidate_codes if code)
    alternatives = vision.get("alternatives") if vision else []
    alt_text = ""
    if isinstance(alternatives, list) and alternatives:
        parts = []
        for item in alternatives[:3]:
            if not isinstance(item, dict):
                continue
            code = item.get("code") or item.get("name") or ""
            reason = item.get("reason") or ""
            parts.append(f"{code}: {reason}".strip(": "))
        if parts:
            alt_text = " | ".join(parts)

    raw_description = str((vision or {}).get("raw_description") or "").strip()
    status_text = "Ảnh có vẻ là biển báo giao thông" if vision.get("is_traffic_sign", True) else "Ảnh chưa đủ tin cậy để xác nhận là biển báo"

    items_html = []
    for label, value in rows:
        items_html.append(
            f"""
            <div class="vision-item">
                <div class="vision-label">{html_escape(label)}</div>
                <div class="vision-value">{html_escape(value)}</div>
            </div>
            """
        )

    extra_bits = []
    if trusted_codes:
        extra_bits.append(f"Mã tin cậy: {html_escape(trusted_codes)}")
    if candidate_codes:
        extra_bits.append(f"Mã dự đoán: {html_escape(candidate_codes)}")
    if alt_text:
        extra_bits.append(f"Phương án khác: {html_escape(alt_text)}")
    if raw_description:
        extra_bits.append(f"Mô tả ảnh: {html_escape(raw_description)}")

    render_html(
        f"""
        <div class="vision-panel">
            <div class="source-title">Phân tích ảnh</div>
            <div class="source-meta">{html_escape(status_text)}</div>
            <div class="vision-grid">
                {''.join(items_html)}
            </div>
            {f'<div class="small-note" style="margin-top:10px;">{" · ".join(extra_bits)}</div>' if extra_bits else ''}
        </div>
        """
    )


def render_answer_markdown(answer: str) -> None:
    sections = split_markdown_sections(answer)
    if not sections:
        st.markdown(answer or "Không có câu trả lời.")
        return

    render_html('<div class="answer-title">Kết quả trả lời</div>')

    if len(sections) == 1 and sections[0]["title"] == "Mở đầu":
        st.markdown(sections[0]["body"] or answer)
        return

    for idx, section in enumerate(sections):
        title = section.get("title") or ""
        body = section.get("body") or ""
        section_class = "answer-section" if idx else "answer-section"
        render_html(
            f"""
            <div class="{section_class}">
                <div class="answer-section-title">{html_escape(title)}</div>
            </div>
            """
        )
        if body:
            st.markdown(body)


def render_analysis(analysis: dict[str, Any] | None, *, compact: bool = False) -> None:
    if not analysis:
        return
    budget = analysis.get("retrieval_budget") or {}
    plan = analysis.get("plan") or {}
    slots = analysis.get("evidence_slots") or plan.get("subquestions") or []
    difficulty = analysis.get("difficulty_label") or analysis.get("difficulty") or "Không rõ"
    facets = ", ".join(analysis.get("facets") or []) or plan.get("intent") or "general"
    wait = analysis.get("max_wait_seconds") or "?"
    contexts = budget.get("max_contexts") or budget.get("top_k") or "?"
    images = budget.get("max_images") or "?"

    render_html(
        f"""
        <div class="info-grid">
            <div class="metric-card">
                <div class="metric-label">Độ khó</div>
                <div class="metric-value">{html_escape(difficulty)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Nhánh xử lý</div>
                <div class="metric-value">{html_escape(facets)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Ngân sách căn cứ</div>
                <div class="metric-value">{html_escape(contexts)} nguồn / {html_escape(images)} ảnh</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Thời gian dự kiến</div>
                <div class="metric-value">Tối đa {html_escape(wait)} giây</div>
            </div>
        </div>
        """
    )

    if compact or not slots:
        return

    with st.expander("Kế hoạch truy vấn tuần tự", expanded=True):
        for idx, slot in enumerate(slots[:12], start=1):
            facet = slot.get("facet") or "general"
            query = slot.get("query") or ""
            reason = slot.get("reason") or ""
            render_html(
                f"""
                <div class="source-card">
                    <div class="source-title">{idx}. {html_escape(facet)}</div>
                    <div class="source-meta">{html_escape(reason)}</div>
                    <div class="small-note">{html_escape(query)}</div>
                </div>
                """
            )


def render_reference_gallery(images: list[str], *, max_visible: int = 12) -> None:
    if not images:
        return
    with st.expander(f"Ảnh căn cứ từ văn bản gốc ({len(images)})", expanded=False):
        cols = st.columns(3)
        for idx, img_path in enumerate(images[:max_visible]):
            caption = img_path.rsplit("/", 1)[-1]
            with cols[idx % 3]:
                st.image(image_url(img_path), caption=caption, use_container_width=True)
        if len(images) > max_visible:
            st.caption(f"Còn {len(images) - max_visible} ảnh căn cứ khác.")


def reference_label(ref: dict[str, Any]) -> str:
    if ref.get("reference_text"):
        return str(ref["reference_text"])
    legal_ref = ref.get("legal_reference") or {}
    parts = []
    if legal_ref.get("point"):
        parts.append(f"Điểm {legal_ref.get('point')}")
    if legal_ref.get("clause"):
        parts.append(f"Khoản {legal_ref.get('clause')}")
    if legal_ref.get("article"):
        parts.append(f"Điều {legal_ref.get('article')}")
    if legal_ref.get("document"):
        parts.append(str(legal_ref.get("document")))
    return ", ".join(parts) or ref.get("source_chunk_id") or "Nguồn dữ liệu"


def render_references(references: list[dict[str, Any]]) -> None:
    if not references:
        return
    with st.expander(f"Căn cứ pháp lý ({len(references)})", expanded=True):
        for idx, ref in enumerate(references[:12], start=1):
            reasons = ", ".join(ref.get("retrieval_reasons") or [])
            score = ref.get("retrieval_score")
            score_text = f"score={score:.3f}" if isinstance(score, (int, float)) else ""
            images = ref.get("images") or ([ref.get("image")] if ref.get("image") else [])
            image_text = f"{len(images)} ảnh" if images else "không có ảnh"
            document_source = document_source_for_reference(ref)
            source_files = source_file_text(document_source)
            source_url = (document_source or {}).get("source_url") or ""
            detail_text = f"{ref.get('modality') or 'text'} · {image_text}"
            if source_files:
                detail_text = f"{detail_text} · tệp {source_files}"
            if SHOW_RETRIEVAL_DETAILS and score_text:
                detail_text = f"{detail_text} · {score_text}"
            source_link = (
                f'<div class="small-note"><a href="{html_escape(source_url)}" target="_blank" rel="noopener noreferrer">Mở văn bản gốc</a></div>'
                if source_url else ""
            )
            render_html(
                f"""
                <div class="source-card">
                    <div class="source-title">{idx}. {html_escape(reference_label(ref))}</div>
                    <div class="source-meta">{html_escape(detail_text)}</div>
                    {source_link}
                    {'<div class="small-note">' + html_escape(reasons) + '</div>' if SHOW_RETRIEVAL_DETAILS and reasons else ''}
                </div>
                """
            )


def reference_source_ids(references: list[dict[str, Any]]) -> list[str]:
    ids = [str(ref.get("source_chunk_id") or "") for ref in references if ref.get("source_chunk_id")]
    return list(dict.fromkeys(ids))


def status_badge(status: str) -> tuple[str, str]:
    if status == "supported":
        return "Có căn cứ mạnh", "status-supported"
    if status == "weak":
        return "Căn cứ yếu", "status-weak"
    return "Cần rà soát", "status-review"


def render_graph_trace(trace: dict[str, Any] | None) -> None:
    if not trace or not trace.get("nodes"):
        st.caption("Chưa có dữ liệu graph để hiển thị.")
        return

    nodes = trace.get("nodes") or []
    edges = trace.get("edges") or []
    relation_counts = trace.get("relation_counts") or {}
    node_type_counts = trace.get("node_type_counts") or {}

    cols = st.columns(4)
    cols[0].metric("Graph nodes", len(nodes))
    cols[1].metric("Graph edges", len(edges))
    cols[2].metric("Loại quan hệ", len(relation_counts))
    cols[3].metric("Backend", trace.get("backend") or "graph")

    if relation_counts:
        relation_text = " · ".join(f"{key}: {value}" for key, value in sorted(relation_counts.items())[:10])
        render_html(f'<div class="small-note">Quan hệ: {html_escape(relation_text)}</div>')
    if node_type_counts:
        node_text = " · ".join(f"{key}: {value}" for key, value in sorted(node_type_counts.items())[:10])
        render_html(f'<div class="small-note">Loại node: {html_escape(node_text)}</div>')

    if edges:
        with st.expander("Sơ đồ liên kết graph", expanded=False):
            limited_nodes = nodes[:38]
            allowed = {node.get("id") for node in limited_nodes}

            def dot_id(value: Any) -> str:
                return "n_" + re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))[:80]

            lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fillcolor="#f7f8fb", color="#d9dee8", fontname="Arial"];']
            for node in limited_nodes:
                label = f"{node.get('type') or 'node'}\\n{str(node.get('label') or node.get('id') or '')[:72]}"
                lines.append(f'{dot_id(node.get("id"))} [label="{html_escape(label)}"];')
            for edge in edges[:70]:
                if edge.get("source") in allowed and edge.get("target") in allowed:
                    label = str(edge.get("type") or "RELATED")[:24]
                    lines.append(f'{dot_id(edge.get("source"))} -> {dot_id(edge.get("target"))} [label="{html_escape(label)}"];')
            lines.append("}")
            st.graphviz_chart("\n".join(lines), use_container_width=True)

    with st.expander("Các node graph liên quan", expanded=False):
        sorted_nodes = sorted(
            nodes,
            key=lambda item: (
                item.get("graph_cost") if item.get("graph_cost") is not None else 999,
                item.get("graph_distance") if item.get("graph_distance") is not None else 999,
                str(item.get("label") or ""),
            ),
        )
        for node in sorted_nodes[:45]:
            bits = [
                str(node.get("type") or "node"),
                f"dist={node.get('graph_distance')}" if node.get("graph_distance") is not None else "",
                f"cost={node.get('graph_cost')}" if node.get("graph_cost") is not None else "",
                str(node.get("graph_via") or ""),
            ]
            meta = " · ".join(bit for bit in bits if bit)
            render_html(
                f"""
                <div class="source-card">
                    <div class="source-title">{html_escape(node.get("label") or node.get("id"))}</div>
                    <div class="source-meta">{html_escape(meta)}</div>
                    <div class="small-note">{html_escape(node.get("id"))}</div>
                </div>
                """
            )


def render_claim_verifier(message: dict[str, Any]) -> None:
    references = message.get("references") or []
    source_ids = reference_source_ids(references)
    if not message.get("content") or not source_ids:
        return

    message_id = message.get("message_id") or "message"
    cache_key = f"claim_verify_{message_id}"
    with st.expander("Kiểm chứng từng kết luận", expanded=False):
        st.caption("Đối chiếu từng dòng kết luận trong câu trả lời với các nguồn đã retrieve. Đây là kiểm chứng lexical, không thay thế rà soát pháp lý cuối cùng.")
        if st.button("Chạy kiểm chứng căn cứ", key=f"run_{cache_key}"):
            try:
                response = api_post(
                    "/chat/verify",
                    data={
                        "answer": message.get("content", ""),
                        "source_chunk_ids": json.dumps(source_ids, ensure_ascii=False),
                    },
                    timeout=35,
                )
                if response.status_code == 200:
                    st.session_state[cache_key] = response.json()
                else:
                    st.error(f"Lỗi backend: {response.text}")
            except Exception as exc:
                st.error(f"Không thể kiểm chứng: {exc}")

        payload = st.session_state.get(cache_key)
        if not payload:
            return

        cols = st.columns(4)
        cols[0].metric("Kết luận", payload.get("claim_count", 0))
        cols[1].metric("Có căn cứ mạnh", payload.get("supported_count", 0))
        cols[2].metric("Căn cứ yếu", payload.get("weak_count", 0))
        cols[3].metric("Cần rà soát", payload.get("needs_review_count", 0))

        for idx, item in enumerate(payload.get("claims") or [], start=1):
            label, cls = status_badge(item.get("status") or "")
            supports = item.get("supports") or []
            support = supports[0] if supports else {}
            render_html(
                f"""
                <div class="source-card">
                    <div class="source-title">{idx}. {html_escape(item.get("claim"))}</div>
                    <div class="source-meta">
                        <span class="status-pill {cls}">{html_escape(label)}</span>
                        score={html_escape(item.get("score"))}
                    </div>
                    <div class="small-note">{html_escape(support.get("reference_text") or "Chưa tìm thấy nguồn khớp đủ mạnh.")}</div>
                </div>
                """
            )
            if support.get("excerpt"):
                st.caption(support["excerpt"])


def exportable_messages() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in st.session_state.messages:
        item = {key: value for key, value in msg.items() if key != "image"}
        out.append(item)
    return out


def case_markdown(messages: list[dict[str, Any]] | None = None) -> str:
    messages = messages or exportable_messages()
    lines = ["# Hồ sơ hỏi đáp luật giao thông", ""]
    for msg in messages:
        role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
        lines.extend([f"## {role}", "", str(msg.get("content") or ""), ""])
        refs = msg.get("references") or []
        if refs:
            lines.append("### Căn cứ")
            for idx, ref in enumerate(refs[:30], start=1):
                lines.append(f"{idx}. {reference_label(ref)}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_case_actions() -> None:
    if not st.session_state.messages:
        return
    messages = exportable_messages()
    markdown_payload = case_markdown(messages)
    col_a, col_b = st.columns([1, 1])
    if col_a.button("Lưu hồ sơ phiên này", use_container_width=True):
        title = next((msg.get("content", "") for msg in messages if msg.get("role") == "user"), "Hồ sơ")
        st.session_state.saved_cases.append({
            "title": title[:80],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
        })
        st.success("Đã lưu hồ sơ trong phiên làm việc.")
    col_b.download_button(
        "Tải hồ sơ Markdown",
        data=markdown_payload,
        file_name="ho-so-luat-giao-thong.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if SHOW_ADVANCED_TOOLS:
        json_payload = json.dumps(messages, ensure_ascii=False, indent=2)
        st.download_button(
            "Tải JSON kỹ thuật",
            data=json_payload,
            file_name="ho-so-luat-giao-thong.json",
            mime="application/json",
            use_container_width=True,
        )


def render_metadata(metadata: dict[str, Any] | None) -> None:
    if not metadata:
        return
    slot_results = metadata.get("slot_results") or []
    if not slot_results:
        return
    with st.expander("Nhật ký truy xuất theo nhánh", expanded=False):
        for item in slot_results:
            slot = item.get("slot") or {}
            status = item.get("status") or "unknown"
            record_count = item.get("record_count", 0)
            images = item.get("images") or []
            render_html(
                f"""
                <div class="source-card">
                    <div class="source-title">{html_escape(slot.get("id"))} · {html_escape(slot.get("facet"))} · {html_escape(status)}</div>
                    <div class="source-meta">{record_count} nguồn · {len(images)} ảnh</div>
                    <div class="small-note">{html_escape(slot.get("reason") or slot.get("query") or "")}</div>
                </div>
                """
            )


def render_answer_trace(trace: dict[str, Any] | None) -> None:
    if not trace:
        return
    verification = trace.get("verification") or {}
    supported = verification.get("supported_count", 0)
    weak = verification.get("weak_count", 0)
    review = verification.get("needs_review_count", 0)
    claim_count = verification.get("claim_count", 0)
    summary = (
        f"{trace.get('slot_count') or 0} nhánh · "
        f"{trace.get('retrieved_context_count') or 0} nguồn · "
        f"{trace.get('reference_count') or 0} căn cứ · "
        f"{claim_count} kết luận đã kiểm chứng"
    )
    render_html(
        f"""
        <div class="action-band">
            <div class="source-title">Truy vết trả lời</div>
            <div class="small-note">{html_escape(summary)}</div>
        </div>
        """
    )
    with st.expander("Chi tiết phân tích, truy xuất và kiểm chứng", expanded=False):
        cols = st.columns(5)
        cols[0].metric("Nhánh", trace.get("slot_count", 0))
        cols[1].metric("Nguồn", trace.get("retrieved_context_count", 0))
        cols[2].metric("Căn cứ", trace.get("reference_count", 0))
        cols[3].metric("Mạnh/yếu", f"{supported}/{weak}")
        cols[4].metric("Cần rà", review)

        for step in trace.get("steps") or []:
            render_html(
                f"""
                <div class="source-card">
                    <div class="source-title">{html_escape(step.get("name") or "")}</div>
                    <div class="small-note">{html_escape(step.get("summary") or "")}</div>
                </div>
                """
            )

        coverage = trace.get("coverage") or []
        if coverage:
            st.caption("Bao phủ từng nhánh truy vấn")
            for item in coverage[:16]:
                status = item.get("status") or "unknown"
                record_count = item.get("record_count", 0)
                image_count = item.get("image_count", 0)
                render_html(
                    f"""
                    <div class="source-card">
                        <div class="source-title">{html_escape(item.get("slot_id") or "")} · {html_escape(item.get("facet") or "general")} · {html_escape(status)}</div>
                        <div class="source-meta">{record_count} nguồn · {image_count} ảnh</div>
                        <div class="small-note">{html_escape(item.get("reason") or item.get("query") or "")}</div>
                    </div>
                    """
                )

        details = trace.get("verification_details") or []
        if details:
            st.caption("Một số kết luận đã đối chiếu nguồn")
            for idx, item in enumerate(details[:8], start=1):
                label, cls = status_badge(item.get("status") or "")
                render_html(
                    f"""
                    <div class="source-card">
                        <div class="source-title">{idx}. {html_escape(item.get("claim") or "")}</div>
                        <div class="source-meta">
                            <span class="status-pill {cls}">{html_escape(label)}</span>
                            score={html_escape(item.get("score"))}
                        </div>
                    </div>
                    """
                )


def query_gaps(question: str, analysis: dict[str, Any] | None = None) -> list[str]:
    qa = ascii_lower(question)
    facets = set((analysis or {}).get("facets") or [])
    gaps: list[str] = [
        str(item).strip()
        for item in ((analysis or {}).get("missing_data_hints") or [])
        if str(item or "").strip()
    ]
    vehicle_terms = ["o to", "xe hoi", "xe con", "xe tai", "xe khach", "xe may", "mo to", "gan may", "may chuyen dung", "xe dap", "tho so"]
    if "out_of_scope" in facets:
        gaps.append("Câu hỏi hiện ngoài phạm vi dữ liệu luật giao thông đường bộ; hãy hỏi lại về quy tắc, xử phạt, biển báo, thủ tục hoặc điều luật giao thông.")
        return gaps
    if "aggregation" in facets and not any(term in qa for term in ["nghi dinh", "luat", "thong tu", "qcvn", "168", "336", "35/2024", "36/2024", "51/2024"]):
        gaps.append("Phạm vi thống kê: nên nêu rõ văn bản hoặc nhóm hành vi, ví dụ Nghị định 168/2024/NĐ-CP hoặc xử phạt quá tốc độ.")
    if "aggregation" in facets and any(term in qa for term in ["hay vi pham", "pho bien", "thuong gap"]):
        gaps.append("Hệ thống chỉ có dữ liệu văn bản pháp luật; nếu muốn 'hay vi phạm nhất' ngoài thực tế cần thêm dataset số vụ/biên bản xử phạt.")
    penalty_like = "penalty" in facets or any(term in qa for term in ["phat", "xu phat", "muc phat", "vi pham", "bi gi", "xu ly"])
    if penalty_like and not any(term in qa for term in vehicle_terms):
        gaps.append("Loại phương tiện: ô tô, mô tô/xe gắn máy, xe máy chuyên dùng, xe đạp/xe thô sơ.")
    if any(term in qa for term in ["toc do", "qua toc", "p127", "p.127"]) and not re.search(r"\d+(?:[.,]\d+)?\s*km/?h", qa):
        gaps.append("Tốc độ thực tế, tốc độ ghi trên biển/đoạn đường và bối cảnh đường ngoài đô thị/khu đông dân cư/cao tốc.")
    if any(term in qa for term in ["nong do con", "hoi con", "ruou bia", "say xin"]) and not re.search(r"\d+(?:[.,]\d+)?", qa):
        gaps.append("Ngưỡng nồng độ cồn trong máu hoặc hơi thở nếu muốn chốt đúng mức cao/thấp.")
    if any(term in qa for term in ["tai nan", "gay tai nan"]):
        gaps.append("Hậu quả tai nạn: thương tích, thiệt hại tài sản, có rời hiện trường hay cứu giúp người bị nạn không.")
    if ("sign" in facets or "bien" in qa) and not re.search(r"\b(?:dp|ie|p|w|r|i|s|e)\s*\\.?\s*\d{2,3}", qa):
        gaps.append("Mã biển báo hoặc ảnh biển báo rõ phần hình dạng, màu sắc, ký hiệu/chữ số.")
    deduped: list[str] = []
    seen = set()
    for gap in gaps:
        key = ascii_lower(gap)
        if key and key not in seen:
            seen.add(key)
            deduped.append(gap)
    return deduped


def render_gap_list(gaps: list[str]) -> None:
    if not gaps:
        st.success("Câu hỏi đã có đủ dữ kiện chính để truy vấn.")
        return
    items = "".join(f"<li>{html_escape(gap)}</li>" for gap in gaps)
    render_html(
        f"""
        <div class="gap-list">
            <div class="source-title">Dữ kiện nên bổ sung để câu trả lời chính xác hơn</div>
            <ul>{items}</ul>
        </div>
        """
    )


def render_query_inspector() -> None:
    render_header()
    render_html(
        """
        <div class="action-band">
            <div class="source-title">Kiểm tra truy vấn trước khi hỏi</div>
            <div class="small-note">
                Xem độ khó, các nhánh pháp lý sẽ được truy xuất và những dữ kiện còn thiếu trong câu hỏi.
            </div>
        </div>
        """
    )

    sample_options = [""] + list(st.session_state.get("suggested_questions") or QUESTION_BANK[:SUGGESTED_QUESTION_COUNT])
    selected_sample = st.selectbox("Câu hỏi mẫu", sample_options, format_func=lambda x: "Chọn câu hỏi mẫu..." if not x else x)
    if selected_sample:
        st.session_state.inspector_query = selected_sample

    question = st.text_area(
        "Câu hỏi cần kiểm tra",
        key="inspector_query",
        height=130,
        placeholder="Ví dụ: Biển P.127 là gì, chạy xe vi phạm thì bị phạt sao?",
    )

    col_a, col_b = st.columns([1, 1])
    analyze = col_a.button("Phân tích truy vấn", type="primary", use_container_width=True, disabled=not question.strip())
    send_to_chat = col_b.button("Đưa sang chat", use_container_width=True, disabled=not question.strip())

    if send_to_chat and question.strip():
        st.session_state.queued_question = question.strip()
        st.session_state.app_mode = "Trò chuyện AI"
        st.rerun()

    if not analyze:
        st.caption("Màn này không gọi sinh câu trả lời, chỉ kiểm tra kế hoạch truy vấn.")
        return

    try:
        response = api_post("/chat/analyze", data={"query": question.strip(), "history": "[]"}, timeout=20)
    except Exception as exc:
        st.error(f"Không thể kết nối backend: {exc}")
        return
    if response.status_code != 200:
        st.error(f"Lỗi backend: {response.text}")
        return

    payload = response.json()
    analysis = payload.get("analysis") or {}
    render_analysis(analysis, compact=False)
    render_gap_list(query_gaps(question, analysis))

    plan = analysis.get("plan") or {}
    slots = analysis.get("evidence_slots") or plan.get("subquestions") or []
    if slots:
        must_answer = sum(1 for slot in slots if slot.get("must_answer", True))
        st.metric("Nhánh bắt buộc trả lời", must_answer)
    with st.expander("JSON phân tích thô", expanded=False):
        st.json(payload)


def fetch_system_status() -> dict[str, Any] | None:
    try:
        response = api_get("/system/status", timeout=25)
    except Exception as exc:
        st.error(f"Không thể kết nối backend: {exc}")
        return None
    if response.status_code != 200:
        st.error(f"Lỗi backend: {response.text}")
        return None
    return response.json()


def render_source_result(result: dict[str, Any], idx: int) -> None:
    score = result.get("retrieval_score")
    score_text = f"score={score:.3f}" if isinstance(score, (int, float)) else ""
    ref = reference_label(result)
    document_source = document_source_for_reference(result)
    source_files = source_file_text(document_source)
    source_url = (document_source or {}).get("source_url") or ""
    meta_bits = [
        result.get("modality") or "text",
        f"tệp {source_files}" if source_files else "",
        f"trang {result.get('page_start')}-{result.get('page_end')}" if result.get("page_start") is not None else "",
        score_text,
    ]
    source_link = (
        f'<div class="small-note"><a href="{html_escape(source_url)}" target="_blank" rel="noopener noreferrer">Mở văn bản gốc</a></div>'
        if source_url else ""
    )
    render_html(
        f"""
        <div class="source-card">
            <div class="source-title">{idx}. {html_escape(ref)}</div>
            <div class="source-meta">{html_escape(" · ".join(bit for bit in meta_bits if bit))}</div>
            {source_link}
            <div class="small-note">{html_escape(result.get("excerpt") or "")}</div>
        </div>
        """
    )
    images = result.get("images") or []
    if images:
        with st.expander(f"Ảnh nguồn {idx}", expanded=False):
            cols = st.columns(3)
            for image_idx, path in enumerate(images[:6]):
                with cols[image_idx % 3]:
                    st.image(image_url(path), caption=path.rsplit("/", 1)[-1], use_container_width=True)


def render_source_explorer() -> None:
    render_header()
    render_html(
        """
        <div class="action-band">
            <div class="source-title">Nguồn pháp lý và graph</div>
            <div class="small-note">
                Lọc văn bản, tra cứu nguồn gốc và xem graph liên kết giữa điều/khoản/điểm, bảng, hình và citation.
            </div>
        </div>
        """
    )

    status = fetch_system_status()
    documents = [""] + [item.get("name", "") for item in (status or {}).get("documents", []) if item.get("name")]
    modalities = [""] + [item.get("name", "") for item in (status or {}).get("modalities", []) if item.get("name")]

    with st.expander("Danh mục tài liệu đang truy xuất", expanded=True):
        render_document_sources((status or {}).get("document_sources") or source_catalog_payload())

    with st.container(border=True):
        col_q, col_doc = st.columns([1.2, 1])
        query = col_q.text_input("Từ khóa nguồn", key="source_query", placeholder="Ví dụ: P.127, vượt quá tốc độ, nồng độ cồn...")
        document = col_doc.selectbox("Văn bản", documents, format_func=lambda x: "Tất cả văn bản" if not x else x)

        col_a, col_b, col_c, col_d = st.columns(4)
        article = col_a.text_input("Điều/Phụ lục", key="source_article", placeholder="6, 7, Phụ lục B")
        clause = col_b.text_input("Khoản", key="source_clause", placeholder="5")
        point = col_c.text_input("Điểm", key="source_point", placeholder="a")
        modality = col_d.selectbox("Loại nguồn", modalities, format_func=lambda x: "Tất cả loại" if not x else x)

        flag_cols = st.columns(5)
        has_penalty = flag_cols[0].checkbox("Có chế tài")
        has_sign = flag_cols[1].checkbox("Có biển báo")
        has_table = flag_cols[2].checkbox("Có bảng")
        has_procedure = flag_cols[3].checkbox("Có thủ tục")
        limit = flag_cols[4].slider("Số nguồn", min_value=5, max_value=80, value=30, step=5)

        if st.button("Tìm nguồn", type="primary", use_container_width=True):
            params = {
                "q": query,
                "document": document,
                "article": article,
                "clause": clause,
                "point": point,
                "modality": modality,
                "has_penalty": has_penalty,
                "has_sign": has_sign,
                "has_table": has_table,
                "has_procedure": has_procedure,
                "limit": limit,
            }
            try:
                response = api_get("/sources/search", params=params, timeout=45)
            except Exception as exc:
                st.error(f"Không thể tìm nguồn: {exc}")
            else:
                if response.status_code == 200:
                    st.session_state.source_search_results = response.json().get("results") or []
                    st.session_state.source_graph_trace = None
                else:
                    st.error(f"Lỗi backend: {response.text}")

    results = st.session_state.get("source_search_results") or []
    if results:
        st.subheader(f"Kết quả nguồn ({len(results)})")
        labels = [
            f"{idx + 1}. {reference_label(result)}"
            for idx, result in enumerate(results)
        ]
        selected_idx = st.selectbox("Nguồn để trace graph", range(len(results)), format_func=lambda idx: labels[idx])
        if st.button("Trace graph từ nguồn đang chọn", use_container_width=True):
            source_id = results[selected_idx].get("source_chunk_id") or ""
            try:
                response = api_get("/graph/trace", params={"source_chunk_ids": source_id, "depth": 4, "limit": 100}, timeout=45)
            except Exception as exc:
                st.error(f"Không thể trace graph: {exc}")
            else:
                if response.status_code == 200:
                    st.session_state.source_graph_trace = response.json().get("trace")
                else:
                    st.error(f"Lỗi backend: {response.text}")

        if st.session_state.get("source_graph_trace"):
            with st.expander("Trace graph từ nguồn", expanded=True):
                render_graph_trace(st.session_state.source_graph_trace)

        for idx, result in enumerate(results, start=1):
            render_source_result(result, idx)
    else:
        st.info("Chưa có kết quả. Nhập từ khóa hoặc bộ lọc rồi bấm tìm nguồn.")

    st.divider()
    st.subheader("Trace graph trực tiếp từ truy vấn")
    graph_query = st.text_input("Truy vấn graph", key="graph_query", placeholder="Ví dụ: P.127 vượt tốc độ ô tô xe máy")
    if st.button("Trace graph theo truy vấn", use_container_width=True, disabled=not graph_query.strip()):
        try:
            response = api_get("/graph/trace", params={"query": graph_query, "depth": 4, "limit": 100}, timeout=60)
        except Exception as exc:
            st.error(f"Không thể trace graph: {exc}")
        else:
            if response.status_code == 200:
                payload = response.json()
                render_graph_trace(payload.get("trace"))
            else:
                st.error(f"Lỗi backend: {response.text}")


def render_status_page() -> None:
    render_header()
    status = fetch_system_status()
    if not status:
        return

    graph = status.get("graph") or {}
    render_html(
        """
        <div class="action-band">
            <div class="source-title">Trạng thái dữ liệu và index</div>
            <div class="small-note">
                Theo dõi backend, vector store, graph store và độ phủ nguồn đã nạp.
            </div>
        </div>
        """
    )

    cols = st.columns(5)
    cols[0].metric("API", status.get("status", "unknown"))
    cols[1].metric("Records", status.get("vector_record_count", 0))
    cols[2].metric("Graph nodes", graph.get("node_count", 0))
    cols[3].metric("Graph edges", graph.get("edge_count", 0))
    cols[4].metric("RAG loaded", "Có" if status.get("rag_loaded") else "Chưa")

    qdrant_status = "đang dùng Qdrant" if status.get("using_qdrant") else "không dùng Qdrant"
    openvino_status = "đang dùng OpenVINO" if status.get("using_openvino") else "không dùng OpenVINO"
    st.caption(
        f"Vector backend: {status.get('vector_backend')} ({qdrant_status}) · "
        f"Graph backend: {status.get('graph_backend')} · Collection: {status.get('qdrant_collection')}"
    )
    st.caption(
        f"Embedding: {status.get('embedding_runtime') or status.get('embedding_backend')} ({openvino_status}) · "
        f"Model: {status.get('embedding_model')} · Dimension: {status.get('embedding_dimension')} · "
        f"OpenVINO dir: {status.get('openvino_model_dir')}"
    )
    if graph.get("path"):
        st.caption(f"Graph file: {graph.get('path')}")

    tab_docs, tab_graph, tab_raw = st.tabs(["Văn bản", "Graph", "JSON"])
    with tab_docs:
        st.subheader("Nguồn tài liệu")
        st.dataframe(status.get("document_sources") or source_catalog_payload(), use_container_width=True, hide_index=True)
        st.subheader("Độ phủ corpus")
        st.dataframe(status.get("documents") or [], use_container_width=True, hide_index=True)
        st.dataframe(status.get("modalities") or [], use_container_width=True, hide_index=True)
    with tab_graph:
        st.dataframe(graph.get("node_types") or [], use_container_width=True, hide_index=True)
        st.dataframe(graph.get("edge_types") or [], use_container_width=True, hide_index=True)
    with tab_raw:
        st.json(status)


def render_message_extras(message: dict[str, Any], *, compact_analysis: bool = True) -> None:
    if SHOW_RETRIEVAL_DETAILS:
        render_analysis(message.get("query_analysis"), compact=compact_analysis)
    if SHOW_RETRIEVAL_DETAILS and message.get("vision"):
        with st.expander("Kết quả nhận diện ảnh", expanded=False):
            st.json(message["vision"])
    if SHOW_RETRIEVAL_DETAILS:
        render_metadata(message.get("metadata"))
    if SHOW_RETRIEVAL_DETAILS and message.get("graph_trace"):
        with st.expander("Đường dẫn graph và liên kết căn cứ", expanded=False):
            render_graph_trace(message.get("graph_trace"))
    if SHOW_RETRIEVAL_DETAILS:
        render_claim_verifier(message)
    render_answer_trace(message.get("answer_trace"))
    render_reference_gallery(message.get("reference_images") or [])
    render_references(message.get("references") or [])


def message_history_payload() -> str:
    payload = [
        {"role": msg.get("role", "user"), "content": msg.get("content", "")}
        for msg in st.session_state.messages[-6:]
        if msg.get("content")
    ]
    return json.dumps(payload, ensure_ascii=False)


def render_sidebar() -> tuple[str, Any, bool, bool]:
    with st.sidebar:
        st.markdown("### Luật giao thông")
        if SHOW_ADVANCED_TOOLS:
            app_mode = st.radio(
                "Chọn màn hình",
                ["Trò chuyện AI", "Kiểm tra truy vấn", "Nguồn & graph", "Trạng thái dữ liệu"],
                key="app_mode",
                label_visibility="collapsed",
            )
        else:
            app_mode = "Trò chuyện AI"
            st.session_state.app_mode = app_mode
            st.caption("Nhập câu hỏi hoặc tải ảnh biển báo để tra cứu.")
        st.divider()

        clear_chat = st.button("Xóa cuộc trò chuyện", use_container_width=True)
        if SHOW_ADVANCED_TOOLS:
            with st.expander("Kết nối backend", expanded=False):
                st.caption(f"Backend: `{API_URL}`")
                if st.button("Kiểm tra backend", use_container_width=True):
                    try:
                        health = requests.get(f"{API_URL}/health", timeout=4).json()
                        st.success(f"Backend sẵn sàng: {health.get('status')}")
                    except Exception as exc:
                        st.error(f"Chưa kết nối được backend: {exc}")

        with st.expander("Nguồn tài liệu", expanded=False):
            render_document_sources(compact=True)

        if st.session_state.get("saved_cases"):
            st.divider()
            st.markdown("### Phiên đã lưu")
            case_options = [
                f"{idx + 1}. {case.get('saved_at')} · {case.get('title')}"
                for idx, case in enumerate(st.session_state.saved_cases)
            ]
            selected_case = st.selectbox("Chọn hồ sơ", range(len(case_options)), format_func=lambda idx: case_options[idx], label_visibility="collapsed")
            if st.button("Mở hồ sơ", use_container_width=True):
                st.session_state.messages = st.session_state.saved_cases[selected_case].get("messages") or []
                st.session_state.app_mode = "Trò chuyện AI"
                st.rerun()

        st.divider()
        st.markdown("### Ảnh biển báo")
        uploaded_file = st.file_uploader("Tải ảnh biển báo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Ảnh đã tải lên", use_container_width=True)
        analyze_image = st.button("Tra cứu biển báo trong ảnh", disabled=uploaded_file is None, use_container_width=True)

        st.divider()
        st.markdown("### Gợi ý câu hỏi")
        for idx, sample in enumerate(st.session_state.get("suggested_questions") or QUESTION_BANK[:SUGGESTED_QUESTION_COUNT], start=1):
            if st.button(sample, key=f"sample_{idx}", use_container_width=True):
                st.session_state.queued_question = sample

    return app_mode, uploaded_file, analyze_image, clear_chat


def render_chat_history() -> None:
    if not st.session_state.messages:
        render_html(
            """
            <div class="welcome-panel">
                <div class="welcome-title">Bạn cần tra cứu tình huống nào?</div>
                <div class="welcome-note">
                    Hãy hỏi bằng ngôn ngữ tự nhiên, ví dụ mức phạt khi vượt đèn đỏ,
                    ý nghĩa biển báo, thủ tục giấy phép lái xe hoặc một tình huống vi phạm cụ thể.
                </div>
            </div>
            """
        )
        return

    for idx, message in enumerate(st.session_state.messages):
        message.setdefault("message_id", f"{message.get('role', 'msg')}_{idx}")
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_markdown(message["content"])
                render_vision_summary(message.get("vision"))
            else:
                st.markdown(message["content"])
            if message.get("image") is not None:
                st.image(message["image"], caption="Ảnh bạn đã gửi", width=320)
            if message["role"] == "assistant":
                render_message_extras(message)


def run_chat_request(question: str, uploaded_file: Any | None) -> None:
    history_payload = message_history_payload()
    user_msg = {"role": "user", "content": question, "image": None, "message_id": f"u_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"}
    image_bytes = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        try:
            user_msg["image"] = Image.open(io.BytesIO(image_bytes))
        except Exception:
            user_msg["image"] = None

    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(question)
        if user_msg["image"] is not None:
            st.image(user_msg["image"], caption="Ảnh bạn đã gửi", width=320)

    with st.chat_message("assistant"):
        step_box = st.empty()
        analysis_box = st.container()
        result_box = st.container()

        query_analysis = None
        with step_box.container():
            render_stepper(active_index=0)

        if ENABLE_PRE_ANALYSIS and uploaded_file is None:
            try:
                analysis_response = api_post(
                    "/chat/analyze",
                    data={"query": question, "history": history_payload},
                    timeout=14,
                )
                if analysis_response.status_code == 200:
                    query_analysis = analysis_response.json().get("analysis")
            except Exception:
                query_analysis = None

        with step_box.container():
            render_stepper(active_index=1, done_until=0)

        wait_seconds = int((query_analysis or {}).get("max_wait_seconds") or 90)
        request_timeout = min(CHAT_REQUEST_TIMEOUT_SECONDS, max(330, wait_seconds + 180))

        with step_box.container():
            render_stepper(active_index=2, done_until=1)
            st.progress(0.72)
            timer_placeholder = st.empty()
            st.caption("Đang tra cứu căn cứ pháp luật liên quan...")

        response_container = {"response": None, "error": None}

        def make_request():
            try:
                if uploaded_file is not None and image_bytes is not None:
                    files = {"image": (uploaded_file.name, image_bytes, uploaded_file.type)}
                    response_container["response"] = api_post("/chat/image", files=files, data={"query": question}, timeout=request_timeout)
                else:
                    response_container["response"] = api_post(
                        "/chat/text",
                        data={"query": question, "history": history_payload},
                        timeout=request_timeout,
                    )
            except Exception as exc:
                response_container["error"] = exc

        # Run request in background thread to allow live UI updates
        req_thread = threading.Thread(target=make_request)
        req_thread.start()

        start_time = time.time()
        while req_thread.is_alive():
            elapsed = int(time.time() - start_time)
            timer_placeholder.info(f"Đang xử lý... {elapsed} giây")
            time.sleep(1)
        
        response = response_container["response"]
        exc = response_container["error"]

        if exc:
            with step_box.container():
                render_stepper(active_index=2, done_until=1)
            st.error("Backend đang mất quá nhiều thời gian để phản hồi. Vui lòng thử lại với câu hỏi ngắn hơn hoặc chờ vài phút rồi gửi lại.")
            if SHOW_ADVANCED_TOOLS:
                st.caption(str(exc))
            return

        with step_box.container():
            render_stepper(active_index=3, done_until=2)
            st.progress(0.94)
            st.caption("Đang tổng hợp câu trả lời và căn cứ hiển thị...")

        if response.status_code != 200:
            if response.status_code in {408, 504}:
                st.error("Truy vấn này quá nặng cho phiên hiện tại. Hãy tách câu hỏi thành từng hành vi hoặc thử lại sau ít phút.")
            else:
                st.error("Backend chưa trả được câu trả lời cho truy vấn này.")
            if SHOW_ADVANCED_TOOLS:
                st.caption(response.text)
            return

        res_json = response.json()
        answer = res_json.get("answer", "Không có câu trả lời.")
        query_analysis = res_json.get("query_analysis") or query_analysis
        reference_images = res_json.get("reference_images") or res_json.get("images") or []
        references = res_json.get("references") or []
        graph_trace = res_json.get("graph_trace")
        answer_trace = res_json.get("answer_trace")
        metadata = res_json.get("metadata") or {}
        vision = res_json.get("vision")

        with step_box.container():
            render_stepper(active_index=3, done_until=3)
            st.progress(1.0)
            st.caption("Hoàn tất.")

        with result_box:
            render_answer_markdown(answer)
            render_vision_summary(vision)
            render_message_extras(
                {
                    "query_analysis": query_analysis,
                    "metadata": metadata,
                    "reference_images": reference_images,
                    "references": references,
                    "graph_trace": graph_trace,
                    "answer_trace": answer_trace,
                    "vision": vision,
                },
                compact_analysis=True,
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "reference_images": reference_images,
                "references": references,
                "graph_trace": graph_trace,
                "answer_trace": answer_trace,
                "query_analysis": query_analysis,
                "metadata": metadata,
                "vision": vision,
                "message_id": f"a_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            }
        )


ensure_state()
app_mode, uploaded_file, analyze_image, clear_chat = render_sidebar()

if clear_chat:
    st.session_state.messages = []
    st.rerun()

if app_mode == "Kiểm tra truy vấn":
    render_query_inspector()
elif app_mode == "Nguồn & graph":
    render_source_explorer()
elif app_mode == "Trạng thái dữ liệu":
    render_status_page()
else:
    render_header()
    render_case_actions()
    render_chat_history()

    queued_question = st.session_state.pop("queued_question", "")
    prompt = st.chat_input("Hỏi về mức phạt, biển báo, thủ tục, tình huống thực tế...")
    should_run_image = bool(analyze_image and uploaded_file is not None)
    question = queued_question or prompt or ("Giải thích biển báo này giúp tôi." if should_run_image else "")

    if question:
        run_chat_request(question, uploaded_file if should_run_image else None)
