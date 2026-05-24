import streamlit as st
import requests
import json
from PIL import Image

API_URL = "http://localhost:8002"

st.set_page_config(page_title="Luật Giao Thông AI", layout="wide")
st.title("Trợ lý Luật Giao thông")

# Khởi tạo lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("image") is not None:
            st.image(message["image"], caption="Ảnh bạn đã gửi", width=300)
        if message.get("query_analysis"):
            analysis = message["query_analysis"]
            st.caption(
                f"Loại câu hỏi: {analysis.get('difficulty_label', 'Không rõ')} | "
                f"Thời gian chờ tối đa: {analysis.get('max_wait_seconds', '?')} giây"
            )
        if "reference_images" in message and message["reference_images"]:
            st.write("  **Căn cứ hình ảnh từ văn bản gốc:**")
            ref_images = message["reference_images"]
            num_cols = min(4, len(ref_images))
            cols = st.columns(num_cols)
            for idx, img_url in enumerate(ref_images):
                try:
                    # Fetch image bytes from local API
                    img_resp = requests.get(f"{API_URL}{img_url}", timeout=5)
                    if img_resp.status_code == 200:
                        cols[idx % num_cols].image(img_resp.content, use_container_width=True)
                except Exception:
                    continue

# Khu vực nhập liệu
with st.sidebar:
    st.header("Tải lên ảnh biển báo")
    uploaded_file = st.file_uploader("Chọn ảnh biển báo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh bạn đã tải lên", use_column_width=True)
    analyze_image = st.button("Giải thích ảnh đã tải lên", disabled=uploaded_file is None)

    if st.button("Xóa lịch sử Chat"):
        st.session_state.messages = []
        st.rerun()


prompt = st.chat_input("Hỏi về mức phạt, biển báo, thủ tục...")

if prompt or analyze_image:
    question = prompt if prompt else "Giải thích biển báo này giúp tôi."
    history_payload = json.dumps(
        [
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in st.session_state.messages[-6:]
            if msg.get("content")
        ],
        ensure_ascii=False,
    )
    user_msg = {"role": "user", "content": question, "image": None}
    if uploaded_file:
        user_msg["image"] = Image.open(uploaded_file)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(question)
        if user_msg["image"] is not None:
            st.image(user_msg["image"], caption="Ảnh bạn đã gửi", width=300)
    with st.chat_message("assistant"):
        query_analysis = None
        if not uploaded_file:
            try:
                analysis_response = requests.post(
                    f"{API_URL}/chat/analyze",
                    data={"query": question, "history": history_payload},
                    timeout=12,
                )
                if analysis_response.status_code == 200:
                    query_analysis = analysis_response.json().get("analysis")
                    if query_analysis:
                        st.info(
                            f"Loại câu hỏi: {query_analysis.get('difficulty_label', 'Không rõ')} - "
                            f"thời gian chờ tối đa {query_analysis.get('max_wait_seconds', '?')} giây."
                        )
            except Exception:
                query_analysis = None

        wait_hint = ""
        if query_analysis:
            wait_hint = f" Tối đa khoảng {query_analysis.get('max_wait_seconds', '?')} giây."
        with st.spinner(f"Đang tra cứu cơ sở dữ liệu luật...{wait_hint}"):
            try:
                if uploaded_file:
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"query": question}
                    response = requests.post(f"{API_URL}/chat/image", files=files, data=data)
                else:
                    data = {"query": question, "history": history_payload}
                    response = requests.post(f"{API_URL}/chat/text", data=data)

                if response.status_code == 200:
                    res_json = response.json()
                    answer = res_json.get("answer", "Không có câu trả lời.")
                    ref_images = res_json.get("reference_images", [])
                    query_analysis = res_json.get("query_analysis") or query_analysis
                    
                    # Hiển thị mô tả ảnh nếu có (từ tính năng Vision)
                    if "description" in res_json:
                        st.info(f"👁️ **AI nhận diện:** {res_json['description']}")
                    
                    st.markdown(answer)
                    
                    # Hiển thị ảnh tham chiếu từ QCVN/Nghị định
                    if ref_images:
                        st.write("🔍 **Căn cứ hình ảnh từ văn bản gốc:**")
                        num_cols = min(4, len(ref_images))
                        cols = st.columns(num_cols)
                        for idx, img_url in enumerate(ref_images):
                            try:
                                # Fetch image bytes from local API
                                img_resp = requests.get(f"{API_URL}{img_url}", timeout=5)
                                if img_resp.status_code == 200:
                                    cols[idx % num_cols].image(img_resp.content, use_container_width=True)
                            except Exception:
                                continue

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "reference_images": ref_images,
                        "query_analysis": query_analysis,
                    })
                else:
                    st.error(f"Lỗi Backend: {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến Backend: {e}")
