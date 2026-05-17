import streamlit as st
import requests
from PIL import Image

API_URL = "http://localhost:8001"

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
        if "reference_images" in message and message["reference_images"]:
            st.write("  **Căn cứ hình ảnh từ văn bản gốc:**")
            cols = st.columns(len(message["reference_images"]))
            for idx, img_url in enumerate(message["reference_images"]):
                cols[idx].image(f"{API_URL}{img_url}", use_container_width=True)

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
    user_msg = {"role": "user", "content": question, "image": None}
    if uploaded_file:
        user_msg["image"] = Image.open(uploaded_file)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(question)
        if user_msg["image"] is not None:
            st.image(user_msg["image"], caption="Ảnh bạn đã gửi", width=300)
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu cơ sở dữ liệu luật..."):
            try:
                if uploaded_file:
                    files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {"query": question}
                    response = requests.post(f"{API_URL}/chat/image", files=files, data=data)
                else:
                    data = {"query": question}
                    response = requests.post(f"{API_URL}/chat/text", data=data)

                if response.status_code == 200:
                    res_json = response.json()
                    answer = res_json.get("answer", "Không có câu trả lời.")
                    ref_images = res_json.get("reference_images", [])
                    
                    # Hiển thị mô tả ảnh nếu có (từ tính năng Vision)
                    if "description" in res_json:
                        st.info(f"👁️ **AI nhận diện:** {res_json['description']}")
                    
                    st.markdown(answer)
                    
                    # Hiển thị ảnh tham chiếu từ QCVN/Nghị định
                    if ref_images:
                        st.write("🔍 **Căn cứ hình ảnh từ văn bản gốc:**")
                        cols = st.columns(min(3, len(ref_images)))
                        for idx, img_url in enumerate(ref_images[:3]):
                            cols[idx].image(f"{API_URL}{img_url}", use_container_width=True)

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "reference_images": ref_images
                    })
                else:
                    st.error(f"Lỗi Backend: {response.text}")
            except Exception as e:
                st.error(f"Không thể kết nối đến Backend: {e}")
