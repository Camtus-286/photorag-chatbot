"""Giao diện PhotoRAG: ô trên tìm ảnh mới (/chat), ô dưới hỏi tiếp về các ảnh đang hiện (/followup).

Chạy (từ thư mục flickr-search/, API phải đang chạy ở terminal khác):
    python3 -m streamlit run app/ui.py
"""
import requests
import streamlit as st
from PIL import Image, ImageOps

API_URL = "http://127.0.0.1:8000"      # địa chỉ của uvicorn app.api:app
PER_ROW = 5                            # số ảnh trên một hàng


def call_api(path, payload):
    """Gọi API. Trả (data, None) nếu thành công, (None, thông báo lỗi) nếu không."""
    try:
        resp = requests.post(f"{API_URL}{path}", json=payload, timeout=120)
    except requests.exceptions.RequestException:
        return None, "Cannot reach the API. Start it first:  python3 -m uvicorn app.api:app"
    if resp.status_code != 200:        # 400: câu hỏi sai; còn lại: lỗi server
        try:
            return None, str(resp.json().get("detail", resp.text))
        except ValueError:
            return None, resp.text
    return resp.json(), None


st.set_page_config(page_title="PhotoRAG", layout="wide")
st.title("PhotoRAG")

col_q, col_k = st.columns([6, 1])      # ô câu hỏi rộng gấp 6 ô số ảnh, như bản thiết kế
question = col_q.text_input("Question ( English only )", placeholder="e.g. dogs playing in the snow").strip()
k = int(col_k.number_input("Number of photos", min_value=1, max_value=20, value=5, step=1))

if not question:
    st.session_state.pop("search", None)
    st.stop()

# Chỉ tìm lại khi câu hỏi hoặc số ảnh ĐỔI. Nhờ vậy hỏi tiếp ở ô dưới không làm mất ảnh và không gọi /chat thừa.
current = st.session_state.get("search")
if current is None or (current["question"], current["k"]) != (question, k):
    with st.spinner("Searching…"):
        data, error = call_api("/chat", {"question": question, "k": k})
    if error:
        st.session_state.pop("search", None)
        st.error(error)
        st.stop()
    current = {"question": question, "k": k, "data": data, "thread": []}   # thread: các lượt hỏi tiếp về bộ ảnh này
    st.session_state["search"] = current

st.markdown(":gray[**Answer**]")
st.write(current["data"]["answer"])

results = current["data"]["results"]
for start in range(0, len(results), PER_ROW):          # mỗi hàng 5 ảnh, nhiều hơn thì xuống hàng
    cols = st.columns(PER_ROW)
    for col, hit in zip(cols, results[start:start + PER_ROW]):
        img = ImageOps.fit(Image.open(hit["path"]), (420, 300))   # cắt về cùng tỉ lệ 7:5 cho hàng ảnh đều nhau
        col.image(img)
        col.caption(f"**[{hit['rank']}]** {hit['caption']}")
        with col.expander("Show all captions"):                   # bot đọc cả 5 caption này khi trả lời câu hỏi tiếp
            for cap in hit.get("captions", [hit["caption"]]):
                st.markdown(f"- {cap}")
if results:
    for turn in current["thread"]:                     # các lượt hỏi tiếp đã có
        st.markdown(f"**{turn['question']}**")
        st.write(turn["answer"])

    with st.form("followup", clear_on_submit=True, border=False):   # Enter trong ô này = gửi câu hỏi tiếp
        follow = st.text_input("Ask about these photos", placeholder="e.g. the second picture, why are they dressed like that?")
        sent = st.form_submit_button("Ask")

    if sent and follow.strip():
        with st.spinner("Thinking…"):
            data, error = call_api("/followup", {
                "question": follow.strip(),
                "image_ids": [hit["image_id"] for hit in results],
                "history": current["thread"],
            })
        if error:
            st.error(error)
        else:
            current["thread"].append({"question": follow.strip(), "answer": data["answer"]})
            st.rerun()                                 # vẽ lại trang để lượt mới hiện phía trên ô nhập