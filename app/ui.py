"""Giao diện PhotoRAG dạng chat: một ô nhập duy nhất ở đáy, lịch sử các cuộc trò chuyện ở sidebar.

Mỗi câu gửi lên /ask; API tự quyết tìm ảnh mới hay trả lời về các ảnh đang hiện,
và tự đọc số ảnh từ câu hỏi ("show me 3 photos of two girls").
Mỗi cuộc trò chuyện được lưu thành một file JSON trong thư mục chats/, nên tải lại trang không mất.
Chạy (từ thư mục flickr-search/, API phải đang chạy ở terminal khác):
    python3 -m streamlit run app/ui.py
"""
import html
import json
import time
import uuid
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageOps

API_URL = "http://127.0.0.1:8000"      # địa chỉ của uvicorn app.api:app
CHAT_DIR = Path(__file__).resolve().parent.parent / "chats"   # flickr-search/chats/
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


# --- Lưu / mở lịch sử trò chuyện ---
def list_chats():
    """Các cuộc trò chuyện đã lưu, mới nhất lên đầu: list (chat_id, title)."""
    chats = []
    for f in CHAT_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # file hỏng thì bỏ qua, không làm sập trang
            continue
        chats.append((d.get("updated", 0), f.stem, d.get("title", "(untitled)")))
    return [(cid, title) for _, cid, title in sorted(chats, reverse=True)]


def save_chat():
    """Ghi cuộc trò chuyện đang mở ra chats/<chat_id>.json. Tiêu đề = câu hỏi đầu tiên."""
    CHAT_DIR.mkdir(exist_ok=True)
    msgs = st.session_state.messages
    first = next((m["content"] for m in msgs if m["role"] == "user"), "New chat")
    data = {
        "title": first[:40] + ("…" if len(first) > 40 else ""),
        "updated": time.time(),
        "messages": msgs,
        "image_ids": st.session_state.image_ids,
    }
    path = CHAT_DIR / f"{st.session_state.chat_id}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def open_chat(chat_id):
    d = json.loads((CHAT_DIR / f"{chat_id}.json").read_text(encoding="utf-8"))
    st.session_state.chat_id = chat_id
    st.session_state.messages = d["messages"]
    st.session_state.image_ids = d.get("image_ids", [])


def new_chat():
    st.session_state.chat_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    st.session_state.messages = []     # mỗi phần tử: {"role", "content", "route"?, "query"?, "results"?}
    st.session_state.image_ids = []    # bộ ảnh đang được hỏi tới


def show_photos(results):
    """Vẽ hàng ảnh, mỗi hàng PER_ROW ảnh, kèm caption và nút xem đủ 5 caption."""
    for start in range(0, len(results), PER_ROW):
        cols = st.columns(PER_ROW)
        for col, hit in zip(cols, results[start:start + PER_ROW]):
            try:
                img = ImageOps.fit(Image.open(hit["path"]), (420, 300))   # cắt về cùng tỉ lệ 7:5
                col.image(img)
            except OSError:            # chat cũ trỏ tới ảnh đã bị xoá / đổi chỗ
                col.warning("Image not found")
            col.caption(f"**[{hit['rank']}]** {hit['caption']}")
            with col.expander("Show all captions"):
                for cap in hit.get("captions", [hit["caption"]]):
                    st.markdown(f"- {cap}")


def user_bubble(text):
    """Câu của người dùng: bong bóng bên phải, không icon (giống ChatGPT)."""
    st.markdown(
        '<div style="display:flex; justify-content:flex-end; margin:12px 0;">'
        '<div style="max-width:70%; padding:10px 16px; border-radius:18px; '
        'background:rgba(128,128,128,0.18); white-space:pre-wrap;">'
        f"{html.escape(text)}</div></div>",
        unsafe_allow_html=True,
    )


def route_label(m):
    """Dòng chữ nhỏ dưới câu trả lời, cho biết hệ thống đã làm gì."""
    if m.get("route") == "search":
        return f"🔎 searched for “{m.get('query', '')}” · {len(m.get('results') or [])} photos"
    if m.get("route") == "current":
        return "💬 about the photos on screen"
    if m.get("route") == "chat":
        return "💬 chat"
    return ""


st.set_page_config(page_title="PhotoRAG", layout="wide")
if "chat_id" not in st.session_state:
    new_chat()

# --- Sidebar: tạo chat mới + danh sách chat cũ ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        new_chat()
        st.rerun()
    st.caption("History")
    for cid, title in list_chats():
        is_open = cid == st.session_state.chat_id
        if st.button(title, key=f"chat_{cid}", use_container_width=True,
                     type="primary" if is_open else "secondary"):
            open_chat(cid)
            st.rerun()

st.title("PhotoRAG")

# --- Vẽ lại cuộc trò chuyện đang mở ---
for m in st.session_state.messages:
    if m["role"] == "user":
        user_bubble(m["content"])
        continue
    with st.container():                 # câu trả lời của bot: chữ thường bên trái, không icon
        st.write(m["content"])
        if m.get("results"):
            show_photos(m["results"])
        if route_label(m):
            st.caption(route_label(m))

# --- Ô nhập duy nhất, luôn nằm ở đáy ---
question = st.chat_input("e.g. show me 3 photos of two girls  ·  English only")
if question:
    user_bubble(question)

    # Gửi cho API chỉ role + content: lượt cũ không kèm ảnh/caption ("AI chỉ cần đọc kết quả cuối")
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    with st.spinner("Thinking…"):
        data, error = call_api("/ask", {
            "question": question,
            "history": history,
            "image_ids": st.session_state.image_ids,
        })

    if error:
        st.error(error)                # lượt lỗi không được lưu vào lịch sử
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({
            "role": "assistant",
            "content": data["answer"],
            "route": data["route"],
            "query": data.get("query"),
            "results": data.get("results"),
        })
        st.session_state.image_ids = data["image_ids"]
        save_chat()
        st.rerun()                     # vẽ lại trang để lượt mới hiện trong lịch sử
