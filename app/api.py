"""API tìm ảnh: nhận câu query qua HTTP, gọi search.py, trả kết quả dạng JSON.

Chạy (từ thư mục flickr-search/, Qdrant phải đang bật):
    uvicorn app.api:app
Thử nhanh:
    curl -X POST localhost:8000/search -H "Content-Type: application/json" \
         -d '{"query": "dogs playing in the snow", "mode": "image", "k": 5}'
Xem tài liệu API tự sinh: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.search import Searcher
from app.chat import Chatbot

@asynccontextmanager
async def lifespan(app):
    # Nạp CLIP + BM25 + nối Qdrant MỘT lần khi server khởi động (mất vài giây),
    # để mỗi request sau đó chỉ việc tìm, không phải nạp lại model.
    app.state.searcher = Searcher()
    app.state.chatbot = Chatbot(app.state.searcher)   # dùng chung Searcher, không nạp CLIP 2 lần
    yield


app = FastAPI(title="Flickr Search API", lifespan=lifespan)


class SearchRequest(BaseModel):
    """Body của POST /search. Chỉ khai báo kiểu ở đây; kiểm tra chi tiết nằm trong search.py."""
    query: str
    mode: str = "image"         # image (ảnh kèm caption) | caption (chỉ caption); luôn tìm bằng hybrid
    k: int = 10                 # số ảnh trả về, 1–100


@app.get("/health")
def health():
    """Kiểm tra server còn sống (UI có thể gọi trước khi tìm)."""
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    """Tìm top-k ảnh cho câu query.

    - Input sai (rỗng, không phải tiếng Anh, quá dài, mode/k sai): search.py raise ValueError
      → trả HTTP 400 kèm thông báo, để UI hiện cho người dùng.
    Lưu ý: sai KIỂU dữ liệu (vd. k = "abc") thì FastAPI tự chặn trước và trả 422.
    """
    try:
        results = app.state.searcher.search(req.query, mode=req.mode, k=req.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"query": req.query, "mode": req.mode, "k": req.k, "results": results}
class ChatRequest(BaseModel):
    """Body của POST /chat."""
    question: str
    k: int = 5                  # số ảnh đưa cho LLM đọc caption


@app.post("/chat")
def chat(req: ChatRequest):
    """Hỏi chatbot: trả {answer, results}. Câu hỏi sai (rỗng, không phải tiếng Anh, quá dài) → HTTP 400."""
    try:
        return app.state.chatbot.chat(req.question, k=req.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
class FollowupRequest(BaseModel):
    """Body của POST /followup."""
    question: str
    image_ids: list[str]        # các ảnh đang hiện trên giao diện, theo đúng thứ tự
    history: list[dict] = []    # các lượt hỏi tiếp trước đó: [{"question": ..., "answer": ...}]


@app.post("/followup")
def followup(req: FollowupRequest):
    """Hỏi tiếp về các ảnh đang hiện, không tìm lại. Trả {answer}. Input sai → HTTP 400."""
    try:
        return app.state.chatbot.followup(req.question, req.image_ids, req.history)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))