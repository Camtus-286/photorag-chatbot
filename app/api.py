"""API của PhotoRAG: nhận câu hỏi qua HTTP, gọi search.py / chat.py, trả kết quả dạng JSON.

Chạy (từ thư mục flickr-search/, Qdrant phải đang bật):
    python3 -m uvicorn app.api:app
Thử nhanh:
    curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
         -d '{"question": "show me 3 photos of dogs playing in the snow"}'
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


app = FastAPI(title="PhotoRAG API", lifespan=lifespan)


class SearchRequest(BaseModel):
    """Body của POST /search. Chỉ khai báo kiểu ở đây; kiểm tra chi tiết nằm trong search.py."""
    query: str
    mode: str = "image"         # image (ảnh kèm caption) | caption (chỉ caption); luôn tìm bằng hybrid
    k: int = 10                 # số ảnh trả về, 1–100


class AskRequest(BaseModel):
    """Body của POST /ask: một lượt chat."""
    question: str
    history: list[dict] = []    # các tin trước: [{"role": "user"/"assistant", "content": ...}], không kèm caption
    image_ids: list[str] = []   # ảnh đang hiện trên giao diện, theo đúng thứ tự; rỗng nếu chưa có


@app.get("/health")
def health():
    """Kiểm tra server còn sống (UI có thể gọi trước khi tìm)."""
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    """Tìm top-k ảnh cho câu query, không qua LLM.

    - Input sai (rỗng, không phải tiếng Anh, quá dài, mode/k sai): search.py raise ValueError
      → trả HTTP 400 kèm thông báo, để UI hiện cho người dùng.
    Lưu ý: sai KIỂU dữ liệu (vd. k = "abc") thì FastAPI tự chặn trước và trả 422.
    """
    try:
        results = app.state.searcher.search(req.query, mode=req.mode, k=req.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"query": req.query, "mode": req.mode, "k": req.k, "results": results}


@app.post("/ask")
def ask(req: AskRequest):
    """Một lượt chat. Router tự quyết tìm ảnh mới / hỏi ảnh đang hiện / trò chuyện, và đọc số ảnh từ câu hỏi.

    Trả {answer, route, query, results, image_ids}; query và results chỉ có khi route = "search".
    Câu hỏi sai (rỗng, không phải tiếng Anh, quá dài, image_id lạ) → HTTP 400.
    """
    try:
        return app.state.chatbot.ask(req.question, history=req.history, image_ids=req.image_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
