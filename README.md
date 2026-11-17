# flickr-search

Tìm kiếm ảnh trên Flickr8k bằng CLIP + BM25 + Qdrant, kèm một chatbot RAG nhỏ (PhotoRAG) trả lời câu hỏi dựa trên caption của ảnh tìm được.

## Hệ thống gồm những gì
- `app/search.py` — tìm kiếm hybrid: gộp T2I, T2T (CLIP + Qdrant) và BM25 bằng RRF.
- `app/chat.py` — chatbot: lấy top-k ảnh từ `search.py`, đưa caption cho LLM viết câu trả lời; hỏi tiếp về các ảnh đang hiện thì đọc cả 5 caption của từng ảnh.
- `app/api.py` — FastAPI với các cửa `/health`, `/search`, `/chat`, `/followup`.
- `app/ui.py` — giao diện Streamlit: ô trên tìm ảnh, ô dưới hỏi tiếp về các ảnh đang hiện.
- `app/build_index.py` — nạp embedding của tập test vào Qdrant.
- `notebooks/` — EDA, tạo ground truth, đo baseline và ablation.

## Cấu hình LLM

Bản demo dùng gói miễn phí của Groq. Muốn chạy LLM ngay trên máy thì đổi ba dòng trên sang Ollama (`http://localhost:11434/v1`).

## Thứ tự chạy
Thư mục `data/`, `artifacts/` và `qdrant_storage/` không nằm trong repo, nên lần đầu phải chạy đủ các bước 3–7 để tạo lại.

1. `pip install -r requirements.txt`
2. `docker compose up -d` — bật Qdrant
3. `notebooks/01_eda.ipynb` — tải HF, xuất ảnh + metadata
4. `notebooks/02_ground_truth.ipynb` — tạo ground truth
5. `notebooks/03_baseline.ipynb` — encode CLIP, đo baseline
6. `notebooks/04_ablation.ipynb` — so sánh, ghi results.csv
7. `python3 app/build_index.py` — nạp Qdrant
8. `python3 -m uvicorn app.api:app` — chạy API (tài liệu tự sinh ở http://127.0.0.1:8000/docs)
9. Ở terminal khác: `python3 -m streamlit run app/ui.py` — mở giao diện

Thử nhanh chatbot trong terminal, không cần API hay giao diện: `python3 -m app.chat`

## Hạn chế
- Chỉ nhận câu hỏi tiếng Anh, và chỉ tìm trong 1.000 ảnh tập test.
- Chatbot không nhìn thấy ảnh, chỉ đọc caption. Chi tiết nào caption không ghi thì nó trả lời là không có thông tin.
- Tìm kiếm luôn trả đủ k ảnh kể cả khi không có ảnh nào liên quan; việc nhận ra "không có ảnh phù hợp" do LLM đảm nhận.
- Mỗi câu ở ô tìm kiếm là một lượt tìm độc lập; chỉ ô hỏi tiếp mới nhớ các lượt trước.
