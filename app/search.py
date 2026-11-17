import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from transformers import CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parent.parent      # thư mục flickr-search/
QDRANT_URL = "http://localhost:6333"
MODEL = "openai/clip-vit-base-patch32"            
RRF_K = 60                                         
N_CANDIDATES = 100                                 # mỗi pp lấy 100 ảnh ứng viên trước khi gộp
MODES = ("caption", "image")

nltk.download("stopwords", quiet=True)
STOP = set(stopwords.words("english"))
stemmer = PorterStemmer()


def tokenize(text):
    """Tiền xử lý cho BM25, giống hệt notebook 04: lowercase, bỏ stopword, stemming."""
    words = re.findall(r"[a-z]+", text.lower())
    return [stemmer.stem(w) for w in words if w not in STOP]


def validate_query(text, mode, k):

    # text phải là chuỗi: None hay số sẽ làm .strip() văng AttributeError khó hiểu
    if not isinstance(text, str):
        raise ValueError(f"query phải là chuỗi, nhận được {type(text).__name__}")
    text = text.strip()
    if not text:                                   # rỗng hoặc toàn dấu cách
        raise ValueError("query rỗng, hãy nhập câu tìm kiếm")
    # Chỉ hỗ trợ tiếng Anh, còn tokenize() của BM25 chỉ giữ a–z
    # có CHỮ CÁI nào ngoài a–z thì coi là không phải tiếng Anh.
    # - Bắt được: tiếng Việt có dấu ("chó chơi trên tuyết"), chữ Trung/Nhật/Nga...
    # - Số, dấu câu, emoji không phải chữ cái nên vẫn cho qua.
    non_english = sorted({ch for ch in text if ch.isalpha() and not ch.isascii()}) # những từ ko phỉa tiếng anh 
    if non_english:
        raise ValueError(f"chỉ hỗ trợ query tiếng Anh, có ký tự lạ: {''.join(non_english)}")

    if mode not in MODES:
        raise ValueError(f"mode phải là một trong {MODES}, nhận được {mode!r}")

    # k phải là số nguyên trong [1, N_CANDIDATES]:
    # - bool là lớp con của int trong Python (True == 1) nên phải loại riêng
    # - k <= 0 làm ranked[:k] cắt sai mà không báo lỗi
    # - mỗi phương pháp chỉ lấy N_CANDIDATES ảnh ứng viên, nên k lớn hơn cũng không ra thêm ảnh
    if not isinstance(k, int):
        raise ValueError(f"k phải là số nguyên, nhận được {type(k).__name__}")
    if not 1 <= k <= N_CANDIDATES:
        raise ValueError(f"k phải nằm trong khoảng 1–{N_CANDIDATES}, nhận được {k}")

    return text


class Searcher:
    """Nạp model + dữ liệu một lần, sau đó gọi search() bao nhiêu lần cũng được."""

    def __init__(self, client=None) : # client gửi tín hiệu giao tiếp cho qdrant
        # client: truyền vào khi muốn dùng Qdrant khác (vd. khi test); mặc định nối tới Docker
        self.client = client or QdrantClient(url=QDRANT_URL)

        # CLIP chỉ dùng phần text: ảnh và caption đã encode sẵn và nằm trong Qdrant
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(MODEL).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL)
        # load model CLIP 
        #tìm kiếm, vừa truy xuất thông tin cần thiết để hiển thị kết quả.
        caps = pd.read_csv(ROOT / "data" / "captions.csv")
        self.caps = caps[caps["split"] == "test"].reset_index(drop=True)
        meta = pd.read_csv(ROOT / "data" / "metadata.csv")
        imgs = meta[meta["split"] == "test"]
        self.img_path = dict(zip(imgs["image_id"], imgs["path"]))          # image_id → đường dẫn ảnh
        self.first_caption = self.caps.groupby("image_id")["caption"].first().to_dict()   # caption dự phòng để hiển thị

        # BM25 dựng trên 4999 caption, thứ tự dòng khớp self.caps
        self.bm25 = BM25Okapi([tokenize(c) for c in self.caps["caption"]])

    # ---------- 3 thành phần của hybrid: mỗi hàm trả về danh sách (image_id, điểm, caption khớp) đã xếp hạng ----------
    # Người dùng không gọi trực tiếp các hàm này; hybrid() gọi cả 3 rồi gộp bằng RRF.

    def encode(self, text): # biến truy vấn thành vector dang pytorch 
        """Câu text → vector CLIP 512 chiều, đã chuẩn hoá L2 (giống lúc encode caption ở notebook 03)."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad(): #tắt gradient vì ko huấn luyện 
            v = self.model.get_text_features(**inputs).pooler_output[0] #Đưa các token vào CLIP để lấy vector biểu diễn ngữ nghĩa của câu.
        return (v / v.norm()).cpu().numpy() # chuân hoá L2 để có thể tính similarity giữa các vector

    def t2i(self, vec, n=N_CANDIDATES): # Tìm ảnh bằng cách so sánh trực tiếp vector text của người dùng với các vector ảnh trong Qdrant.
        """Text → Image: so query với vector ảnh trong collection `images`."""
        hits = self.client.query_points("images", query=vec.tolist(), limit=n).points
        return [(h.payload["image_id"], h.score, None) for h in hits]   # T2I không có caption khớp

    def t2t(self, vec, n=N_CANDIDATES):
        """Text → Text: so query với vector caption trong collection `captions`, rồi gộp theo ảnh."""
        # lấy dư caption (×5) vì mỗi ảnh có ~5 caption, gộp xong vẫn đủ n ảnh
        hits = self.client.query_points("captions", query=vec.tolist(), limit=n * 5).points
        return self._best_per_image((h.payload["image_id"], h.score, h.payload["caption"]) for h in hits)[:n]

    def bm25_search(self, text, n=N_CANDIDATES):
        """BM25: so từ khoá của query với 4999 caption, rồi gộp theo ảnh."""
        scores = self.bm25.get_scores(tokenize(text))

        # Query không có từ nào BM25 dùng được (toàn stopword như "the a of", toàn số/ký hiệu,
        # hoặc từ không có trong caption nào) → mọi điểm đều bằng 0.
        # Khi đó argsort vẫn trả về một thứ tự CỐ ĐỊNH ([0, 3335, 3334, ...]), tức là luôn cùng
        # ~100 ảnh không liên quan gì tới query. Nếu giữ lại, hybrid sẽ cho các ảnh này phiếu RRF
        # giả và làm lệch top-k. Vì vậy trả [] để báo "BM25 không có ý kiến":
        # BM25 không cộng phiếu nào → hybrid tự lùi về T2I + T2T (CLIP-only)
        if scores.max() == 0:
            return []

        top = np.argsort(-scores)[: n * 5] # vì mỗi ảnh cần 5 caption, mà muốn lấy 3 ảnh thì nó sẽ đi qua đủ 15 caption và chọn được cái ảnh tốt nhất, nó trả về xếp hạng theo vị trí của ảnh  
        rows = self.caps.iloc[top] # lấy caption tương ứng 
        return self._best_per_image(zip(rows["image_id"], scores[top], rows["caption"]))[:n]

    @staticmethod
    def _best_per_image(hits): # loại bỏ ảnh trùng hits là danh sách caption có độ tương đồng cao 
        """Gộp theo ảnh: mỗi ảnh giữ caption điểm cao nhất (hits đã xếp từ cao xuống thấp)."""
        seen, out = set(), []
        for img_id, score, caption in hits:
            if img_id not in seen:
                seen.add(img_id)
                out.append((img_id, float(score), caption))
        return out

    def hybrid(self, text, vec, n=N_CANDIDATES):
        """RRF T2I + T2T + BM25: mỗi phương pháp 'bỏ phiếu' 1/(60 + hạng), ảnh nào tổng cao nhất thì thắng.

        Nếu BM25 trả [] (query không có từ khoá dùng được) thì vòng lặp bên dưới không cộng phiếu nào
        từ BM25, nên kết quả tự động chỉ là RRF của T2I + T2T. Không cần if riêng.
        """
        score, caption = {}, {} # lưu tổng điểm RRF từng ảnh và caption đại diện cho từng ảnh
        for ranked in (self.t2i(vec, n), self.t2t(vec, n), self.bm25_search(text, n)):
            for rank, (img_id, _, cap) in enumerate(ranked, start=1):
                score[img_id] = score.get(img_id, 0) + 1 / (RRF_K + rank)   # ảnh không có trong top n của 1 phương pháp thì được 0 điểm từ phương pháp đó
                if cap and img_id not in caption: # dừng khi duyệt hết 
                    caption[img_id] = cap                                    # giữ caption khớp đầu tiên (T2T trước, rồi BM25)
        order = sorted(score, key=score.get, reverse=True)
        return [(i, score[i], caption.get(i)) for i in order]
# vòng ngoài lấy 1 phương pháp vòng trong là tính điểm caption cao nhất cho ảnh đó của phương pháp đó, sau đó cứ lặp lại đến hết  
    def check_clip_length(self, text):
        """Câu dài quá giới hạn của CLIP → raise ValueError thay vì để CLIP cắt đuôi âm thầm.

        CLIP chỉ đọc tối đa 77 token (kể cả 2 token đánh dấu đầu/cuối câu), khoảng 50–60 từ tiếng Anh.
        encode() dùng truncation=True nên câu dài không crash mà bị bỏ phần đuôi, người dùng không biết.
        Đếm bằng chính tokenizer của CLIP vì 1 từ có thể thành nhiều token (đếm từ/ký tự sẽ sai).
        Nằm trong Searcher chứ không trong validate_query vì cần tokenizer đã nạp ở __init__.
        Trong hybrid còn có lý do nữa: BM25 đọc cả câu, CLIP chỉ đọc nửa đầu → hai bên lệch nhau.
        """
        tok = self.processor.tokenizer
        max_len = tok.model_max_length                                     # = 77 với CLIP
        n = len(tok(text, truncation=False, verbose=False)["input_ids"])   # verbose=False: tắt cảnh báo "quá dài" của HF
        if n > max_len:
            raise ValueError(f"câu quá dài ({n} token, CLIP chỉ đọc tối đa {max_len}), hãy rút gọn")

    # ---------- hàm chính mà api.py / ui.py gọi ----------

    def search(self, text, mode="image", k=10): # cái này là hàm điều phối hệ thống
        """Tìm bằng hybrid T2I + T2T + BM25, trả về top k kết quả.

        mode="image":   list dict {rank, image_id, score, caption, path}  → UI hiện ảnh kèm caption
        mode="caption": list dict {rank, image_id, score, caption}        → UI chỉ hiện caption
        Hai mode dùng CHUNG một bảng xếp hạng, chỉ khác phần trả về.

        Input sai → raise ValueError (xem validate_query và check_clip_length).
        """
        text = validate_query(text, mode, k)         # kiểm tra input trước khi tốn công encode CLIP
        self.check_clip_length(text)
        ranked = self.hybrid(text, self.encode(text))

        results = []
        for r, (img_id, score, cap) in enumerate(ranked[:k], start=1):
            hit = {
                "rank": r,
                "image_id": img_id,
                "score": round(score, 4),
                "caption": cap or self.first_caption[img_id],   # ảnh chỉ đến từ T2I thì không có caption khớp → hiện caption đầu của ảnh
            }
            if mode == "image":
                hit["path"] = str(ROOT / self.img_path[img_id])  # đường dẫn tuyệt đối để UI mở ảnh
            results.append(hit)
        return results


if __name__ == "__main__":
    # chạy thử: python app/search.py
    s = Searcher()
    for m in MODES:
        print(f"\n== {m}")
        for hit in s.search("The dogs are in the snow in front of a fence .", mode=m, k=5):
            print(hit["rank"], hit["image_id"], hit["score"], "|", hit["caption"])
