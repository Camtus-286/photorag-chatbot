"""Chatbot RAG của PhotoRAG: một ô chat duy nhất.

Mỗi lượt, router (một lần gọi LLM ngắn) quyết định câu hỏi thuộc loại nào:
    search  → tìm bộ ảnh mới bằng search.py rồi trả lời
    current → hỏi về các ảnh đang hiện, không tìm lại
    chat    → chào hỏi / nói chuyện, không cần ảnh
History đi vào mảng messages đúng role (system → user → assistant → ...).
Các lượt cũ chỉ giữ câu hỏi gốc và câu trả lời cuối; caption chỉ gắn vào câu hỏi mới nhất.
"""
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from app.search import Searcher

load_dotenv()

MAX_HISTORY = 10                       # số tin gần nhất gửi cho LLM (= 5 cặp hỏi–đáp)
DEFAULT_K = 5                          # số ảnh khi người dùng không nói muốn mấy ảnh
MAX_K = 20                             # chặn trên, tránh "show me 500 photos"
ROUTES = ("search", "current", "chat")

SYSTEM_PROMPT = """You are a friendly librarian at a photo library. The library holds photos, and each photo has a few captions written by different people. You cannot see the photos; you only know their captions.
The photos currently on screen, with their captions, are listed in the visitor's latest message.

How to respond:
- If the visitor greets you, thanks you, or makes small talk, reply warmly in one short sentence and offer to help. Do not mention captions in that case.
- If the visitor asks what you can do, explain briefly that you can find photos and answer questions about them based on their captions.
- If the visitor looks for photos or asks about the photos, follow the rules below.

Rules for photos:
1. Use ONLY the captions listed in the visitor's latest message. Earlier messages are only there to understand what the visitor means.
2. If none of the captions match what the visitor is looking for, say that no suitable photo was found.
3. If the captions do not contain the answer, say politely that the captions do not say. Do not guess.
4. Do not use outside knowledge. Do not add details, feelings, moods or reasons that are not written in the captions.
5. "The first picture" means photo [1], "the second picture" means photo [2], and so on.
6. Cite the photos you used by number with plain square brackets, like [1] or [3].
Keep the answer to 2-4 sentences."""

ROUTER_PROMPT = """You route messages for a photo search chatbot.
Photos currently on screen: {on_screen}

Recent conversation:
{history}

New message: {question}

Reply with exactly one word:
- search  : the message describes something to find, so NEW photos are needed (e.g. "dogs in the snow", "now show me cats")
- current : the message asks about the photos already on screen (e.g. "the second one", "why are they dressed like that?", "what color is the car?")
- chat    : greeting, thanks, small talk, or asking what the bot can do; no photos needed
Answer:"""


def parse_k(question):
    """Chữ số trong câu = số ảnh cần tìm ("5 two girls" → 5). Trả (k, câu đã bỏ chữ số đó).

    Nội dung ảnh người dùng viết bằng chữ ("two girls"), nên chữ số chỉ dùng để chỉ số ảnh.
    Không có chữ số thì lấy DEFAULT_K.
    """
    match = re.search(r"\d+", question)
    if not match:
        return DEFAULT_K, question
    k = max(1, min(int(match.group(0)), MAX_K))
    rest = " ".join((question[:match.start()] + " " + question[match.end():]).split())
    return k, rest or question


class Chatbot:
    def __init__(self, searcher=None):
        self.searcher = searcher or Searcher()          # dùng Searcher có sẵn nếu được đưa, không thì tự tạo
        self.client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
        self.model = os.getenv("LLM_MODEL")
        # image_id → cả 5 caption của ảnh (search() chỉ trả 1 caption mỗi ảnh)
        self.all_captions = self.searcher.caps.groupby("image_id")["caption"].apply(list).to_dict()

    def _ask(self, messages, temperature=0.3):
        """Gửi mảng messages cho LLM, trả về câu trả lời dạng chuỗi."""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )
        text = resp.choices[0].message.content or ""
        return text.replace("【", "[").replace("】", "]").strip()   # model hay viết 【1】, đổi về [1]

    def _route(self, question, history, has_photos):
        """Hỏi LLM: câu này cần tìm ảnh mới (search), hỏi ảnh đang hiện (current) hay chỉ trò chuyện (chat)."""
        recent = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:]) or "(no conversation yet)"
        prompt = ROUTER_PROMPT.format(on_screen="yes" if has_photos else "none",
                                      history=recent, question=question)
        out = self._ask([{"role": "user", "content": prompt}], temperature=0).lower()
        for word in re.findall(r"[a-z]+", out):          # lấy từ hợp lệ đầu tiên model viết ra
            if word in ROUTES:
                return word
        return "search"                                   # không rõ thì tìm mới, an toàn hơn

    def ask(self, question, history=None, image_ids=None):
        """Một lượt chat.

        question:  câu người dùng vừa gõ.
        history:   các tin trước, list {"role": "user"/"assistant", "content": ...} (không kèm caption).
        image_ids: ảnh đang hiện trên màn hình, theo đúng thứ tự (ảnh đầu là [1]); rỗng nếu chưa có.
        Số ảnh cần tìm là chữ số trong câu ("5 two girls"), không có thì lấy DEFAULT_K.
        Trả về {"answer", "route", "query", "results", "image_ids"}; query và results chỉ có khi vừa tìm ảnh mới.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("câu hỏi rỗng, hãy nhập câu hỏi")
        question = question.strip()
        history = [m for m in (history or []) if m.get("role") in ("user", "assistant")][-MAX_HISTORY:]
        image_ids = list(image_ids or [])
        unknown = [i for i in image_ids if i not in self.all_captions]
        if unknown:
            raise ValueError(f"không có ảnh với image_id: {unknown}")

        route = self._route(question, history, has_photos=bool(image_ids))
        k, query = parse_k(question)                     # chỉ dùng khi route = search
        if route == "current" and not image_ids:          # chưa có ảnh thì không thể hỏi về ảnh đang hiện
            route = "search"

        results = None
        if route == "search":
            hits = self.searcher.search(query, mode="image", k=k)   # tìm bằng câu đã bỏ chữ số
            for h in hits:                                 # kèm cả 5 caption để giao diện cho xem đủ
                h["captions"] = self.all_captions.get(h["image_id"], [h["caption"]])
            results = hits
            image_ids = [h["image_id"] for h in hits]

        if route == "chat":
            user_msg = question
        else:
            context = "\n".join(
                f"[{n}] " + " | ".join(self.all_captions[img_id]) for n, img_id in enumerate(image_ids, start=1)
            )
            user_msg = f"Photos on screen:\n{context}\n\nQuestion: {question}"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_msg}]
        answer = self._ask(messages)
        return {"answer": answer, "route": route, "query": query if route == "search" else None,
                "results": results, "image_ids": image_ids}


if __name__ == "__main__":
    bot = Chatbot()
    history, image_ids = [], []
    print("Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            out = bot.ask(q, history=history, image_ids=image_ids)
        except ValueError as e:                         # câu hỏi rỗng / không phải tiếng Anh / quá dài
            print("Error:", e)
            continue
        print(f"  (route: {out['route']}" + (f", query: {out['query']!r}" if out["query"] else "") + ")")
        for h in out["results"] or []:
            print(f"  [{h['rank']}] {h['caption']}")
        print("Bot:", out["answer"])
        image_ids = out["image_ids"]
        history += [{"role": "user", "content": q},       # lưu câu hỏi GỐC, không kèm caption
                    {"role": "assistant", "content": out["answer"]}]
