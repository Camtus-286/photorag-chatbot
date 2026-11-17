"""Chatbot RAG đơn giản: search.py tìm ảnh → LLM trả lời dựa trên caption."""
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.search import Searcher

load_dotenv()

PROMPT = """You are an assistant for a photo collection. You cannot see the photos.
Rules:
1. Use ONLY the captions in the context below.
2. If none of the captions match what the user asked for, say that no suitable photo was found.
3. Do not use outside knowledge. Do not add details, feelings, moods or reasons that are not written in the captions.
4. Cite the photos you used by number with plain square brackets, like [1] or [3].
Keep the answer to 2-4 sentences.

Context (captions of the photos found):
{context}

Question: {question}"""

FOLLOWUP_PROMPT = """You are a friendly librarian at a photo library. The library holds photos, and each photo has a few captions written by different people. You cannot see the photos; you only know their captions.
The visitor is looking at the photos listed below.

How to respond:
- If the visitor greets you, thanks you, or makes small talk, reply warmly in one short sentence and offer to help, for example "Hi! How can I help you with these photos?". Do not mention captions in that case.
- If the visitor asks what you can do, explain briefly that you can find photos and answer questions about them based on their captions.
- If the visitor asks about the photos, answer following the rules below.

Rules for questions about the photos:
1. Use ONLY the captions below.
2. If the captions do not contain the answer, say politely that the captions do not say. Do not guess.
3. Do not add details, feelings, moods or reasons that are not written in the captions.
4. "The first picture" means photo [1], "the second picture" means photo [2], and so on.
5. Cite the photos you used by number with plain square brackets, like [1] or [3].
Keep the answer to 1-3 sentences.

Photos on screen:
{context}

Earlier questions and answers about these photos:
{history}

Question: {question}"""


class Chatbot:
    def __init__(self, searcher=None):
        self.searcher = searcher or Searcher()          # dùng Searcher có sẵn nếu được đưa, không thì tự tạo
        self.client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))
        self.model = os.getenv("LLM_MODEL")             # người viết
        # image_id → cả 5 caption của ảnh, dùng cho câu hỏi tiếp (search() chỉ trả 1 caption mỗi ảnh)
        self.all_captions = self.searcher.caps.groupby("image_id")["caption"].apply(list).to_dict()

    def _ask(self, prompt):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return text.replace("【", "[").replace("】", "]").strip()   # model hay viết 【1】, đổi về [1] cho khớp số dưới ảnh

    def chat(self, question, k=5):
        """Câu hỏi mới: tìm ảnh rồi trả lời."""
        hits = self.searcher.search(question, mode="image", k=k)             # 1. tìm top-k ảnh
        context = "\n".join(f"[{h['rank']}] {h['caption']}" for h in hits)   # 2. ghép caption thành ngữ cảnh
        answer = self._ask(PROMPT.format(context=context, question=question))   # 3. đưa prompt cho LLM
        for h in hits:                                                       # kèm cả 5 caption để giao diện cho xem đủ
            h["captions"] = self.all_captions.get(h["image_id"], [h["caption"]])
        return {"answer": answer, "results": hits}

    def followup(self, question, image_ids, history=None):
        """Câu hỏi tiếp về các ảnh ĐANG HIỆN: không tìm lại, chỉ đọc caption của đúng các ảnh đó.

        image_ids: danh sách image_id theo đúng thứ tự trên màn hình (ảnh đầu là [1]).
        history:   các lượt hỏi tiếp trước đó, list {"question": ..., "answer": ...}.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("câu hỏi rỗng, hãy nhập câu hỏi")
        if not image_ids:
            raise ValueError("chưa có ảnh nào để hỏi, hãy tìm ảnh trước")
        unknown = [i for i in image_ids if i not in self.all_captions]
        if unknown:
            raise ValueError(f"không có ảnh với image_id: {unknown}")

        context = "\n".join(
            f"[{n}] " + " | ".join(self.all_captions[img_id]) for n, img_id in enumerate(image_ids, start=1)
        )
        past = "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in (history or [])[-5:]) or "(none)"
        answer = self._ask(FOLLOWUP_PROMPT.format(context=context, history=past, question=question.strip()))
        return {"answer": answer}


if __name__ == "__main__":
    bot = Chatbot()
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            out = bot.chat(q)
        except ValueError as e:                         # câu hỏi rỗng / không phải tiếng Anh / quá dài
            print("Error:", e)
            continue
        for h in out["results"]:
            print(f"  [{h['rank']}] {h['caption']}")
        print("Bot:", out["answer"])
