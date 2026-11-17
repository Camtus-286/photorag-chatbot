"""Nạp embedding CLIP của tập test vào Qdrant: 2 collection `images` và `captions`."""
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parent.parent      # thư mục flickr-search/
EMB = ROOT / "artifacts" / "embeddings"
QDRANT_URL = "http://localhost:6333"
BATCH = 256


def load_data():
    caps = pd.read_csv(ROOT / "data" / "captions.csv")
    caps = caps[caps["split"] == "test"].reset_index(drop=True)
    meta = pd.read_csv(ROOT / "data" / "metadata.csv")
    imgs = meta[meta["split"] == "test"].reset_index(drop=True)

    img_vecs = np.load(EMB / "test_images.npy")     # (1000, 512), thứ tự khớp imgs
    cap_vecs = np.load(EMB / "test_captions.npy")   # (4999, 512), thứ tự khớp caps
    assert len(imgs) == len(img_vecs) and len(caps) == len(cap_vecs)
    return caps, imgs, img_vecs, cap_vecs


def create_collection(client, name, vecs, payloads):
    """Xoá collection cũ (nếu có), tạo mới, rồi nạp vector + payload theo từng batch."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name, vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE)
    )
    for i in range(0, len(vecs), BATCH):
        points = [
            PointStruct(id=j, vector=vecs[j].tolist(), payload=payloads[j])   # id = số dòng
            for j in range(i, min(i + BATCH, len(vecs)))
        ]
        client.upsert(name, points=points)
    print(f"{name}: {client.count(name).count} điểm")


def check_against_numpy(client, cap_vecs, img_vecs, n=200, k=10):
    """So top-k T2I của Qdrant với numpy trên n query đầu: phải trùng nhau."""
    expected = np.argsort(-(cap_vecs[:n] @ img_vecs.T), axis=1)[:, :k]
    same = 0
    for q in range(n):
        hits = client.query_points("images", query=cap_vecs[q].tolist(), limit=k).points
        same += [h.id for h in hits] == expected[q].tolist()
    print(f"Khớp numpy: {same}/{n} query")   # kỳ vọng: 200/200


def main():
    caps, imgs, img_vecs, cap_vecs = load_data()
    client = QdrantClient(url=QDRANT_URL)

    create_collection(client, "images", img_vecs,
                      imgs[["image_id", "path"]].to_dict("records"))

    caps = caps.merge(imgs[["image_id", "path"]], on="image_id", how="left")   # gắn đường dẫn ảnh cho từng caption
    create_collection(client, "captions", cap_vecs,
                      caps[["caption_id", "image_id", "caption", "path"]].to_dict("records"))

    check_against_numpy(client, cap_vecs, img_vecs)


if __name__ == "__main__":
    main()
