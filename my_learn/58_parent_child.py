import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

import numpy as np
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from utils.embedding_utils import get_online_embedder

CHUNKS_JSON = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks.json"
QUESTION = "DCNN 是如何缓解梯度消失问题的？"
TOP_N = 5

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 1. 聚合父块：按 parent_title 分组，按 part 排序后拼接
groups = {}
order = []
for c in chunks:
    pt = c["parent_title"]
    if pt not in groups:
        groups[pt] = []
        order.append(pt)
    groups[pt].append((c.get("part", 0), c["content"]))

parents = []
for idx, pt in enumerate(order):
    pieces = sorted(groups[pt], key=lambda x: x[0])
    full = "\n".join(p[1] for p in pieces)
    parents.append({"parent_id": f"p{idx}", "parent_title": pt, "content": full})

# 2. 父块切子块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
children = []
for p in parents:
    for j, text in enumerate(splitter.split_text(p["content"])):
        text = text.strip()
        if text:
            children.append({"child_id": f"{p['parent_id']}_c{j}",
                             "parent_id": p["parent_id"],
                             "parent_title": p["parent_title"],
                             "content": text})

print("=== 父块（章节级）===")
for p in parents:
    n_child = sum(1 for c in children if c["parent_id"] == p["parent_id"])
    print(f"  {p['parent_id']} | {p['parent_title'][:40]} | {len(p['content'])}字 | 子块{n_child}")
print(f"父块数：{len(parents)} | 子块总数：{len(children)}")

# 3. 子块在线向量化（每批10条）
embedder = get_online_embedder()
texts = [c["content"] for c in children]
all_vecs = []
for i in range(0, len(texts), 10):
    all_vecs.extend(embedder.embed_documents(texts[i:i + 10]))
    print(f"  子块向量化 {min(i+10, len(texts))}/{len(texts)}")
mat = np.array(all_vecs)

# 4. 问题向量 + 余弦相似度
qvec = np.array(embedder.embed_query(QUESTION))
sims = mat @ qvec / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qvec) + 1e-8)
top_idx = sims.argsort()[::-1][:TOP_N]

print("\n=== 命中的子块（小、聚焦）===")
hit_parent_ids = []
for rank, i in enumerate(top_idx, 1):
    c = children[i]
    hit_parent_ids.append(c["parent_id"])
    print(f"  #{rank} sim={round(float(sims[i]),3)} | {c['parent_title'][:24]}")
    print(f"       {c['content'][:90].replace(chr(10),' ')}…")

# 5. 回取父块（small-to-big），去重保序
seen, feed_parents = set(), []
for pid in hit_parent_ids:
    if pid not in seen:
        seen.add(pid)
        feed_parents.append(next(p for p in parents if p["parent_id"] == pid))

child_ctx_len = sum(len(children[i]["content"]) for i in top_idx)
parent_ctx_len = sum(len(p["content"]) for p in feed_parents)
print("\n=== small-to-big 回取的父块 ===")
for p in feed_parents:
    print(f"  {p['parent_title'][:40]} | {len(p['content'])}字")
print(f"\n只给命中子块的上下文：{child_ctx_len} 字")
print(f"回取父块后的上下文  ：{parent_ctx_len} 字（完整章节，供 LLM 生成）")
