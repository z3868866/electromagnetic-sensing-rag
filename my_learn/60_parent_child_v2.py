import os
import sys
import re
import json

sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

import numpy as np
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from utils.embedding_utils import get_online_embedder

MD = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法_new.md"
QUESTION = "DCNN 是如何缓解梯度消失问题的？"
TOP_N = 5
WHITELIST = {"参考文献", "摘要", "关键词"}

with open(MD, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

title_line_re = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
num_re = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
exclude_re = re.compile(r"^(表|图|Table|Figure|Fig\.?)\s*\d")
fence_open_re = re.compile(r"^(`{3,}|~{3,})(.*)$")
fence_close_re = re.compile(r"^(`{3,}|~{3,})$")


def is_real_title(text, level):
    if exclude_re.match(text):
        return False
    if level == 1:  # 文档主标题（作者/摘要/关键词所在头部）
        return True
    if num_re.match(text):
        return True
    if text.replace(" ", "") in WHITELIST:
        return True
    return False


# 1. 扫描分界：构建父块
in_code = False
start_marker = None
parents = []
cur_title = None
cur_body = []


def flush():
    if cur_title is not None:
        body = "\n".join(cur_body).strip()
        parents.append({"parent_title": cur_title, "content": body})


for line in lines:
    s = line.strip()

    # 围栏状态机（开始可带语言，结束为纯围栏）
    if not in_code and fence_open_re.match(s):
        in_code = True
        start_marker = fence_open_re.match(s).group(1)
        if cur_title is not None:
            cur_body.append(line)
        continue
    if in_code and fence_close_re.match(s):
        cm = fence_close_re.match(s).group(1)
        if cm[0] == start_marker[0] and len(cm) >= len(start_marker):
            in_code = False
            start_marker = None
        if cur_title is not None:
            cur_body.append(line)
        continue

    # 真标题 -> 分界
    m = title_line_re.match(line)
    if not in_code and m and is_real_title(m.group(2), len(m.group(1))):
        flush()
        cur_title = s
        cur_body = []
    else:
        if cur_title is not None:
            cur_body.append(line)
flush()

for i, p in enumerate(parents):
    p["parent_id"] = f"p{i}"

# 2. 父块切子块
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    separators=["\n\n", "\n", "。", "！", "？", " ", ""],
)
children = []
for p in parents:
    if not p["content"]:
        continue
    for j, t in enumerate(splitter.split_text(p["content"])):
        t = t.strip()
        if t:
            children.append({"child_id": f"{p['parent_id']}_c{j}",
                             "parent_id": p["parent_id"],
                             "parent_title": p["parent_title"],
                             "content": t})

print("=== 父块（规范章节，空块保留）===")
for p in parents:
    nc = sum(1 for c in children if c["parent_id"] == p["parent_id"])
    tag = "（空）" if not p["content"] else f"{len(p['content'])}字"
    print(f"  {p['parent_id']} | {p['parent_title'][:42]} | {tag} | 子块{nc}")
print(f"父块数：{len(parents)} | 子块总数：{len(children)}")

# 3. 子块向量化（每批10）
embedder = get_online_embedder()
texts = [c["content"] for c in children]
vecs = []
for i in range(0, len(texts), 10):
    vecs.extend(embedder.embed_documents(texts[i:i + 10]))
mat = np.array(vecs)

# 4. 检索 + small-to-big
qv = np.array(embedder.embed_query(QUESTION))
sims = mat @ qv / (np.linalg.norm(mat, axis=1) * np.linalg.norm(qv) + 1e-8)
top_idx = sims.argsort()[::-1][:TOP_N]

print("\n=== 命中的子块 ===")
hit_pids = []
for rank, i in enumerate(top_idx, 1):
    c = children[i]
    hit_pids.append(c["parent_id"])
    print(f"  #{rank} sim={round(float(sims[i]),3)} | {c['parent_title'][:26]}")
    print(f"       {c['content'][:88].replace(chr(10),' ')}…")

seen, feed = set(), []
for pid in hit_pids:
    if pid not in seen:
        seen.add(pid)
        feed.append(next(p for p in parents if p["parent_id"] == pid))

child_len = sum(len(children[i]["content"]) for i in top_idx)
parent_len = sum(len(p["content"]) for p in feed)
print("\n=== small-to-big 回取的父块 ===")
for p in feed:
    print(f"  {p['parent_title'][:42]} | {len(p['content'])}字")
print(f"\n只给命中子块：{child_len} 字 | 回取父块：{parent_len} 字")
