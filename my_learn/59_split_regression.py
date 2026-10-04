import os
import sys

sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from processor.import_processor.nodes.node_document_split import NodeDocumentSplit

MD = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法_new.md"
FILE_TITLE = "基于深度卷积神经网络的协作频谱感知方法"

with open(MD, "r", encoding="utf-8") as f:
    md_content = f.read()

state = {"task_id": "task_split_regression", "md_content": md_content, "file_title": FILE_TITLE}
state = NodeDocumentSplit()(state)

chunks = state["chunks"]
parents = []
for c in chunks:
    if c["parent_title"] not in parents:
        parents.append(c["parent_title"])

print("\n================ 修复后识别到的章节（parent_title）================")
for i, p in enumerate(parents, 1):
    n = sum(1 for c in chunks if c["parent_title"] == p)
    print(f"  {i:2d}. {p} | 块数 {n}")
print(f"\n章节数：{len(parents)} | 总 chunk 数：{len(chunks)}")

must_have = ["## 3.5 DCNN频谱感知算法", "## 4 实验结果及分析", "## 5 结论", "## 参 考 文 献"]
print("\n关键章节回归：")
for m in must_have:
    print(f"  {'已识别' if m in parents else '仍缺失'}：{m}")
