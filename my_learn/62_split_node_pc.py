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

state = {"task_id": "task_pc_node", "md_content": md_content, "file_title": FILE_TITLE}
state = NodeDocumentSplit()(state)

parents, children = state["parents"], state["chunks"]
print("\n================ 父块（完整章节）================")
for p in parents:
    nc = sum(1 for c in children if c["parent_uid"] == p["parent_uid"])
    tag = "（空）" if not p["content"] else f"{len(p['content'])}字"
    print(f"  {p['parent_uid']} | {p['parent_title'][:42]} | {tag} | 子块{nc}")
print(f"\n父块 {len(parents)} | 子块 {len(children)}")
print("子块字段示例：", list(children[0].keys()))
