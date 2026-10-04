import os
import sys

sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from config.milvus_config import milvus_config
from processor.import_processor.nodes.node_document_split import NodeDocumentSplit
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus
from utils.milvus_utils import get_milvus_client

MD = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法_new.md"
FILE_TITLE = "基于深度卷积神经网络的协作频谱感知方法"
COLL = milvus_config.chunks_collection

with open(MD, "r", encoding="utf-8") as f:
    md_content = f.read()

# 0. drop 旧集合（旧 schema）
client = get_milvus_client()
if client.has_collection(COLL):
    client.drop_collection(COLL)
    print("已删除旧集合（旧 schema）")

# 1. 节点链（跳过 MinerU，直接用已有 Markdown）
state = {"task_id": "reimport_pc_001", "file_title": FILE_TITLE, "md_content": md_content}
state = NodeDocumentSplit()(state)
state = NodeItemNameRecognition()(state)
state = NodeBGEEmbedding()(state)
state = NodeImportMilvus()(state)

parents, children = state["parents"], state["chunks"]
print(f"\n切分：父块 {len(parents)} | 子块 {len(children)}")

# 2. 集合内 parent / child 条数
p_rows = client.query(COLL, filter='chunk_level=="parent"',
                      output_fields=["chunk_id"], limit=500)
c_rows = client.query(COLL, filter='chunk_level=="child"',
                      output_fields=["chunk_id"], limit=500)
print(f"集合 {COLL}：父块 {len(p_rows)} | 子块 {len(c_rows)} | 总 {len(p_rows)+len(c_rows)}")

# 3. 子块检索 + 回取父块 验证
from utils.milvus_utils import online_hybrid_search
res = online_hybrid_search(
    client, COLL, "DCNN 如何缓解梯度消失",
    expr=f'chunk_level=="child" and item_name in ["{FILE_TITLE}"]',
    limit=5, output_fields=["parent_uid", "parent_title"])
hits = res[0]
print("\n=== 子块检索 ===")
uids = []
for h in hits:
    e = h.get("entity", {})
    uids.append(e.get("parent_uid"))
    print(" ", e.get("parent_title", "")[:30], "|", e.get("parent_uid"))

from processor.query_processor.nodes.node_rerank import NodeRerank
expanded = NodeRerank()._step_4_expand_parents([
    {"source": "local", "parent_uid": u, "score": 0.9 - i*0.1} for i, u in
    enumerate(dict.fromkeys(uids))])
print("\n=== 回取父块（喂给生成）===")
for d in expanded:
    print(f"  {d['title'][:34]} | {len(d['content'])}字 | score={round(d['score'],3)}")
