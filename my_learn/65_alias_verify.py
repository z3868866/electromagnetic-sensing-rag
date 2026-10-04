# my_learn/65_alias_verify.py
"""
优化点②验证：重建 kb_item_names（标准名 + 别名 match_text），
测试缩写 “DCNN” 能否对齐到标准文献名。
"""
import os
import sys
import json

sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client, online_hybrid_search
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition

FILE_TITLE = "基于深度卷积神经网络的协作频谱感知方法"
CHILDREN_JSON = r"D:\output\基于深度卷积神经网络的协作频谱感知方法\children.json"

client = get_milvus_client()
coll = milvus_config.item_name_collection

# 1. 删除旧集合（schema 已变更）
if client.has_collection(coll):
    client.drop_collection(coll)
    print("已删除旧 kb_item_names")

# 2. 载入子块，跑主体识别节点（抽取别名 + match_text 建索引）
with open(CHILDREN_JSON, "r", encoding="utf-8") as f:
    children = json.load(f)

state = NodeItemNameRecognition()({
    "task_id": "task_alias_01",
    "file_title": FILE_TITLE,
    "chunks": children,
})

# 3. 查回标准名记录
rows = client.query(coll, filter='file_title == "基于深度卷积神经网络的协作频谱感知方法"',
                    output_fields=["item_name", "aliases", "match_text"],
                    consistency_level="Strong")
print("\n=== 标准名记录 ===")
for r in rows:
    print("item_name :", r.get("item_name"))
    print("aliases   :", r.get("aliases"))
    print("match_text:", str(r.get("match_text", ""))[:300])

# 4. 对齐测试
print("\n=== 对齐检索测试 ===")
for q in ["DCNN", "深度卷积神经网络", "今天天气"]:
    res = online_hybrid_search(
        client=client,
        collection_name=coll,
        query_text=q,
        limit=3,
        output_fields=["item_name"],
        consistency_level="Strong",
    )
    print(f"\n查询「{q}」:")
    for h in res[0]:
        print(f"  score={round(h.get('distance', 0), 4)} | {h.get('entity', {}).get('item_name')}")
