"""Milvus 清理：删除练习集合 practice_vectors。"""
import os
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
mc = MilvusClient(uri=os.getenv("MILVUS_URL", "http://localhost:19530"))
COLL = "practice_vectors"
if mc.has_collection(COLL):
    mc.drop_collection(COLL)
    print("已删除集合：", COLL)
print("剩余集合：", mc.list_collections())
