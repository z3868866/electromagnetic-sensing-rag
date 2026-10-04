import sys
sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client, online_hybrid_search

FILE_TITLE = "基于深度卷积神经网络的协作频谱感知方法"
COLL = milvus_config.chunks_collection
client = get_milvus_client()

print("集合存在：", client.has_collection(COLL))

p = client.query(COLL, filter='chunk_level=="parent"', output_fields=["chunk_id"], limit=500)
c = client.query(COLL, filter='chunk_level=="child"', output_fields=["chunk_id"], limit=500)
print(f"父块 {len(p)} | 子块 {len(c)}")

print("\n=== 子块样本（看 item_name 实际值）===")
sample = client.query(COLL, filter='chunk_level=="child"',
                      output_fields=["item_name", "parent_title", "parent_uid"], limit=5)
for s in sample:
    print(f"  item_name=「{s.get('item_name')}」| {s.get('parent_title','')[:24]} | {s.get('parent_uid')}")

print("\n=== 检索A：只过滤 child（无 item_name）===")
ra = online_hybrid_search(client, COLL, "DCNN 如何缓解梯度消失",
                          expr='chunk_level=="child"', limit=5,
                          output_fields=["parent_uid", "parent_title", "item_name"])
for h in (ra[0] if ra else []):
    e = h.get("entity", {})
    print(" ", e.get("parent_title", "")[:26], "|", e.get("parent_uid"))

print("\n=== 检索B：child + item_name 过滤 ===")
rb = online_hybrid_search(client, COLL, "DCNN 如何缓解梯度消失",
                          expr=f'chunk_level=="child" and item_name in ["{FILE_TITLE}"]',
                          limit=5, output_fields=["parent_uid", "parent_title"])
for h in (rb[0] if rb else []):
    e = h.get("entity", {})
    print(" ", e.get("parent_title", "")[:26], "|", e.get("parent_uid"))
