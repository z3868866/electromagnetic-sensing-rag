import os
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")
COLLECTION = "kb_chunks"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))

query_text = "深度卷积神经网络 频谱感知"
print("查询文本：", query_text)

results = client.search(
    collection_name=COLLECTION,
    data=[query_text],          # 传原始文本，Milvus 自动转 BM25 稀疏向量
    anns_field="sparse_vector",
    limit=3,
    output_fields=["title", "parent_title", "item_name"],
    search_params={"metric_type": "BM25"}
)

hits = results[0]
print("命中条数：", len(hits))
for hit in hits:
    entity = hit.get("entity", {})
    print("  BM25分数:", round(hit.get("distance", 0), 4),
          "| title:", str(entity.get("title", ""))[:30])
