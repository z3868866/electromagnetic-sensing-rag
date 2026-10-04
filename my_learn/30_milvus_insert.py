import os, json
from pymilvus import MilvusClient
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")
COLLECTION = "kb_chunks"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))

# 读取带 dense_vector 的 chunks
chunks_path = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks_with_vector.json"
with open(chunks_path, encoding="utf-8") as f:
    chunks = json.load(f)
print("读取 chunk 数：", len(chunks))

# ===== 步骤3：幂等清理（按 item_name）=====
item_name = chunks[0].get("item_name", "")
print("幂等清理 item_name：", item_name)
client.delete(COLLECTION, filter=f'item_name == "{item_name}"')

# ===== 步骤4：组装数据并插入（不传 sparse_vector、不传 chunk_id）=====
data_to_insert = []
for c in chunks:
    row = {
        "content": c.get("content", ""),
        "title": c.get("title", ""),
        "parent_title": c.get("parent_title", ""),
        "part": c.get("part", 0),
        "file_title": c.get("file_title", ""),
        "item_name": c.get("item_name", ""),
        "dense_vector": c["dense_vector"],
    }
    data_to_insert.append(row)

result = client.insert(COLLECTION, data=data_to_insert)
print("insert_count：", result.get("insert_count"))
print("返回 ids：", result.get("ids"))

# 回填 chunk_id
ids = result.get("ids", [])
for idx, c in enumerate(chunks):
    c["chunk_id"] = str(ids[idx])

# ===== 验证 =====
# MilvusClient 用 count(*) 聚合统计条数（没有 num_entities 方法）
cnt = client.query(
    COLLECTION,
    filter=f'item_name == "{item_name}"',
    output_fields=["count(*)"]
)
print("该 item_name 条数：", cnt[0].get("count(*)"))

# 查回记录，确认 sparse_vector 已由 Milvus 自动生成
sample = client.query(
    COLLECTION,
    filter=f'item_name == "{item_name}"',
    output_fields=["chunk_id", "item_name", "sparse_vector"],
    limit=3
)
for s in sample:
    sv = s.get("sparse_vector")
    print("chunk_id:", s.get("chunk_id"),
          "| sparse_vector 非空:", bool(sv),
          "| 稀疏词项数:", len(sv) if sv else 0)

