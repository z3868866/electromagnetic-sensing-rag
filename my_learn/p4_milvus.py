"""Milvus 最小场景（纯 dense 向量）：
建集合(schema) -> 建向量索引 -> 插入几句话的 embedding -> 加载 -> 单向量语义检索。
集合为练习集合，结尾保留，便于你改查询反复试。
"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import MilvusClient, DataType

load_dotenv()  # 从项目根 .env 读取 OPENAI_API_KEY / OPENAI_API_BASE

# ---------- 1. Embedding 客户端（百炼 OpenAI 兼容模式）----------
emb = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
             base_url=os.getenv("OPENAI_API_BASE"))
DIM = 1024

def embed_texts(texts):
    resp = emb.embeddings.create(
        model="text-embedding-v4",
        input=texts,
        dimensions=DIM,
    )
    return [d.embedding for d in resp.data]

# ---------- 2. 连接 Milvus ----------
mc = MilvusClient(uri=os.getenv("MILVUS_URL", "http://localhost:19530"))
COLL = "practice_vectors"
if mc.has_collection(COLL):
    mc.drop_collection(COLL)  # 练习用：每次重跑先删旧集合

# ---------- 3. 定义 schema（字段结构）----------
schema = mc.create_schema(auto_id=True, enable_dynamic_field=False)
schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)  # 主键，自动生成
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)            # 稠密向量字段，必须指定维度
schema.add_field("text", DataType.VARCHAR, max_length=500)            # 标量字段，存原文用于展示

# ---------- 4. 建向量索引 ----------
index_params = mc.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",     # 让 Milvus 自动选择合适的 ANN 索引
    metric_type="COSINE",       # 相似度度量：余弦，越大越相似
)
mc.create_collection(COLL, schema=schema, index_params=index_params)

# ---------- 5. 插入数据（先把文本变成向量）----------
texts = [
    "DCNN 通过添加捷径连接缓解梯度消失",
    "残差学习使梯度能够直接回传到浅层网络",
    "频谱感知用于判断某频段是否被信号占用",
    "今天天气晴朗，适合出门散步",
]
vectors = embed_texts(texts)
rows = [{"vector": v, "text": t} for v, t in zip(vectors, texts)]
mc.insert(COLL, rows)

# ---------- 6. 加载集合到内存（检索前必须 load_collection）----------
mc.load_collection(COLL)

# ---------- 7. 语义检索 ----------
query = "网络层数太深出现梯度消失该怎么解决"
qvec = embed_texts([query])[0]
result = mc.search(
    collection_name=COLL,
    data=[qvec],
    limit=4,
    output_fields=["text"],
)
print("查询：", query)
print("检索结果（distance=COSINE，越接近 1 越语义相似）：")
for hit in result[0]:
    print(f"  {round(hit['distance'], 4)}  | {hit['entity']['text']}")
