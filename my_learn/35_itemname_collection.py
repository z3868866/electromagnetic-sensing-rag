import os
from pymilvus import MilvusClient, DataType, Function, FunctionType
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:\RAG\YUANLAI\.env")
COLLECTION = "kb_item_names"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))

if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)
    print("已删除旧集合")

# 1. Schema：标准名 + 双向量（sparse 由 BM25 Function 自动生成）
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("name_id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("item_name", DataType.VARCHAR, max_length=65535,
                 enable_analyzer=True, analyzer_params={"type": "chinese"})
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)

bm25 = Function(name="bm25_function", function_type=FunctionType.BM25,
                input_field_names=["item_name"], output_field_names=["sparse_vector"])
schema.add_function(bm25)

# 2. 索引
index_params = client.prepare_index_params()
index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
print("集合创建成功")

# 3. 标准主体名向量化并插入（不传 sparse_vector）
standard_names = ["基于深度卷积神经网络的协作频谱感知方法"]
embedder = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False
)
dense_vecs = embedder.embed_documents(standard_names)

data = [{"item_name": n, "dense_vector": v}
        for n, v in zip(standard_names, dense_vecs)]
res = client.insert(COLLECTION, data=data)
print("插入条数：", res.get("insert_count"))

# 4. 验证
check = client.query(
    COLLECTION,
    filter='item_name == "基于深度卷积神经网络的协作频谱感知方法"',
    output_fields=["item_name"],
    consistency_level="Strong"
)
print("查回：", check)

