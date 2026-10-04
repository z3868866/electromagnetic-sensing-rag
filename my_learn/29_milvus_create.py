import os
from pymilvus import MilvusClient, DataType, Function, FunctionType
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")

COLLECTION = "kb_chunks"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))

# 学习环境：若集合已存在先删除，保证可重复运行
if client.has_collection(COLLECTION):
    client.drop_collection(COLLECTION)
    print("已删除旧集合")

# 1. 定义 Schema
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("content", DataType.VARCHAR, max_length=65535,
                 enable_analyzer=True, analyzer_params={"type": "chinese"})
schema.add_field("title", DataType.VARCHAR, max_length=65535)
schema.add_field("parent_title", DataType.VARCHAR, max_length=65535)
schema.add_field("part", DataType.INT8)
schema.add_field("file_title", DataType.VARCHAR, max_length=65535)
schema.add_field("item_name", DataType.VARCHAR, max_length=65535)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)

# 2. 注册 BM25 Function：content → sparse_vector
bm25_function = Function(
    name="bm25_function",
    function_type=FunctionType.BM25,
    input_field_names=["content"],
    output_field_names=["sparse_vector"]
)
schema.add_function(bm25_function)

# 3. 定义索引
index_params = client.prepare_index_params()
index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
index_params.add_index(
    field_name="sparse_vector",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25"
)


# 4. 创建集合
client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)

print("集合创建成功：", client.has_collection(COLLECTION))
desc = client.describe_collection(COLLECTION)
print("字段：", [f["name"] for f in desc["fields"]])
print("Functions：", desc.get("functions"))
