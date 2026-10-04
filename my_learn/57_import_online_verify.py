import os
import sys
import json

# 让脚本能 import 项目根下的包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client, online_hybrid_search, escape_milvus_string
from processor.import_processor.nodes.node_item_name_recognition import NodeItemNameRecognition
from processor.import_processor.nodes.node_bge_embedding import NodeBGEEmbedding
from processor.import_processor.nodes.node_import_milvus import NodeImportMilvus

CHUNKS_JSON = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks.json"

# 1. 载入切分产物（跳过 MinerU PDF 解析）
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

file_title = chunks[0]["file_title"]
state = {"task_id": "verify_001", "file_title": file_title, "chunks": chunks}
print("载入切片数：", len(chunks))

# 2. 主体识别（在线 LLM）：回填每个 chunk 的 item_name，并写 kb_item_names
state = NodeItemNameRecognition()(state)
print("识别 item_name：", state.get("item_name"))

# 3. 在线向量化（text-embedding-v4）：为每个 chunk 绑定 dense_vector
state = NodeBGEEmbedding()(state)
print("全部含 dense_vector：", all("dense_vector" in c for c in state["chunks"]),
      "| 维度：", len(state["chunks"][0]["dense_vector"]))

# 4. 入库（幂等删旧 -> 插入标量+dense；sparse_vector 由 BM25 Function 自动生成）
state = NodeImportMilvus()(state)

# 5. 校验：按 file_title 查条数
client = get_milvus_client()
ft = escape_milvus_string(file_title)
rows = client.query(
    milvus_config.chunks_collection,
    filter=f"file_title=='{ft}'",
    output_fields=["title"],
    limit=200,
)
print("该 file_title 在库条数：", len(rows))

# 6. 混合检索校验（dense COSINE + sparse BM25 自动）
q = "DCNN 如何通过捷径连接缓解梯度消失"
res = online_hybrid_search(
    client,
    milvus_config.chunks_collection,
    q,
    limit=5,
    output_fields=["title", "item_name"],
)
print("查询：", q)
for hit in res[0]:
    ent = hit.get("entity", {})
    print(f"  score={round(hit.get('distance', 0), 4)} | {str(ent.get('title',''))[:34]}")
