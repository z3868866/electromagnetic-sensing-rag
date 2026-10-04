import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:\RAG\YUANLAI\.env")
COLLECTION = "kb_item_names"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))

embedder = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False
)


def search_one(name: str, limit=5):
    # 稠密路：传 embedding 向量
    dense_vec = embedder.embed_query(name)
    dense_req = AnnSearchRequest(
        data=[dense_vec], anns_field="dense_vector",
        param={"metric_type": "COSINE"}, limit=limit
    )
    # 稀疏路：传原始文本，BM25 Function 自动转换
    sparse_req = AnnSearchRequest(
        data=[name], anns_field="sparse_vector",
        param={"metric_type": "BM25"}, limit=limit
    )
    res = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.8, 0.2),
        limit=limit,
        output_fields=["item_name"],
        consistency_level="Strong"
    )
    return [{"item_name": h.get("entity", {}).get("item_name"),
             "score": h.get("distance")} for h in res[0]]


def align(extracted_names):
    confirmed, options = [], []
    for extracted in extracted_names:
        matches = search_one(extracted)
        print(f"提取名「{extracted}」→",
              [(m["item_name"][:18], round(m["score"], 3)) for m in matches])

        high = [m for m in matches if m["score"] > 0.85]
        mid = [m for m in matches if m["score"] >= 0.6]
        if high:  # 高置信：优先与原名相同，否则取最高分
            picked = next((m for m in high if m["item_name"] == extracted), high[0])
            confirmed.append(picked["item_name"])
        elif mid:
            options.extend(m["item_name"] for m in mid[:5])

    return {"confirmed": list(set(confirmed)), "options": list(set(options))}


if __name__ == "__main__":
    for names in [
        ["基于深度卷积神经网络的协作频谱感知方法"],   # 完整名 → 预期确认
        ["DCNN"],                                   # 简称 → 看实际分数
        ["今天天气"]                                # 无关 → 预期无匹配丢弃
    ]:
        print("对齐结果：", align(names))
        print("-" * 55)
