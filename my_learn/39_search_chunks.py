import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:\RAG\YUANLAI\.env")
COLLECTION = "kb_chunks"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))
embedder = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False
)


def search_kb(rewritten_query, item_names, recall=10):
    # 过滤条件：限定在已确认主体范围内
    expr = None
    if item_names:
        quoted = ", ".join(f'"{n}"' for n in item_names)
        expr = f"item_name in [{quoted}]"
        print("过滤条件：", expr)

    dense_vec = embedder.embed_query(rewritten_query)
    dense_req = AnnSearchRequest(data=[dense_vec], anns_field="dense_vector",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[rewritten_query], anns_field="sparse_vector",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)

    res = client.hybrid_search(
        collection_name=COLLECTION,
        reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.8, 0.2),
        limit=recall,
        output_fields=["chunk_id", "title", "parent_title", "content", "item_name"],
        consistency_level="Strong"
    )
    chunks = []
    for h in res[0]:
        entity = h.get("entity", {})
        entity["score"] = h.get("distance")
        chunks.append(entity)
    return chunks


if __name__ == "__main__":
    state = {
        "rewritten_query": "基于深度卷积神经网络的协作频谱感知方法的网络结构是什么？",
        "item_names": ["基于深度卷积神经网络的协作频谱感知方法"]
    }
    chunks = search_kb(state["rewritten_query"], state["item_names"], recall=8)
    print("召回切片数：", len(chunks))
    for i, c in enumerate(chunks, 1):
        print(f"{i}. score={round(c['score'],3)} | {str(c.get('title',''))[:30]} | {len(c.get('content',''))}字")
