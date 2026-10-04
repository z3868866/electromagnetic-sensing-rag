import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:\RAG\YUANLAI\.env")
client = MilvusClient(uri=os.getenv("MILVUS_URL"))
embedder = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False)


def search_with_weight(query, item_names, wd, ws, recall=8):
    quoted = ", ".join(f'"{n}"' for n in item_names)
    expr = f"item_name in [{quoted}]"
    dense_vec = embedder.embed_query(query)
    dense_req = AnnSearchRequest(data=[dense_vec], anns_field="dense_vector",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[query], anns_field="sparse_vector",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)
    res = client.hybrid_search(
        collection_name="kb_chunks", reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(wd, ws), limit=recall,
        output_fields=["title"], consistency_level="Strong")
    return [(h.get("entity", {}).get("title"), round(h.get("distance"), 3)) for h in res[0]]


if __name__ == "__main__":
    query = "基于深度卷积神经网络的协作频谱感知方法的网络结构是什么？"
    item_names = ["基于深度卷积神经网络的协作频谱感知方法"]
    for wd, ws in [(0.9, 0.1), (0.5, 0.5), (0.1, 0.9)]:
        print(f"=== dense {wd} / sparse {ws} ===")
        for title, score in search_with_weight(query, item_names, wd, ws)[:5]:
            print(f"  {score} | {str(title)[:32]}")
