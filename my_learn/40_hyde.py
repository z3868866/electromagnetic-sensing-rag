import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv(r"D:\RAG\YUANLAI\.env")
COLLECTION = "kb_chunks"
client = MilvusClient(uri=os.getenv("MILVUS_URL"))
embedder = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
    check_embedding_ctx_length=False)
llm = ChatOpenAI(
    model=os.getenv("LLM_DEFAULT_MODEL", "qwen-flash"),
    api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0.1)

HYDE_PROMPT = """请针对下面的问题，直接给出一段陈述性的"假设答案"（150~300字）。
要求：
1. 不要说"我不确定"，也不要提问，直接以陈述语气写出一个可能的答案；
2. 答案即使不完全准确也没关系，但语言风格要像一篇技术文档中的段落；
3. 包含问题涉及的关键术语。

问题：{query}"""


def generate_hyde(query):
    messages = [SystemMessage(content="你是技术文档撰写助手。"),
                HumanMessage(content=HYDE_PROMPT.format(query=query))]
    return llm.invoke(messages).content.strip()


def search_with_hyde(hyde_doc, item_names, recall=8):
    expr = None
    if item_names:
        quoted = ", ".join(f'"{n}"' for n in item_names)
        expr = f"item_name in [{quoted}]"
    dense_vec = embedder.embed_query(hyde_doc)
    dense_req = AnnSearchRequest(data=[dense_vec], anns_field="dense_vector",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[hyde_doc], anns_field="sparse_vector",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)
    res = client.hybrid_search(
        collection_name=COLLECTION, reqs=[dense_req, sparse_req],
        ranker=WeightedRanker(0.8, 0.2), limit=recall,
        output_fields=["chunk_id", "title", "parent_title", "content", "item_name"],
        consistency_level="Strong")
    chunks = []
    for h in res[0]:
        entity = h.get("entity", {})
        entity["score"] = h.get("distance")
        chunks.append(entity)
    return chunks


if __name__ == "__main__":
    query = "基于深度卷积神经网络的协作频谱感知方法的网络结构是什么？"
    item_names = ["基于深度卷积神经网络的协作频谱感知方法"]
    print("=== Step1 生成假设答案 ===")
    hyde_doc = generate_hyde(query)
    print(hyde_doc)
    print("\n=== Step2 用假设答案检索 ===")
    chunks = search_with_hyde(hyde_doc, item_names, recall=8)
    print("召回切片数：", len(chunks))
    for i, c in enumerate(chunks, 1):
        print(f"{i}. score={round(c['score'],3)} | {str(c.get('title',''))[:30]}")
