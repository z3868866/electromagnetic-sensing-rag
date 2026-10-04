import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:\RAG\YUANLAI\.env")


def get_embedding_client():
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        dimensions=int(os.getenv("EMBEDDING_DIM", "1024")),
        check_embedding_ctx_length=False   # 新增：不做 token 化，直接发原始字符串
    )



def generate_dense_embeddings(texts):
    client = get_embedding_client()
    return client.embed_documents(texts)


def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    texts = ["协作频谱感知", "频谱信号检测", "今天天气很好"]
    vecs = generate_dense_embeddings(texts)

    print("返回条数：", len(vecs))
    print("向量维度：", len(vecs[0]))
    print("第1条前5个值：", [round(x, 4) for x in vecs[0][:5]])
    print("=" * 30)
    print("语义相似度验证（预期：相近主题高、无关主题低）")
    print("  频谱感知 vs 频谱检测：", round(cosine(vecs[0], vecs[1]), 4))
    print("  频谱感知 vs 天气很好：", round(cosine(vecs[0], vecs[2]), 4))
