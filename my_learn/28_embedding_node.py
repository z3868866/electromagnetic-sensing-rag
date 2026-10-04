import json
import os
from pathlib import Path
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv(r"D:\RAG\YUANLAI\.env")


def get_embedding_client():
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        dimensions=int(os.getenv("EMBEDDING_DIM", "1024")),
        check_embedding_ctx_length=False
    )


def generate_dense_embeddings(texts):
    return get_embedding_client().embed_documents(texts)


class BaseNode(ABC):
    name = "base_node"
    def __init__(self):
        if self.name == "base_node":
            raise ValueError("必须覆盖 name")
    def __call__(self, state):
        print(f"--- {self.name} 开始 ---")
        state = self.process(state)
        print(f"--- {self.name} 完成 ---")
        return state
    @abstractmethod
    def process(self, state):
        pass


class NodeBGEEmbedding(BaseNode):
    name = "node_bge_embedding"
    batch_size = 5

    def process(self, state):
        chunks = self.step_1(state)
        output = self.step_2(chunks)
        state["chunks"] = output
        return state

    def step_1(self, state):
        chunks = state.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks 为空或非列表，无法向量化")
        return chunks

    def step_2(self, chunks):
        output = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            texts = [f"{c['item_name']}\n{c['content']}" for c in batch]

            vectors = generate_dense_embeddings(texts)

            for j, c in enumerate(batch):
                c["dense_vector"] = vectors[j]
                output.append(c)

            print(f"已处理 {min(i + self.batch_size, len(chunks))}/{len(chunks)}")
        return output


if __name__ == "__main__":
    base = Path(r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法")
    chunks = json.loads((base / "chunks.json").read_text(encoding="utf-8"))

    # 模拟主体识别节点的输出：为每个 chunk 打 item_name 标签
    item_name = "基于深度卷积神经网络的协作频谱感知方法"
    for c in chunks:
        c["item_name"] = item_name

    state = {"chunks": chunks}
    result = NodeBGEEmbedding()(state)
    rc = result["chunks"]

    print("=" * 30)
    print("总 chunk 数：", len(rc))
    print("全部含 dense_vector：", all("dense_vector" in c for c in rc))
    print("向量维度：", len(rc[0]["dense_vector"]))

    out = base / "chunks_with_vector.json"
    out.write_text(json.dumps(rc, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已备份：", out)
