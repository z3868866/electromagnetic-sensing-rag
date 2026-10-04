import json
from pathlib import Path
from abc import ABC, abstractmethod


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


class NodeItemNameRecognition(BaseNode):
    name = "node_item_name_recognition"
    K = 5            # 取前5个切片
    MAX_CHARS = 2500  # 上下文总字符上限

    def process(self, state):
        file_title, chunks = self.step_1(state)
        context = self.step_2(chunks)
        print("file_title：", file_title)
        print("上下文长度：", len(context))
        print("=" * 30, "上下文前600字", "=" * 30)
        print(context[:600])
        return state

    def step_1(self, state):
        file_title = state.get("file_title", "")
        if not file_title:
            raise ValueError("file_title 为空")
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks 为空或非列表")
        return file_title, chunks

    def step_2(self, chunks):
        parts = []
        total = 0
        for idx, chunk in enumerate(chunks[:self.K], start=1):
            t = chunk.get("title", "").strip()
            c = chunk.get("content", "").strip()
            piece = f"【切片{idx}】\n标题：{t}\n内容：{c}"
            parts.append(piece)
            total += len(piece)
            if total > self.MAX_CHARS:
                print(f"累计 {total} 字符已超限，停止拼接")
                break
        context = "\n\n".join(parts).strip()
        return context[:self.MAX_CHARS]


if __name__ == "__main__":
    chunks_path = Path(r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks.json")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    state = {
        "file_title": "基于深度卷积神经网络的协作频谱感知方法",
        "chunks": chunks
    }
    NodeItemNameRecognition()(state)
