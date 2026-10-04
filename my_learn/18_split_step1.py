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


class NodeDocumentSplit(BaseNode):
    name = "node_document_split"

    def process(self, state):
        content, title = self.step_1(state)
        print("file_title：", title)
        print("标准化后字符数：", len(content))
        print("仍含 CRLF：", "\r\n" in content, "| 仍含 CR：", "\r" in content)
        return state

    def step_1(self, state):
        file_title = state.get("file_title", "")
        if not file_title:
            raise ValueError("file_title 缺失")
        md_content = state.get("md_content", "")
        if not md_content:
            raise ValueError("md_content 缺失")
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title


if __name__ == "__main__":
    new_md = Path(r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法_new.md")
    state = {
        "file_title": new_md.stem.replace("_new", ""),
        "md_content": new_md.read_text(encoding="utf-8")
    }
    NodeDocumentSplit()(state)
