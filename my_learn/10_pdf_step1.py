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


class NodePDFToMD(BaseNode):
    name = "node_pdf_to_md"

    def process(self, state):
        pdf_obj, out_obj = self.step_1(state)
        print("校验通过：")
        print("  PDF ：", pdf_obj.name, "| 存在:", pdf_obj.exists())
        print("  输出目录：", out_obj, "| 已就绪:", out_obj.exists())
        return state

    def step_1(self, state):
        pdf_path = state.get("pdf_path")
        file_dir = state.get("file_dir")
        if not pdf_path:
            raise ValueError("pdf_path 不能为空")
        if not file_dir:
            raise ValueError("file_dir 不能为空")

        pdf_obj = Path(pdf_path)
        out_obj = Path(file_dir)
        if not pdf_obj.exists():
            raise FileNotFoundError(f"PDF 不存在：{pdf_obj}")

        if not out_obj.exists():
            print("输出目录不存在，自动创建")
            out_obj.mkdir(parents=True, exist_ok=True)
        return pdf_obj, out_obj


if __name__ == "__main__":
    state = {
        "pdf_path": r"D:\浏览器下载\基于深度卷积神经网络的协作频谱感知方法.pdf",
        "file_dir": r"D:\RAG\YUANLAI\my_learn\output"
    }
    NodePDFToMD()(state)
