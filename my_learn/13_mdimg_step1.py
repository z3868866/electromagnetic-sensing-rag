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


class NodeMDImg(BaseNode):
    name = "node_md_img"

    def process(self, state):
        md_content, md_obj, images_dir = self.step_1(state)
        print("MD 文件：", md_obj.name)
        print("MD 字符数：", len(md_content))
        print("图片目录：", images_dir, "| 存在:", images_dir.exists())
        if images_dir.exists():
            print("目录内文件数：", len(list(images_dir.iterdir())))
        return state

    def step_1(self, state):
        md_path = state.get("md_path")
        if not md_path:
            raise ValueError("md_path 不能为空")
        md_obj = Path(md_path)
        if not md_obj.exists():
            raise FileNotFoundError(f"MD 不存在：{md_obj}")

        md_content = md_obj.read_text(encoding="utf-8")   # Path 自带方法，等价 open+read
        images_dir = md_obj.parent / "images"
        return md_content, md_obj, images_dir


if __name__ == "__main__":
    state = {
        "md_path": r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法.md"
    }
    NodeMDImg()(state)
