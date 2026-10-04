import os
import re
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
        if not images_dir.exists():
            print("无图片目录，跳过")
            return state
        targets = self.step_2(md_content, images_dir)
        print(f"扫描完成：images 共 {len(list(images_dir.iterdir()))} 个文件，MD 实际引用 {len(targets)} 张")
        if targets:
            f, p, (pre, post) = targets[0]
            print("第一张：", f)
            print("  前文：", pre[-40:].replace("\n", " "))
            print("  后文：", post[:40].replace("\n", " "))
        state["target_images"] = targets
        return state

    def step_1(self, state):
        md_obj = Path(state["md_path"])
        md_content = md_obj.read_text(encoding="utf-8")
        images_dir = md_obj.parent / "images"
        return md_content, md_obj, images_dir

    def step_2(self, md_content, images_dir):
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        target_images = []
        for image_file in os.listdir(images_dir):
            file_ext = os.path.splitext(image_file)[1].lower()
            if file_ext not in image_extensions:
                continue
            context = self.find_image(md_content, image_file)
            if not context:
                continue
            target_images.append((image_file, str(images_dir / image_file), context))
        return target_images

    def find_image(self, md_content, image_file, context_len=100):
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        return pre_text, post_text


if __name__ == "__main__":
    state = {
        "md_path": r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法.md"
    }
    NodeMDImg()(state)
