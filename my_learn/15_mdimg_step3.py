import os
import base64
import time
from collections import deque
from pathlib import Path
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(r"D:\RAG\YUANLAI\.env")


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
        targets = self.step_2(md_content, images_dir)
        targets = targets[:3]          # 学习阶段只处理前3张；处理全部请删除这行
        print(f"本次处理 {len(targets)} 张图片")
        summaries = self.step_3(md_obj.stem, targets)
        for f, s in summaries.items():
            print("•", f[:16], "→", s)
        state["summaries"] = summaries
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
        import re
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
        match = pattern.search(md_content)
        if not match:
            return None
        start, end = match.span()
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        return pre_text, post_text

    def step_3(self, doc_stem, targets):
        summaries = {}
        req_deque = deque()
        for img_file, img_path, context in targets:
            self.rate_limit(req_deque, max_requests=2, window_seconds=30)
            print(f"开始生成摘要：{img_file[:16]}…")
            summaries[img_file] = self.summarize_image(img_path, doc_stem, context)
        return summaries

    def summarize_image(self, image_path, root_folder, context):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")

        chat = ChatOpenAI(
            model=os.getenv("VL_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")),
        )
        messages = [{"role": "user", "content": [
            {"type": "text", "text": f'这是"{root_folder}"文件中的一张图片，图片上文为"{context[0][-80:]}",下文为"{context[1][:80]}",请用中文简要总结图片内容，用于Markdown图片标题，50字以内。'},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        ]}]
        resp = chat.invoke(messages)
        return resp.content.strip().replace("\n", "")

    def rate_limit(self, req_times, max_requests, window_seconds=60):
        now = time.time()
        while req_times and now - req_times[0] >= window_seconds:
            req_times.popleft()
        if len(req_times) >= max_requests:
            sleep_t = window_seconds - (now - req_times[0])
            if sleep_t > 0:
                print(f"  触发限流，等待 {sleep_t:.1f} 秒…")
                time.sleep(sleep_t)
                now = time.time()
                while req_times and now - req_times[0] >= window_seconds:
                    req_times.popleft()
        req_times.append(now)


if __name__ == "__main__":
    state = {
        "md_path": r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法.md"
    }
    NodeMDImg()(state)
