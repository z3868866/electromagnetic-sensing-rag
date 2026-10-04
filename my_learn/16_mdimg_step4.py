import os
import re
import base64
import time
from collections import deque
from pathlib import Path
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject

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
        targets = self.step_2(md_content, images_dir)          # 全量，去掉 [:3]
        summaries = self.step_3(md_obj.stem, targets)
        new_md = self.step_4(md_obj.stem, targets, summaries, md_content)
        new_path = self.step_5(state["md_path"], new_md)

        state["md_content"] = new_md
        state["md_path"] = new_path
        print("新文件：", new_path)
        print("字符数：", len(new_md), "| 处理图片：", len(targets))
        return state


    def step_1(self, state):
        md_obj = Path(state["md_path"])
        md_content = md_obj.read_text(encoding="utf-8")
        images_dir = md_obj.parent / "images"
        return md_content, md_obj, images_dir

    def step_2(self, md_content, images_dir):
        exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
        out = []
        for f in os.listdir(images_dir):
            if os.path.splitext(f)[1].lower() not in exts:
                continue
            ctx = self.find_image(md_content, f)
            if ctx:
                out.append((f, str(images_dir / f), ctx))
        return out

    def find_image(self, md, image_file, n=100):
        m = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)").search(md)
        if not m:
            return None
        s, e = m.span()
        return md[max(0, s-n):s], md[e:min(len(md), e+n)]

    def step_3(self, doc_stem, targets):
        summaries, dq = {}, deque()
        for f, p, ctx in targets:
            self.rate_limit(dq, 10, 60)
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("utf-8")
            chat = ChatOpenAI(
                model=os.getenv("VL_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")))
            msgs = [{"role": "user", "content": [
                {"type": "text", "text": f'这是"{doc_stem}"中的图片，上文"{ctx[0][-60:]}",下文"{ctx[1][:60]}",用中文简要总结，50字以内。'},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
            summaries[f] = chat.invoke(msgs).content.strip().replace("\n", "")
        return summaries

    def rate_limit(self, dq, mx, win):
        now = time.time()
        while dq and now - dq[0] >= win:
            dq.popleft()
        if len(dq) >= mx:
            st = win - (now - dq[0])
            if st > 0:
                print(f"  触发限流，等待 {st:.1f}s"); time.sleep(st)
        dq.append(time.time())

    # —— Step 4 相关 ——
    def get_minio(self):
        return Minio(
            os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False)

    def step_4(self, doc_stem, targets, summaries, md_content):
        client = self.get_minio()
        bucket = os.getenv("MINIO_BUCKET_NAME")
        upload_dir = f"{os.getenv('MINIO_IMG_DIR')}/{doc_stem}".replace(" ", "")

        self.clean_minio(client, bucket, upload_dir)

        urls = {}
        for img_file, img_path, _ in targets:
            url = self.upload(client, bucket, img_path, f"{upload_dir}/{img_file}")
            if url:
                urls[img_file] = url

        image_info = {}
        for img_file, s in summaries.items():
            if u := urls.get(img_file):
                image_info[img_file] = (s, u)
        return self.replace_md(md_content, image_info)

    def clean_minio(self, client, bucket, prefix):
        objs = client.list_objects(bucket, prefix=prefix, recursive=True)
        delete_list = [DeleteObject(o.object_name) for o in objs]
        if delete_list:
            for err in client.remove_objects(bucket, delete_list):
                print("删除错误：", err)

    def upload(self, client, bucket, local_path, object_name):
        try:
            ext = os.path.splitext(local_path)[1][1:]
            client.fput_object(bucket, object_name, local_path, content_type=f"image/{ext}")
            return f"http://{os.getenv('MINIO_ENDPOINT')}/{bucket}/{object_name}"
        except Exception as e:
            print("上传失败：", e)
            return None

    def replace_md(self, md_content, image_info):
        for img_file, (summary, url) in image_info.items():
            pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_file) + r".*?\)")
            md_content = pattern.sub(lambda m: f"![{summary}]({url})", md_content)
        return md_content
    
    def step_5(self, origin_md_path, md_content):
        new_name = os.path.splitext(origin_md_path)[0] + "_new.md"
        with open(new_name, "w", encoding="utf-8") as f:
            f.write(md_content)
        return new_name


if __name__ == "__main__":
    state = {
        "md_path": r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法.md"
    }
    NodeMDImg()(state)
