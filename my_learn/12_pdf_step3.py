import os
import shutil
import time
import zipfile
from pathlib import Path
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

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


class NodePDFToMD(BaseNode):
    name = "node_pdf_to_md"

    def process(self, state):
        pdf_obj, out_obj = self.step_1(state)
        zip_url = self.step_2(pdf_obj)
        md_path = self.step_3(zip_url, out_obj, pdf_obj.stem)

        # 读出 Markdown 内容，写入 state
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_path"] = str(md_path)
        state["md_content"] = md_content

        print("md 字符数：", len(md_content))
        print("前 300 字预览：\n", md_content[:300])
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
            out_obj.mkdir(parents=True, exist_ok=True)
        return pdf_obj, out_obj

    def step_2(self, pdf_obj):
        token = os.getenv("MINERU_API_TOKEN")
        base_url = os.getenv("MINERU_BASE_URL")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

        url = f"{base_url}/file-urls/batch"
        body = {"files": [{"name": pdf_obj.name}], "model_version": "vlm"}
        resp = requests.post(url, headers=headers, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"申请失败 {resp.status_code}")
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"申请业务失败: {data}")
        signed_url = data["data"]["file_urls"][0]
        batch_id = data["data"]["batch_id"]

        with open(pdf_obj, "rb") as f:
            up = requests.put(signed_url, data=f)
        if up.status_code != 200:
            raise RuntimeError("上传失败")

        poll_url = f"{base_url}/extract-results/batch/{batch_id}"
        start = time.time()
        while True:
            if time.time() - start > 600:
                raise TimeoutError("轮询超时")
            try:
                pr = requests.get(poll_url, headers=headers, timeout=10)
            except Exception:
                time.sleep(5); continue
            pj = pr.json()
            item = pj["data"]["extract_result"][0]
            if item["state"] == "done":
                return item["full_zip_url"]
            elif item["state"] == "failed":
                raise RuntimeError("解析失败")
            time.sleep(5)

    def step_3(self, zip_url, out_obj, pdf_stem):
        # 1. 下载 ZIP
        resp = requests.get(zip_url)
        if resp.status_code != 200:
            raise RuntimeError(f"ZIP 下载失败 {resp.status_code}")
        zip_path = out_obj / f"{pdf_stem}_result.zip"
        with open(zip_path, "wb") as f:
            f.write(resp.content)
        print("ZIP 已下载：", zip_path.name)

        # 2. 准备解压目录（幂等：已存在先删除）
        extract_dir = out_obj / pdf_stem
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        # 3. 解压
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print("已解压到：", extract_dir)

        # 4. 找到 full.md 并重命名为与 PDF 同名
        full_md = extract_dir / "full.md"
        if not full_md.exists():
            raise FileNotFoundError("解压结果中没有 full.md")
        new_md = full_md.with_name(f"{pdf_stem}.md")
        full_md.rename(new_md)
        print("MD 文件：", new_md.name)
        return new_md


if __name__ == "__main__":
    state = {
        "pdf_path": r"D:\浏览器下载\基于深度卷积神经网络的协作频谱感知方法.pdf",
        "file_dir": r"D:\RAG\YUANLAI\my_learn\output"
    }
    NodePDFToMD()(state)
