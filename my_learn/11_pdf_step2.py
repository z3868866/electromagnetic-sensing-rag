import os
import time
from pathlib import Path
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

# 加载项目根目录的 .env
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
        pdf_obj = Path(state["pdf_path"])
        zip_url = self.step_2(pdf_obj)
        state["full_zip_url"] = zip_url
        return state

    def step_2(self, pdf_obj):
        token = os.getenv("MINERU_API_TOKEN")
        base_url = os.getenv("MINERU_BASE_URL")
        if not token or not base_url:
            raise ValueError("MinerU 配置缺失：检查 .env")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }

        # —— 1. 申请上传链接 ——
        url = f"{base_url}/file-urls/batch"
        body = {"files": [{"name": pdf_obj.name}], "model_version": "vlm"}
        resp = requests.post(url, headers=headers, json=body)
        if resp.status_code != 200:                                  # 校验1：HTTP层
            raise RuntimeError(f"申请失败 {resp.status_code}: {resp.text}")
        data = resp.json()
        if data.get("code") != 0:                                   # 校验2：业务码
            raise RuntimeError(f"申请业务失败: {data}")
        signed_url = data["data"]["file_urls"][0]
        batch_id = data["data"]["batch_id"]
        print("已获取上传链接，batch_id:", batch_id)

        # —— 2. PUT 上传 PDF 到 OSS ——
        with open(pdf_obj, "rb") as f:
            up = requests.put(signed_url, data=f)
        if up.status_code != 200:
            raise RuntimeError(f"上传失败 {up.status_code}")
        print("PDF 上传成功，等待解析…")

        # —— 3. 轮询任务状态 ——
        poll_url = f"{base_url}/extract-results/batch/{batch_id}"
        start = time.time()
        timeout_seconds, interval = 600, 5
        while True:
            elapsed = time.time() - start
            if elapsed > timeout_seconds:
                raise TimeoutError("轮询超时")

            try:
                pr = requests.get(poll_url, headers=headers, timeout=10)
            except Exception as e:
                print("网络异常，5秒后重试:", e)
                time.sleep(interval)
                continue

            if pr.status_code != 200:                               # 校验1：HTTP层
                raise RuntimeError(f"轮询 HTTP 失败 {pr.status_code}")
            pj = pr.json()
            if pj.get("code") != 0:                                 # 校验2：业务码
                raise RuntimeError(f"轮询业务失败: {pj}")

            item = pj["data"]["extract_result"][0]
            state_text = item["state"]
            if state_text == "done":                                # 校验3：state==done
                print(f"解析完成，耗时 {int(elapsed)}s")
                return item["full_zip_url"]
            elif state_text == "failed":
                raise RuntimeError(f"解析失败: {item.get('err_msg')}")
            else:
                print(f"处理中… 已耗时 {int(elapsed)}s，状态: {state_text}")
                time.sleep(interval)


if __name__ == "__main__":
    state = {"pdf_path": r"D:\浏览器下载\基于深度卷积神经网络的协作频谱感知方法.pdf"}
    NodePDFToMD()(state)
