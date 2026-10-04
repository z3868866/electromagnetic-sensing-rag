import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")

base = os.getenv("MINERU_BASE_URL")
token = os.getenv("MINERU_API_TOKEN")
pdf = Path(r"D:\RAG\YUANLAI\my_learn\test_5g.pdf")

headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

# 1. 申请上传链接（短文件名；不传 model_version，走默认 pipeline 管线）
body = {"files": [{"name": pdf.name}]}
r = requests.post(f"{base}/file-urls/batch", headers=headers, json=body, timeout=20)
print("申请 HTTP:", r.status_code)
d = r.json()
print("code:", d.get("code"))
signed = d["data"]["file_urls"][0]
bid = d["data"]["batch_id"]
print("batch_id:", bid)

# 2. 上传
with open(pdf, "rb") as f:
    up = requests.put(signed, data=f)
print("上传 HTTP:", up.status_code)

# 3. 密集轮询约 120 秒，观察状态是否进入 running/done
poll = f"{base}/extract-results/batch/{bid}"
start = time.time()
last = None
while time.time() - start < 120:
    time.sleep(5)
    pr = requests.get(poll, headers=headers, timeout=15).json()
    er = pr["data"]["extract_result"][0]
    st = er.get("state")
    if st != last:
        print(f"[{int(time.time()-start)}s] 状态变化: {last} -> {st} | keys={list(er.keys())}")
        last = st
    else:
        print(f"[{int(time.time()-start)}s] 仍为 {st}")
    if st in ("done", "failed"):
        print("err_msg:", er.get("err_msg"))
        if st == "done":
            print("zip:", er.get("full_zip_url"))
        break
