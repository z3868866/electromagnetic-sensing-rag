"""MinerU 最小场景（自包含，requests 直接调）：
申请上传链接 -> PUT 上传 PDF -> 轮询任务状态 -> 下载 ZIP 并解压 -> 看 Markdown。
"""
import os
import time
import zipfile
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()
BASE = os.getenv("MINERU_BASE_URL")     # MinerU API 根地址
TOKEN = os.getenv("MINERU_API_TOKEN")   # MinerU Token
PDF = r"my_learn/test_5g.pdf"
OUT = r"my_learn/p5_output"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}",
}

# ---------- Step1：申请上传链接（告诉服务端要传哪个文件）----------
print("=== Step1 申请上传链接 ===")
r = requests.post(
    f"{BASE}/file-urls/batch",
    headers=headers,
    json={"files": [{"name": Path(PDF).name}], "model_version": "vlm"},
)
print("HTTP:", r.status_code)
j = r.json()
print("业务 code:", j["code"])
batch_id = j["data"]["batch_id"]          # 任务批次 ID，后续靠它查结果
upload_url = j["data"]["file_urls"][0]    # 服务端签发的临时上传地址（指向其对象存储）
print("batch_id:", batch_id)

# ---------- Step2：把 PDF 直接 PUT 到那个临时地址 ----------
print("\n=== Step2 上传 PDF ===")
with open(PDF, "rb") as f:
    ru = requests.put(upload_url, data=f)
print("上传 HTTP:", ru.status_code)

# ---------- Step3：轮询解析状态（异步任务）----------
print("\n=== Step3 轮询任务状态 ===")
poll_url = f"{BASE}/extract-results/batch/{batch_id}"
t0 = time.time()
zip_url = None
while time.time() - t0 < 300:             # 最多等 5 分钟
    rp = requests.get(poll_url, headers=headers)
    rj = rp.json()
    er = rj["data"]["extract_result"][0]
    state = er["state"]                   # pending / running / done / failed
    print(f"[{int(time.time() - t0)}s] 状态: {state}")
    if state == "done":
        zip_url = er["full_zip_url"]      # 完成后给结果 ZIP 的下载地址
        break
    if state == "failed":
        print("解析失败：", er.get("err_msg"))
        break
    time.sleep(3)

# ---------- Step4：下载 ZIP、解压、读取 Markdown ----------
if zip_url:
    print("\n=== Step4 下载并解压 ===")
    Path(OUT).mkdir(parents=True, exist_ok=True)
    zip_path = Path(OUT) / "result.zip"
    zip_path.write_bytes(requests.get(zip_url).content)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(OUT)
    mds = list(Path(OUT).rglob("*.md"))
    print("解压得到的 Markdown：", mds)
    if mds:
        text = mds[0].read_text(encoding="utf-8")
        print(f"Markdown 长度：{len(text)} 字符")
        print("前 300 字：\n", text[:300])
