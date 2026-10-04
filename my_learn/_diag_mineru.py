import os
import json
import requests
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")

base = os.getenv("MINERU_BASE_URL")
tok = os.getenv("MINERU_API_TOKEN")
bids = [
    "aba4c7c9-6496-49d8-9bb0-c5ce890965ea",
    "52bec653-8d55-4116-9461-43124acc0532",
]
for bid in bids:
    r = requests.get(
        f"{base}/extract-results/batch/{bid}",
        headers={"Authorization": f"Bearer {tok}"},
        timeout=15,
    )
    d = r.json()
    er = d["data"]["extract_result"][0]
    print("=" * 50)
    print("bid:", bid)
    print("state:", er.get("state"), "| err_msg:", er.get("err_msg"))
    print("keys:", list(er.keys()))
