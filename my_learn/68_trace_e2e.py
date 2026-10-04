# my_learn/68_trace_e2e.py
"""
优化点⑤端到端：跑完整问答图，查回 MongoDB 中的 trace，
打印各节点 span（耗时/状态）与 RAG 关键汇总，并确认文件日志生成。
"""
import os
import sys
sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from processor.query_processor.main_graph import KBQueryWorkflow
from utils.mongo_history_utils import get_history_mongo_tool

sid = "trace_e2e_03"
wf = KBQueryWorkflow()
state = wf.run({
    "session_id": sid,
    "original_query": "DCNN 是如何缓解梯度消失问题的？",
    "is_stream": False,
})

# 查回 trace
tool = get_history_mongo_tool()
doc = tool.db["traces"].find_one({"session_id": sid})

print("\n==================== trace ====================")
print("outcome：", doc["summary"]["outcome"])
print("节点总耗时：", doc["total_node_duration_ms"], "ms")
print("-------------------- spans --------------------")
for s in doc["spans"]:
    print(f"  {s['node']:<30}{s['duration_ms']:>9.2f} ms | {s['status']}")
print("-------------------- summary ------------------")
print("recall      ：", doc["summary"]["recall"])
print("rrf_count   ：", doc["summary"]["rrf_count"])
print("rerank_count：", doc["summary"]["rerank_count"])
print("rerank_top1 ：", doc["summary"]["rerank_top1"])

# 文件日志
logp = r"D:\RAG\YUANLAI\logs\app.log"
print("\n日志文件：", os.path.exists(logp),
      "| 大小：", os.path.getsize(logp) if os.path.exists(logp) else 0, "字节")
