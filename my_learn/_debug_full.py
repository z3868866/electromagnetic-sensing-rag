import sys, uuid
sys.path.insert(0, r"D:\RAG\YUANLAI")
from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")
from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm

sid = "dbgfull_" + uuid.uuid4().hex[:8]
state = {
    "session_id": sid,
    "original_query": "基于深度卷积神经网络的协作频谱感知方法是如何缓解梯度消失的？"
}
node = NodeItemNameConfirm()
out = node(state)
print("FINAL item_names:", out.get("item_names"))
print("FINAL answer:", out.get("answer"))
print("FINAL rewritten:", out.get("rewritten_query"))
print("FULL DONE")
