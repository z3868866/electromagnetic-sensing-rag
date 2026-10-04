# my_learn/67_floor_verify.py
"""
优化点③验证：
  A. rerank 绝对 floor：全低分 → 返回空；正常带断崖 → 保留高相关簇；
  B. answer 硬拦截：reranked_docs 为空 → 直接拒答、不调用 LLM。
"""
import sys
sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_answer_output import NodeAnswerOutput

# ---------- A. rerank floor / cliff ----------
rerank = NodeRerank()

def mk(scores):
    return [{"title": f"doc{i}", "content": f"内容{i}", "score": s} for i, s in enumerate(scores)]

cases = {
    "全低分（floor 拦截）": mk([0.31, 0.25, 0.2, 0.15]),
    "正常带断崖": mk([0.95, 0.9, 0.4, 0.2]),
    "空列表": [],
}
for name, docs in cases.items():
    out = rerank._step_3_cliff_cutoff(docs)
    kept = [round(d["score"], 3) for d in out]
    print(f"[{name}] 保留 {len(out)} 条: {kept}")

# ---------- B. answer 硬拦截 ----------
print("\n--- Answer 节点：空参考文档 ---")
ans_node = NodeAnswerOutput()
state = ans_node({
    "session_id": "floor_verify_01",
    "original_query": "一个知识库完全不涉及的问题",
    "rewritten_query": "一个知识库完全不涉及的问题",
    "item_names": ["基于深度卷积神经网络的协作频谱感知方法"],
    "history": [],
    "reranked_docs": [],
    "is_stream": False,
})
print("answer：", state.get("answer"))
