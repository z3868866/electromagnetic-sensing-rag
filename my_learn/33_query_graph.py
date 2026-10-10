import time
import copy
from typing import List, TypedDict
from langgraph.graph import StateGraph, END


# ---- 复用阶段二 State（内联，保证文件独立可运行）----
class QueryGraphState(TypedDict, total=False):
    task_id: str
    session_id: str
    original_query: str
    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list
    rrf_chunks: list
    reranked_docs: list
    prompt: str
    answer: str
    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool


def create_default_state(**overrides) -> QueryGraphState:
    defaults = {k: ([] if k.endswith("chunks") or k in ("web_search_docs", "reranked_docs", "history", "item_names") else "")
                for k in QueryGraphState.__annotations__}
    defaults["is_stream"] = False
    defaults.update(overrides)
    return defaults


# ---- 7 个骨架节点（打印 + 模拟耗时 + 返回占位数据）----
def node_confirm(state):
    print("【节点1 文献确认】")
    return {"rewritten_query": state["original_query"],
            "item_names": ["基于DCNN的协作频谱感知方法"]}

def node_search_embedding(state):
    print("【节点2 向量检索】开始")
    time.sleep(2)
    print("【节点2 向量检索】完成")
    return {"embedding_chunks": [{"content": "向量检索结果"}]}

def node_hyde(state):
    print("【节点3 HyDE】开始")
    time.sleep(2)
    print("【节点3 HyDE】完成")
    return {"hyde_embedding_chunks": [{"content": "HyDE结果"}]}

def node_web_search(state):
    print("【节点4 网络搜索】开始")
    time.sleep(2)
    print("【节点4 网络搜索】完成")
    return {"web_search_docs": [{"content": "网络搜索结果"}]}

def node_rrf(state):
    print("【节点5 RRF融合】")
    total = len(state.get("embedding_chunks", [])) + \
            len(state.get("hyde_embedding_chunks", [])) + \
            len(state.get("web_search_docs", []))
    print("  汇合到的文档总数：", total)
    return {"rrf_chunks": [{"content": "融合结果"}]}

def node_rerank(state):
    print("【节点6 Rerank精排】")
    return {"reranked_docs": [{"content": "精排结果"}]}

def node_answer(state):
    print("【节点7 答案输出】")
    return {"answer": "这是最终答案（骨架）"}


# ---- 条件路由：有 answer 直走答案节点；无 answer 扇出三路 ----
def route_after_confirm(state):
    if state.get("answer"):
        return "node_answer"
    return ["node_search_embedding", "node_hyde", "node_web_search"]


# ---- 构建图 ----
workflow = StateGraph(QueryGraphState)
workflow.add_node("node_confirm", node_confirm)
workflow.add_node("node_search_embedding", node_search_embedding)
workflow.add_node("node_hyde", node_hyde)
workflow.add_node("node_web_search", node_web_search)
workflow.add_node("node_rrf", node_rrf)
workflow.add_node("node_rerank", node_rerank)
workflow.add_node("node_answer", node_answer)

workflow.set_entry_point("node_confirm")
workflow.add_conditional_edges(
    "node_confirm",
    route_after_confirm,
    ["node_answer", "node_search_embedding", "node_hyde", "node_web_search"]
)
# fan-in：三路都汇合到 node_rrf
workflow.add_edge("node_search_embedding", "node_rrf")
workflow.add_edge("node_hyde", "node_rrf")
workflow.add_edge("node_web_search", "node_rrf")
workflow.add_edge("node_rrf", "node_rerank")
workflow.add_edge("node_rerank", "node_answer")
workflow.add_edge("node_answer", END)

app = workflow.compile()


if __name__ == "__main__":
    state = create_default_state(session_id="q1",
                                 original_query="DCNN 是如何解决梯度消失问题的？")
    start = time.time()
    result = app.invoke(state)
    print("总耗时：", round(time.time() - start, 1), "秒（≈2s 说明三路并行，≈6s 说明串行）")
    print("answer：", result["answer"])
