# utils/trace_utils.py
"""
轻量链路追踪（自建，复用 MongoDB 持久化）。
- trace_span：上下文管理器，记录单个节点的 span（节点名/起止/耗时/状态/异常）；
- build_trace_summary：从 state 提取 RAG 关键中间量（各检索召回数、rerank Top1、结果类型）；
- save_trace：组装完整 trace 文档写入 MongoDB 的 traces 集合。
"""
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any

from tool.logger import logger
from utils.mongo_history_utils import get_history_mongo_tool

TRACES_COLLECTION = "traces"


@contextmanager
def trace_span(state: Dict[str, Any], node_name: str):
    """
    记录单个节点 span。span 在 with 退出时（含异常）追加到 state['trace_spans']。
    """
    spans = state.setdefault("trace_spans", [])
    span: Dict[str, Any] = {
        "node": node_name,
        "start_ts": datetime.now().isoformat(timespec="milliseconds"),
        "status": "ok",
    }
    t0 = time.perf_counter()
    try:
        yield span
    except Exception as e:
        span["status"] = "failed"
        span["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        span["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        spans.append(span)


def _safe_len(v) -> int:
    try:
        return len(v)
    except Exception:
        return 0


def build_trace_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """从 state 提取 RAG 关键中间量，并判定本次请求的结果类型 outcome。"""
    reranked = state.get("reranked_docs") or []
    top1 = reranked[0].get("score") if reranked else None
    answer = state.get("answer") or ""

    if answer.startswith("未找到相关文献"):
        outcome = "rejected_no_literature"      # 节点1：无此文献
    elif answer.startswith("抱歉，知识库中未找到相关内容"):
        outcome = "rejected_no_context"         # floor：有文献但无相关切片
    elif ("您是想问" in answer) or ("请明确" in answer):
        outcome = "clarify"                     # 节点1：候选反问
    elif answer:
        outcome = "answered"
    else:
        outcome = "unknown"

    return {
        "outcome": outcome,
        "recall": {
            "dense": _safe_len(state.get("embedding_chunks")),
            "hyde": _safe_len(state.get("hyde_embedding_chunks")),
            "web": _safe_len(state.get("web_search_docs")),
        },
        "rrf_count": _safe_len(state.get("rrf_chunks")),
        "rerank_count": len(reranked),
        "rerank_top1": round(top1, 4) if isinstance(top1, (int, float)) else top1,
        "item_names": state.get("item_names") or [],
    }


def save_trace(state: Dict[str, Any]) -> str:
    """组装完整 trace 文档写入 MongoDB；失败仅记录日志、不影响主流程。"""
    try:
        tool = get_history_mongo_tool()
        coll = tool.db[TRACES_COLLECTION]
        coll.create_index([("session_id", 1)])

        spans = state.get("trace_spans") or []
        total_ms = round(sum(s.get("duration_ms", 0) for s in spans), 2)
        doc = {
            "trace_id": state.get("session_id"),
            "session_id": state.get("session_id"),
            "original_query": state.get("original_query"),
            "rewritten_query": state.get("rewritten_query"),
            "created_at": datetime.now(),
            "total_node_duration_ms": total_ms,
            "spans": spans,
            "summary": build_trace_summary(state),
        }
        result = coll.insert_one(doc)
        logger.info(
            f"trace 已保存：{state.get('session_id')} | spans={len(spans)} "
            f"| outcome={doc['summary']['outcome']}"
        )
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"trace 保存失败（不影响主流程）：{e}")
        return ""
