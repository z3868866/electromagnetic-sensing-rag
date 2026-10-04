import os
from http import HTTPStatus
import dashscope
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")
dashscope.api_key = os.getenv("OPENAI_API_KEY")
RERANK_MODEL = os.getenv("TEXT_RERANK_MODEL", "qwen3-rerank")
RERANK_INSTRUCT = os.getenv("TEXT_RERANK_INSTRUCT", "")
MAX_TOPK, MIN_TOPK, GAP_ABS, GAP_RATIO = 10, 3, 0.5, 0.25


def rerank_documents(query, documents):
    resp = dashscope.TextReRank.call(
        model=RERANK_MODEL, query=query, documents=documents,
        top_n=len(documents), return_documents=False, instruct=RERANK_INSTRUCT)
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"rerank失败 {resp.status_code}: {resp.message}")
    scores = [0.0] * len(documents)
    for r in resp.output.get("results", []):
        scores[r["index"]] = r["relevance_score"]
    return scores


def merge_sources(state):
    docs = []
    for d in state.get("rrf_chunks") or []:
        docs.append({"content": d.get("content"), "title": d.get("title"),
                     "chunk_id": d.get("chunk_id"), "url": "", "source": "local"})
    for d in state.get("web_search_docs") or []:
        docs.append({"content": d.get("snippet"), "title": d.get("title"),
                     "chunk_id": None, "url": d.get("url"), "source": "web"})
    return docs


def cliff_cutoff(ranked):
    upper = min(MAX_TOPK, len(ranked))
    lower = min(MIN_TOPK, upper)
    pos = upper
    for idx in range(lower - 1, upper - 1):
        cur, nxt = ranked[idx].get("score"), ranked[idx + 1].get("score")
        if cur is None or nxt is None:
            continue
        abs_gap = cur - nxt
        rel_gap = abs_gap / (abs(cur) + 1e-6)
        if abs_gap >= GAP_ABS or rel_gap >= GAP_RATIO:
            pos = idx + 1
            break
    return ranked[:pos]


def node_rerank(state):
    query = state.get("rewritten_query") or state.get("original_query")
    merged = merge_sources(state)
    print("待精排文档：", len(merged))
    scores = rerank_documents(query, [d["content"] for d in merged])
    scored = [{**d, "score": s} for d, s in zip(merged, scores)]
    ranked = sorted(scored, key=lambda x: x["score"], reverse=True)  # 先排序
    return cliff_cutoff(ranked), ranked                              # 再截断



if __name__ == "__main__":
    mock = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "l1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以用万用表蜂鸣档测量。"},
            {"chunk_id": "l2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"}],
        "web_search_docs": [
            {"url": "https://a/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感对地阻值，阻值偏低就是短路。"},
            {"url": "https://a/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"}],
    }
    cutoff, ranked = node_rerank(mock)
    print("\n全部精排打分：")
    for d in ranked:
        print(f"  {round(d['score'],4)} | {d['source']:5} | {d['title']}")
    print("\n断崖截断后保留：", len(cutoff), "条")
    for d in cutoff:
        print(f"  {round(d['score'],4)} | {d['title']}")
