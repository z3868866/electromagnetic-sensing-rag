import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from openai import OpenAI
from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client, online_hybrid_search

DIR = os.path.dirname(os.path.abspath(__file__))
TOP_K = 8
MODEL = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")

llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
)

with open(os.path.join(DIR, "test_set.json"), "r", encoding="utf-8") as f:
    test_set = json.load(f)

milvus = get_milvus_client()


def chat_json(messages, max_retry=4):
    for attempt in range(max_retry):
        try:
            resp = llm.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or attempt < max_retry - 1:
                wait = min(5 * (attempt + 1), 30)
                print(f"  触发限流/异常，等待 {wait}s …")
                time.sleep(wait)
            else:
                raise
    return {}


def chat_text(system, user, max_retry=4):
    for attempt in range(max_retry):
        try:
            resp = llm.chat.completions.create(
                model=MODEL,
                temperature=0.1,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or attempt < max_retry - 1:
                wait = min(5 * (attempt + 1), 30)
                print(f"  触发限流/异常，等待 {wait}s …")
                time.sleep(wait)
            else:
                raise
    return ""


GEN_SYS = (
    "你是电磁感知知识库问答助手。请严格依据用户提供的参考资料回答问题，"
    "答案中的每个结论都必须能在资料中找到依据，不得编造资料中没有的内容、数据或图片链接；"
    "若资料不足以回答问题，请直接回答“抱歉，知识库中未找到相关内容。”"
)

JUDGE_SYS = (
    "你是严格的RAG答案评测裁判。请依据给定的问题、参考资料上下文、系统答案和人工参考答案评分，"
    "输出JSON，字段为：faithfulness(0到1，系统答案中可由上下文支撑的陈述比例，防幻觉)、"
    "relevancy(0到1，系统答案是否切题、在回答原问题)、"
    "correctness(0到1，与人工参考答案相比事实是否正确、要点是否齐全)、"
    "unsupported_claims(数组，列出无上下文依据的陈述)、reason(简短中文理由)。只输出JSON。"
)

scores = {"faithfulness": [], "relevancy": [], "correctness": []}
records = []
print(f"生成质量评测（Top-{TOP_K}，模型 {MODEL}）\n" + "=" * 60)

for item in test_set:
    qid, question = item["qid"], item["question"]
    reference = item.get("reference_answer", "")

    res = online_hybrid_search(
        milvus,
        milvus_config.chunks_collection,
        question,
        expr='chunk_level=="child"',
        limit=TOP_K,
        output_fields=["parent_uid", "parent_title", "title"],
    )
    hits = res[0]

    # 命中子块 → 有序去重 parent_uid → 回取父块完整章节（small-to-big）
    parent_uids = []
    for h in hits:
        uid = h.get("entity", {}).get("parent_uid", "")
        if uid and uid not in parent_uids:
            parent_uids.append(uid)

    context_blocks = []
    if parent_uids:
        uid_list = ", ".join(f'"{u}"' for u in parent_uids)
        p_rows = milvus.query(
            milvus_config.chunks_collection,
            filter=f'parent_uid in [{uid_list}] and chunk_level == "parent"',
            output_fields=["parent_title", "content", "parent_uid"],
        )
        p_map = {r.get("parent_uid"): r for r in p_rows}
        for uid in parent_uids:
            r = p_map.get(uid)
            if r:
                context_blocks.append(f"【章节：{r.get('parent_title','')}】\n{r.get('content','')}")
    context = "\n\n".join(context_blocks)

    answer = chat_text(GEN_SYS, f"参考资料：\n{context}\n\n问题：{question}")
    judge = chat_json([
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": f"问题：{question}\n\n参考资料上下文：\n{context}\n\n"
                                    f"系统答案：\n{answer}\n\n人工参考答案：\n{reference}"},
    ])

    for k in scores:
        v = judge.get(k)
        if isinstance(v, (int, float)):
            scores[k].append(float(v))

    records.append({"qid": qid, "question": question, "answer": answer, "judge": judge})
    print(f"{qid}: F={judge.get('faithfulness')} R={judge.get('relevancy')} "
          f"C={judge.get('correctness')} | {str(judge.get('reason',''))[:50]}")

with open(os.path.join(DIR, "generation_eval_result.json"), "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("=" * 60)
n = len(scores["faithfulness"])
print(f"参与评测问题数：{n}")
for k, name in [("faithfulness", "Faithfulness 忠实度"),
                ("relevancy", "Answer Relevancy 相关性"),
                ("correctness", "Answer Correctness 正确性")]:
    vals = scores[k]
    print(f"{name}：{round(sum(vals)/len(vals), 4) if vals else 'N/A'}")
print("明细已备份：my_learn/eval/generation_eval_result.json")
