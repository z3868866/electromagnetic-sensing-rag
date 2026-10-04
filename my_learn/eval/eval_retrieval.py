import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from config.milvus_config import milvus_config
from utils.milvus_utils import get_milvus_client, online_hybrid_search

DIR = os.path.dirname(os.path.abspath(__file__))
TOP_K = 8

with open(os.path.join(DIR, "test_set.json"), "r", encoding="utf-8") as f:
    test_set = json.load(f)

client = get_milvus_client()

hit_flags, rr_scores, chapter_recalls = [], [], []
print(f"检索评测（Top-{TOP_K}）\n" + "=" * 60)

for item in test_set:
    qid, question = item["qid"], item["question"]
    relevant = set(item.get("relevant_parent_titles", []))

    if not relevant:
        print(f"{qid}: 跳过（拒答测试）")
        continue

    res = online_hybrid_search(
        client,
        milvus_config.chunks_collection,
        question,
        expr='chunk_level=="child"',
        limit=TOP_K,
        output_fields=["parent_title", "title", "parent_uid"],
    )
    hits = res[0]
    hit_parents = [h.get("entity", {}).get("parent_title", "") for h in hits]

    # Hit@K
    is_hit = any(p in relevant for p in hit_parents)
    hit_flags.append(1 if is_hit else 0)

    # MRR：第一个相关 chunk 的排名
    rr = 0.0
    first_rank = None
    for i, p in enumerate(hit_parents, start=1):
        if p in relevant:
            rr = 1.0 / i
            first_rank = i
            break
    rr_scores.append(rr)

    # 章节召回比例
    recalled = len(set(hit_parents) & relevant)
    chapter_recalls.append(recalled / len(relevant))

    print(f"{qid}: Hit={1 if is_hit else 0} | 首个相关排名={first_rank} | RR={round(rr,3)} "
          f"| 章节召回 {recalled}/{len(relevant)}")

n = len(hit_flags)
print("=" * 60)
print(f"参与评测问题数：{n}")
print(f"Hit Rate@{TOP_K}：{round(sum(hit_flags)/n, 4)}")
print(f"MRR：{round(sum(rr_scores)/n, 4)}")
print(f"平均章节召回比例：{round(sum(chapter_recalls)/n, 4)}")
