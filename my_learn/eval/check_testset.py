import os
import sys
import json

DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_JSON = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks.json"

with open(os.path.join(DIR, "test_set.json"), "r", encoding="utf-8") as f:
    test_set = json.load(f)
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

valid_parents = sorted({c["parent_title"] for c in chunks})
print("chunks 中真实存在的 parent_title：")
for p in valid_parents:
    print("  -", p)

print("\n标注校验：")
all_ok = True
for item in test_set:
    rel = item.get("relevant_parent_titles", [])
    if not rel:
        print(f"  {item['qid']}: 无相关章节（用于拒答测试）")
        continue
    missing = [r for r in rel if r not in valid_parents]
    if missing:
        all_ok = False
        print(f"  {item['qid']}: 存在无法匹配的标注 -> {missing}")
    else:
        print(f"  {item['qid']}: OK（{len(rel)} 个章节）")

print("\n样本总数：", len(test_set))
print("校验结果：", "全部通过" if all_ok else "存在不一致，需修正")
