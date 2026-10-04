# my_learn/66_confirm_alias_e2e.py
"""
优化点②端到端：用户用缩写提问，验证节点1经别名词典精确匹配直接确认，
而不是反问；并对比无关问题。
"""
import os
import sys

sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm

node = NodeItemNameConfirm()

for idx, q in enumerate(["DCNN的网络结构是什么？", "今天天气怎么样？"], start=1):
    state = node({
        "session_id": f"alias_e2e_{idx}",
        "original_query": q,
    })
    print("=" * 60)
    print("问题：", q)
    print("item_names：", state.get("item_names"))
    print("answer：", state.get("answer"))
