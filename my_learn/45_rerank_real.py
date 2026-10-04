import importlib.util


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = r"D:\RAG\YUANLAI\my_learn"
m39 = load(B + r"\39_search_chunks.py", "m39")
m40 = load(B + r"\40_hyde.py", "m40")
m42 = load(B + r"\42_rrf.py", "m42")
m41 = load(B + r"\41_web_search.py", "m41")
m44 = load(B + r"\44_rerank.py", "m44")

if __name__ == "__main__":
    query = "基于深度卷积神经网络的协作频谱感知方法的网络结构是什么？"
    item_names = ["基于深度卷积神经网络的协作频谱感知方法"]

    # 召回三路
    emb = m39.search_kb(query, item_names, recall=8)
    hyde_doc = m40.generate_hyde(query)
    hyde = m40.search_with_hyde(hyde_doc, item_names, recall=8)
    web = m41.node_web_search(query).get("web_search_docs", [])

    # RRF 融合本地两路
    merged = m42.rrf_merge([(emb, 1.0), (hyde, 1.0)], k=60, max_results=10)
    rrf_chunks = [d for d, _ in merged]

    # Rerank（本地 + 网络统一精排）
    state = {"rewritten_query": query, "rrf_chunks": rrf_chunks,
             "web_search_docs": web}
    cutoff, ranked = m44.node_rerank(state)

    print("待精排：", len(ranked), "条")
    for d in ranked:
        print(f"  {round(d['score'],4)} | {d['source']:5} | {str(d.get('title',''))[:36]}")
    print("断崖截断保留：", len(cutoff), "条")
