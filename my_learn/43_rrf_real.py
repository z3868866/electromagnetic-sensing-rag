import importlib.util


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = r"D:\RAG\YUANLAI\my_learn"
m39 = load(BASE + r"\39_search_chunks.py", "m39")
m40 = load(BASE + r"\40_hyde.py", "m40")
m42 = load(BASE + r"\42_rrf.py", "m42")

if __name__ == "__main__":
    query = "基于深度卷积神经网络的协作频谱感知方法的网络结构是什么？"
    item_names = ["基于深度卷积神经网络的协作频谱感知方法"]

    # 1. 节点2：直接向量检索
    emb = m39.search_kb(query, item_names, recall=8)
    # 2. 节点3：HyDE（先生成假设答案，再检索）
    hyde_doc = m40.generate_hyde(query)
    hyde = m40.search_with_hyde(hyde_doc, item_names, recall=8)

    print(f"节点2 {len(emb)} 条 | HyDE {len(hyde)} 条")

    # 3. RRF 融合（复用42算法）
    merged = m42.rrf_merge([(emb, 1.0), (hyde, 1.0)], k=60, max_results=10)
    print("\n真实 RRF 融合排序：")
    for i, (doc, score) in enumerate(merged, 1):
        print(f"{i}. {round(score,5)} | {str(doc.get('title',''))[:34]}")
