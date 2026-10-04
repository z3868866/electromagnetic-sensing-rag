def normalize(docs):
    out = []
    for d in docs or []:
        if isinstance(d, dict) and d.get("chunk_id") is not None:
            out.append(d)
    return out


def rrf_merge(rrf_inputs, k=60, max_results=10):
    chunk_scores, chunk_data = {}, {}
    for docs, weight in rrf_inputs:
        for rank, doc in enumerate(normalize(docs), start=1):
            cid = doc["chunk_id"]
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + weight / (k + rank)
            chunk_data.setdefault(cid, doc)

    sorted_results = sorted(
        [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
        key=lambda x: x[1], reverse=True)
    return sorted_results[:max_results]


def node_rrf(state):
    rrf_inputs = [
        (state.get("embedding_chunks"), 1.0),
        (state.get("hyde_embedding_chunks"), 1.0),
    ]
    merged = rrf_merge(rrf_inputs, k=60, max_results=10)
    return {"rrf_chunks": [doc for doc, _ in merged],
            "_scores": [(doc["chunk_id"], round(s, 5)) for doc, s in merged]}


if __name__ == "__main__":
    mock_state = {
        "embedding_chunks": [
            {"chunk_id": 1, "content": "向量#1", "title": "t1"},
            {"chunk_id": 2, "content": "向量#2", "title": "t2"},
            {"chunk_id": 3, "content": "向量#3", "title": "t3"},
        ],
        "hyde_embedding_chunks": [
            {"chunk_id": 1, "content": "HyDE#1", "title": "t1"},
            {"chunk_id": 4, "content": "HyDE#4", "title": "t4"},
            {"chunk_id": 2, "content": "HyDE#2", "title": "t2"},
        ],
    }
    out = node_rrf(mock_state)
    print("融合后顺序（chunk_id, RRF分）：")
    for cid, s in out["_scores"]:
        print(f"  chunk {cid}: {s}")
