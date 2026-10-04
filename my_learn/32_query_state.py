import copy
from typing import List, TypedDict


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


graph_default_state: QueryGraphState = {
    "task_id": "",
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False
}


def create_default_state(**overrides) -> QueryGraphState:
    state = copy.deepcopy(graph_default_state)
    state.update(overrides)
    return state


if __name__ == "__main__":
    state = create_default_state(
        task_id="task_001",
        session_id="query_001",
        original_query="DCNN 是如何解决梯度消失问题的？"
    )
    print(state)
