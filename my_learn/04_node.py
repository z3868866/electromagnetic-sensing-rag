from abc import ABC, abstractmethod
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.constants import START, END


class MyState(TypedDict, total=False):
    is_pdf: bool
    is_md: bool
    trace: list


class BaseNode(ABC):
    name = "base_node"

    def __init__(self):
        if self.name == "base_node":
            raise ValueError("必须覆盖 name")

    def __call__(self, state):
        print(f"  [{self.name}] 开始")
        state = self.process(state)
        print(f"  [{self.name}] 完成")
        return state

    @abstractmethod
    def process(self, state):
        pass


class EntryNode(BaseNode):
    name = "entry"
    def process(self, state):
        return state            # 真实成品里这里判断扩展名、设置开关


class PdfNode(BaseNode):
    name = "pdf"
    def process(self, state):
        state["trace"].append("pdf")
        return state


class MdNode(BaseNode):
    name = "md"
    def process(self, state):
        state["trace"].append("md")
        return state


def route(state):
    if state.get("is_pdf"):
        return "pdf"
    if state.get("is_md"):
        return "md"
    return "end"


graph = StateGraph(MyState)
graph.add_node("entry", EntryNode())   # 注册的是 BaseNode 实例（可调用对象）
graph.add_node("pdf", PdfNode())
graph.add_node("md", MdNode())
graph.add_edge(START, "entry")
graph.add_conditional_edges("entry", route, {"pdf": "pdf", "md": "md", "end": END})
graph.add_edge("pdf", END)
graph.add_edge("md", END)
app = graph.compile()


if __name__ == "__main__":
    for name, init in [
        ("PDF", {"is_pdf": True, "trace": []}),
        ("Markdown", {"is_md": True, "trace": []}),
        ("其他", {"trace": []}),
    ]:
        print(f"===== {name} =====")
        r = app.invoke(init)
        print("  trace:", r["trace"])
