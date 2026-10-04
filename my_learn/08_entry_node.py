from pathlib import Path
from abc import ABC, abstractmethod


class BaseNode(ABC):
    name = "base_node"
    def __init__(self):
        if self.name == "base_node":
            raise ValueError("必须覆盖 name")
    def __call__(self, state):
        print(f"--- {self.name} 开始 ---")
        state = self.process(state)
        print(f"--- {self.name} 完成 ---")
        return state
    @abstractmethod
    def process(self, state):
        pass


class NodeEntry(BaseNode):
    name = "node_entry"

    def process(self, state):
        # 1. 读取路径并非空校验（成品抛 StateFieldError）
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise ValueError("import_file_path 不能为空")

        # 2. 转 Path 对象，检查存在性（成品抛 FileProcessingError）
        path = Path(import_file_path)
        if not path.exists():
            raise ValueError(f"文件 {path.name} 不存在")

        # 3. 按后缀分支（成品不支持类型抛 ValidationError）
        if path.suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif path.suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            raise ValueError(f"不支持的文件类型 {path.suffix}")

        # 4. 提取不含后缀的文件名
        state["file_title"] = path.stem
        return state


def fresh_state():
    return {
        "task_id": "", "is_pdf_read_enabled": False, "is_md_read_enabled": False,
        "pdf_path": "", "md_path": "", "file_title": "",
    }


if __name__ == "__main__":
    node = NodeEntry()

    print("=== 场景1：真实 Markdown ===")
    s = fresh_state()
    s["import_file_path"] = r"D:\RAG\YUANLAI\my_learn\test_doc.md"
    print(node(s))

    print("=== 场景2：文件不存在 ===")
    try:
        s = fresh_state()
        s["import_file_path"] = r"D:\RAG\YUANLAI\my_learn\no_exist.pdf"
        node(s)
    except ValueError as e:
        print("捕获：", e)

    print("=== 场景3：不支持的类型（真实存在的 .cmd）===")
    try:
        s = fresh_state()
        s["import_file_path"] = r"D:\RAG\YUANLAI\tmp\sample.cmd"
        node(s)
    except ValueError as e:
        print("捕获：", e)

    print("=== 场景4：空路径 ===")
    try:
        s = fresh_state()
        s["import_file_path"] = ""
        node(s)
    except ValueError as e:
        print("捕获：", e)
