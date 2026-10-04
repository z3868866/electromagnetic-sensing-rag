from typing import TypedDict,List
import copy

class ImportGraphState(TypedDict):
    task_id: str
    is_pdf_read_enable:bool
    is_md_read_enable:bool
    import_file_path:str
    file_title:str
    item_name:str
    md_content:str
    chunks:List


GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "import_file_path": "",
    "file_title": "",
    "item_name": "",
    "md_content": "",
    "chunks": [],
}


def get_default_state()->ImportGraphState:
    return copy.deepcopy(GRAPH_DEFAULT_STATE)

if __name__=="__main__":
    state=get_default_state()
    print(state)