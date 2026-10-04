# processor/import_processor/parent_split_utils.py
"""
父子切片（Parent-Child Chunking）工具。

父块：规范章节（完整原文），不直接参与检索，命中子块后回取喂给 LLM。
子块：父块内约 500 字的切分，带向量、用于检索匹配。
"""
import re
from typing import List, Dict, Tuple

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

CHILD_SIZE = 500
CHILD_OVERLAP = 80
WHITELIST = {"参考文献", "摘要", "关键词"}
MAX_PARENT_LEN = 60000  # 父块入库长度兜底阈值，低于 Milvus VARCHAR 上限 65535

_title_line_re = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
_num_re = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
_exclude_re = re.compile(r"^(表|图|Table|Figure|Fig\.?)\s*\d")
_fence_open_re = re.compile(r"^(`{3,}|~{3,})(.*)$")
_fence_close_re = re.compile(r"^(`{3,}|~{3,})$")


def _is_real_title(text: str, level: int) -> bool:
    # 图表题注（图/表/Figure/Table 开头）不是独立章节，排除
    if _exclude_re.match(text):
        return False
    # Markdown 的 # 是作者显式标记的标题结构信号：
    # 默认尊重并作为章节，不再强制要求数字编号，以兼容无编号的英文标题
    # （如 Related work / Methodology / Conclusion），避免整篇塌缩成单个父块。
    return True


def build_parent_child(md_content: str) -> Tuple[List[Dict], List[Dict]]:
    """
    从 Markdown 文本构建父子两级。

    返回：
        parents:  [{parent_uid, parent_title, content}]，空章节保留（content=""）
        children: [{parent_uid, parent_title, title, content, part}]
    """
    lines = md_content.split("\n")

    in_code, start_marker = False, None
    raw_parents = []
    cur_title, cur_body = None, []

    def flush():
        if cur_title is not None:
            raw_parents.append({"parent_title": cur_title,
                                "content": "\n".join(cur_body).strip()})

    for line in lines:
        s = line.strip()
        mo, mc = _fence_open_re.match(s), _fence_close_re.match(s)

        if not in_code and mo:
            in_code, start_marker = True, mo.group(1)
            if cur_title:
                cur_body.append(line)
            continue
        if in_code and mc:
            if mc.group(1)[0] == start_marker[0] and \
                    len(mc.group(1)) >= len(start_marker):
                in_code, start_marker = False, None
            if cur_title:
                cur_body.append(line)
            continue

        mt = _title_line_re.match(line)
        if not in_code and mt and _is_real_title(mt.group(2), len(mt.group(1))):
            flush()
            cur_title, cur_body = s, []
        elif cur_title:
            cur_body.append(line)
    flush()

    # 兜底（防御纵深）：即便标题识别异常导致出现超大父块，
    # 也在入库前按 MAX_PARENT_LEN 强制拆分，保证不超过 Milvus VARCHAR 上限 65535。
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_PARENT_LEN,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    bounded_parents = []
    for rp in raw_parents:
        if len(rp["content"]) <= MAX_PARENT_LEN:
            bounded_parents.append(rp)
            continue
        for k, piece in enumerate(parent_splitter.split_text(rp["content"]), start=1):
            piece = piece.strip()
            if piece:
                suffix = "" if k == 1 else f"（续{k}）"
                bounded_parents.append({
                    "parent_title": rp["parent_title"] + suffix,
                    "content": piece,
                })
    raw_parents = bounded_parents

    parents, children = [], []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_SIZE,
        chunk_overlap=CHILD_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    for i, p in enumerate(raw_parents):
        parent_uid = f"pc_{i}"
        parents.append({"parent_uid": parent_uid,
                        "parent_title": p["parent_title"],
                        "content": p["content"]})
        if not p["content"]:
            continue
        for j, t in enumerate(splitter.split_text(p["content"]), start=1):
            t = t.strip()
            if t:
                children.append({
                    "parent_uid": parent_uid,
                    "parent_title": p["parent_title"],
                    "title": p["parent_title"].lstrip("#").strip(),
                    "content": f"{p['parent_title'].lstrip('#').strip()}\n{t}",
                    "part": j,
                })

    return parents, children
