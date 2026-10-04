import re
from pathlib import Path
from abc import ABC, abstractmethod
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

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


class NodeDocumentSplit(BaseNode):
    name = "node_document_split"

    def process(self, state):
        content, file_title = self.step_1(state)
        sections, title_count, lines_count = self.step_2(content, file_title)
        sections = self.step_3(content, sections, title_count, file_title)

        refined = []
        for sec in sections:
            refined.extend(self.split_long(sec, 2000))
        final = self.merge_short(refined, 500)

        for s in final:
            if not s.get("parent_title"):
                s["parent_title"] = s.get("title") or ""

        self.step_5(lines_count, final)
        self.step_6(state, final)

        state["chunks"] = final
        return state





    def step_1(self, state):
        file_title = state.get("file_title", "")
        if not file_title:
            raise ValueError("file_title 缺失")
        md_content = state.get("md_content", "")
        if not md_content:
            raise ValueError("md_content 缺失")
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title

    def step_2(self, content, file_title):
        title_pattern = r'\s*#{1,6}\s+.+'
        lines = content.split("\n")
        sections = []
        title_count = 0
        current_title = ""
        current_lines = []
        in_code_block = False
        code_block_start_marker = None

        def flush_section():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title
            })

        for line in lines:
            stripped = line.strip()

            marker_match = re.match(r'^(`{3,}|~{3,})$', stripped)
            if marker_match:
                marker = marker_match.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_block_start_marker = marker
                elif in_code_block and stripped == code_block_start_marker:
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(stripped)
                continue

            is_title = (not in_code_block) and re.match(title_pattern, line)
            if is_title:
                flush_section()
                current_title = stripped
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(stripped)

        flush_section()
        return sections, title_count, len(lines)
    
    def step_3(self, content, sections, title_count, file_title):
        if title_count == 0:
            return [{"title": "无标题", "content": content, "file_title": file_title}]
        return sections

    def split_long(self, section, max_len=2000):
        content = section["content"]
        if len(content) <= max_len:
            return [section]

        title = section["title"]
        prefix = f"{title}\n\n" if title else ""
        available = max_len - len(prefix)
        if available <= 0:
            return [section]

        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "？", "！", "；", ".", "?", "!", ";", " "]
        )

        subs = []
        for idx, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue
            subs.append({
                "parent_title": title,
                "title": f"{title}-{idx}",
                "content": (prefix + text).strip(),
                "part": idx,
                "file_title": section["file_title"]
            })
        return subs
    def merge_short(self, sections, min_len=500):
        if not sections:
            return []
        merged = []
        current = None
        for sec in sections:
            if current is None:
                current = sec
                continue

            is_current_short = len(current["content"]) < min_len
            is_same_parent = current.get("parent_title") == sec.get("parent_title")

            if is_current_short and is_same_parent:
                parent = sec.get("parent_title", "")
                nxt = sec["content"]
                if parent and nxt.startswith(parent):
                    nxt = nxt[len(parent):].lstrip()
                current["content"] += "\n\n" + nxt
                if "part" in sec:
                    current["part"] = sec["part"]
            else:
                merged.append(current)
                current = sec

        if current is not None:
            merged.append(current)
        return merged

    def step_5(self, lines_count, sections):
        print("=" * 20, "文档切分统计", "=" * 20)
        print("MD原始总行数：", lines_count)
        print("最终Chunk数量：", len(sections))

    def step_6(self, state, sections):
        try:
            backup = Path(state["md_path"]).parent / "chunks.json"
            with open(backup, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            print("chunks 已备份：", backup)
        except Exception as e:
            print("备份失败：", e)



if __name__ == "__main__":
    new_md = Path(r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法_new.md")
    state = {
        "md_path": str(new_md),
        "file_title": new_md.stem.replace("_new", ""),
        "md_content": new_md.read_text(encoding="utf-8")
    }

    node = NodeDocumentSplit()
    node(state)

    # 构造一份完全没有标题的纯文本
    plain = "这是第一段，全文没有任何 Markdown 标题。\n这是第二段，同样没有标题。"
    secs, tc, lc = node.step_2(plain, "纯文本测试")
    secs = node.step_3(plain, secs, tc, "纯文本测试")
    print(f"[无标题文本] title_count={tc}，兜底后：")
    for s in secs:
        print("   ", s["title"], "|", len(s["content"]), "字")

