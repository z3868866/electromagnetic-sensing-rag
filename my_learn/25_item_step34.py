import json
import os
from pathlib import Path
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv(r"D:\RAG\YUANLAI\.env")

TEMPLATE = """
请从以下信息中识别文档的核心主体（产品名称、型号、方法或主题）：
文件名：{file_title}

正文切片（用于辅助识别）：
{context}

要求：
1. 返回一个简洁、规范的主体名称，例如"苏泊尔5000W大功率电磁炉""基于深度卷积神经网络的协作频谱感知方法"；
2. 只返回主体名称本身，不要添加任何解释、引号或其他内容；
3. 如果无法从正文明确识别，请直接返回文件名作为主体名称。
"""

SYSTEM_PROMPT = "你是一个专业的文档主体识别模型，能从产品手册、技术文档或论文中准确提取其核心描述对象的名称。"



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


class NodeItemNameRecognition(BaseNode):
    name = "node_item_name_recognition"
    K = 5
    MAX_CHARS = 2500

    def process(self, state):
        file_title, chunks = self.step_1(state)
        context = self.step_2(chunks)
        item_name = self.step_3(file_title, context)
        self.step_4(state, chunks, item_name)
        print("识别主体：", item_name)
        print("state['item_name']：", state["item_name"])
        print("前3个 chunk 的标签：", [c.get("item_name") for c in chunks[:3]])
        return state

    def step_1(self, state):
        file_title = state.get("file_title", "")
        if not file_title:
            raise ValueError("file_title 为空")
        chunks = state.get("chunks", [])
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks 为空或非列表")
        return file_title, chunks

    def step_2(self, chunks):
        parts, total = [], 0
        for idx, chunk in enumerate(chunks[:self.K], start=1):
            t = chunk.get("title", "").strip()
            c = chunk.get("content", "").strip()
            piece = f"【切片{idx}】\n标题：{t}\n内容：{c}"
            parts.append(piece)
            total += len(piece)
            if total > self.MAX_CHARS:
                break
        return "\n\n".join(parts).strip()[:self.MAX_CHARS]

    def step_3(self, file_title, context):
        if not context:
            return file_title
        try:
            prompt = TEMPLATE.format(file_title=file_title, context=context)
            llm = ChatOpenAI(
                model=os.getenv("LLM_DEFAULT_MODEL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"),
                temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1")),
                extra_body={"enable_thinking": False}
            )
            messages = [SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=prompt)]
            item_name = llm.invoke(messages).content
            item_name = (item_name.replace(" ", "").replace("\n", "")
                         .replace("\t", "").replace("\r", ""))
            invalid_values = {"", "空字符串", "无", "没有", "NONE", "NULL", "未知"}
            if item_name in invalid_values:
                return file_title
            return item_name

        except Exception as e:
            print("LLM 异常：", e)
            return file_title

    def step_4(self, state, chunks, item_name):
        state["item_name"] = item_name
        for chunk in chunks:
            chunk["item_name"] = item_name
        state["chunks"] = chunks


if __name__ == "__main__":
    chunks_path = Path(r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks.json")
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    state = {
        "file_title": "基于深度卷积神经网络的协作频谱感知方法",
        "chunks": chunks
    }
    NodeItemNameRecognition()(state)
