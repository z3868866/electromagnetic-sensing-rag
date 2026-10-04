# utils/alias_utils.py
"""
别名 / 缩写自动构建工具（优化点②）。

作用：学术文献存在大量等价指称（英文标题、方法英文名、标准缩写、中文简称），
仅用标准文献名建立索引时，缩写（如 DCNN）无法对齐到标准名。
本模块在离线导入阶段用 LLM 从文献内容抽取别名，并构造
“标准名 + 别名”的可检索文本 match_text，供 dense / sparse 建索引。
"""
import json
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config.lm_config import lm_config

ALIAS_SYSTEM_PROMPT = (
    "你是学术文献别名抽取助手。请从给定的文献标题、摘要和正文中，抽取能够唯一指代"
    "【该文献本身或该文提出的核心方法/模型】的所有等价指称，包括：英文标题、方法/模型英文名、"
    "标准英文缩写、中文简称等。要求："
    "1) 只抽取能唯一指代该文献或其核心方法的名称，不要抽取泛化术语（如“深度学习”“神经网络”“算法”）；"
    "2) 缩写必须是文中以“全称(缩写)”或“全称, 缩写”形式定义过的；"
    "3) 去重，输出 JSON：{\"aliases\": [\"名称1\", \"名称2\"]}，无则输出空数组。只输出 JSON。"
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=lm_config.item_model,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        temperature=0.0,
        extra_body={"enable_thinking": False},
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def extract_aliases(item_name: str, chunks: List[Dict], max_chars: int = 3500) -> List[str]:
    """
    用 LLM 从文献前若干切片抽取该文献 / 核心方法的别名。
    :param item_name: 标准文献名
    :param chunks: 切片列表（取标题/摘要等缩写定义集中的前若干块）
    :param max_chars: 喂给 LLM 的语料字符上限
    :return: 别名列表（已清洗、去重、排除标准名）；异常时返回空列表
    """
    parts: List[str] = []
    total = 0
    for c in chunks:
        piece = f"标题：{c.get('title', '')}\n内容：{c.get('content', '')}"
        if total + len(piece) > max_chars:
            break
        parts.append(piece)
        total += len(piece)
    corpus = "\n\n".join(parts)
    user_prompt = f"文献标准名：{item_name}\n文献内容：\n{corpus}"

    try:
        resp = _get_llm().invoke([
            SystemMessage(content=ALIAS_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        data = json.loads(resp.content)
        aliases = data.get("aliases", [])
        out: List[str] = []
        for a in aliases:
            a = str(a).strip()
            if a and a != item_name and a not in out:
                out.append(a)
        return out
    except Exception:
        return []


def build_match_text(item_name: str, aliases: List[str]) -> str:
    """
    构造可检索文本：标准名 + 别名（去重、保序），用空格分隔，
    使 dense 向量与 BM25 关键词都覆盖标准名和全部别名。
    """
    seen, parts = set(), []
    for x in [item_name] + list(aliases):
        x = (x or "").strip()
        if x and x not in seen:
            seen.add(x)
            parts.append(x)
    return "  ".join(parts)
