import os, json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv(r"D:\RAG\YUANLAI\.env")

llm = ChatOpenAI(
    model=os.getenv("LLM_DEFAULT_MODEL", "qwen-flash"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    temperature=0.1
)

PROMPT_TEMPLATE = """历史会话：
{history_text}

当前用户问题：{query}

请根据历史会话和当前问题：
1. item_names 只提取"具有明确名称、可能在文献知识库中检索到的具体实体"，例如论文名、方法名、模型名、系统名，可能有一个或多个、不重复。
   - 若使用"这个/它"等代词，请结合历史做指代消解，还原为具体实体名；
   - 对于问候、闲聊、天气、以及"人生/时间"这类泛指的抽象概念（不是一个具体可检索的实体），item_names 必须返回空列表 []。

2. 改写用户问题（rewritten_query），使其成为包含主体名称、脱离上下文也能理解的完整问题。

请只返回 JSON，格式：
{{"item_names": ["主体名"], "rewritten_query": "改写后的完整问题"}}"""


def extract_info(query, history=None):
    history = history or []
    history_text = "\n".join(f"{m['role']}: {m['text']}" for m in history)
    prompt = PROMPT_TEMPLATE.format(history_text=history_text or "（无历史）", query=query)
    messages = [
        SystemMessage(content="你是专业的检索助手，擅长意图理解、指代消解和问题改写。"),
        HumanMessage(content=prompt)
    ]
    try:
        content = llm.invoke(messages).content.strip()
        if content.startswith("```"):                      # 清洗可能的代码块包裹
            content = content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        data.setdefault("item_names", [])
        data.setdefault("rewritten_query", query)
        return data
    except Exception as e:
        print("提取失败：", e)
        return {"item_names": [], "rewritten_query": query}


if __name__ == "__main__":
    print("场景1（明确问题）：", extract_info("DCNN 是如何解决梯度消失问题的？"))

    history = [
        {"role": "user", "text": "基于深度卷积神经网络的协作频谱感知方法这篇论文讲了什么？"},
        {"role": "assistant", "text": "它提出了用 DCNN 做协作频谱感知。"}
    ]
    print("场景2（代词，需指代消解）：", extract_info("那它的网络结构是什么？", history))

    print("场景3（无主体）：", extract_info("今天天气怎么样？"))
