import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(r"D:\RAG\YUANLAI\.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_API_BASE"))
LLM = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")

RELEVANCE_FLOOR = 0.5   # 相关性下限：Top1 低于此值则拒答

ANSWER_PROMPT = """你是一个智能助手，请严格依据【参考内容】回答【用户问题】。
要求：
1. 仅基于参考内容作答，不得编造参考内容中不存在的事实。
2. 若参考内容不足以回答问题，必须直接回复“抱歉，知识库中未找到相关内容。”，不得强行作答。
3. 如需图片辅助，在答案末尾另起一行写“【图片】”，每行一个图片URL，且图片只能来自参考内容。

【参考内容】
{context}

【历史对话】
{history}

【相关实体】
{item_names}

【用户问题】
{question}

请回答："""


def build_context(docs, max_chars=12000):
    lines, used = [], 0
    for i, d in enumerate(docs, 1):
        entry = f"[{i}] [source={d.get('source')}] [score={round(d.get('score',0),3)}]\n{d.get('content','')}"
        if used + len(entry) > max_chars:
            break
        lines.append(entry)
        used += len(entry) + 2
    return "\n\n".join(lines)


def node_answer(state):
    # Step1 已有 answer（反问/拒识）→ 直接输出
    if state.get("answer"):
        return state["answer"]

    docs = state.get("reranked_docs", [])

    # Step2 绝对低分 floor → 拒答
    if not docs or docs[0].get("score", 0) < RELEVANCE_FLOOR:
        return "抱歉，知识库中未找到相关内容。"

    # Step3 构建 prompt
    prompt = ANSWER_PROMPT.format(
        context=build_context(docs),
        history="暂无历史对话",
        item_names=", ".join(state.get("item_names", [])) or "无",
        question=state.get("rewritten_query") or state.get("original_query", ""),
    )

    # Step4 非流式生成
    resp = client.chat.completions.create(
        model=LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


if __name__ == "__main__":
    # 场景1：有高分资料 → 正常生成
    s1 = {
        "original_query": "DCNN如何解决梯度消失？",
        "item_names": ["基于深度卷积神经网络的协作频谱感知方法"],
        "reranked_docs": [
            {"source": "local", "score": 0.95,
             "content": "该网络通过添加捷径连接实现恒等映射，以残差学习方式训练，从而缓解深层网络的梯度消失问题。"},
            {"source": "web", "score": 0.8,
             "content": "残差结构使梯度可通过捷径直接回传。"},
        ],
    }
    print("场景1：\n", node_answer(s1), "\n", "=" * 40)

    # 场景2：Top1 低于 floor → 拒答
    s2 = {"original_query": "今天天气", "item_names": [],
          "reranked_docs": [{"source": "web", "score": 0.2, "content": "无关内容"}]}
    print("场景2：\n", node_answer(s2), "\n", "=" * 40)

    # 场景3：已有 answer（节点1反问）→ 直出
    s3 = {"answer": "您是想问以下哪个产品：A、B？请明确型号。"}
    print("场景3：\n", node_answer(s3))
