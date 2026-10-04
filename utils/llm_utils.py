# utils/llm_utils.py

from langchain_openai import ChatOpenAI

from config.lm_config import lm_config


'''
这个函数就是一个"带缓存的大模型客户端工厂"。你告诉它用哪个模型、要不要 JSON 模式，它先查缓存，有就直接给你，没有就创建一个存起来再给你。不同节点可以用不同模型，但同一个配置只会创建一次客户端。
'''

# 键:(模型名, 是否是json模式)
# 值:模型客户端对象
_llm_client_cache = {}


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 客户端实例
    - model: 允许不同节点使用不同模型
    - json_mode: True 时要求输出 JSON
    """
    m = model or lm_config.llm_model
    key = (m, json_mode)
    if key in _llm_client_cache:
        return _llm_client_cache[key]

    extra_body = {"enable_thinking": False}

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatOpenAI(
        model=m,
        temperature=lm_config.llm_temperature,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        extra_body=extra_body,
        model_kwargs=model_kwargs,
    )
    _llm_client_cache[key] = client
    return client

if __name__ == "__main__":
    client = get_llm_client()
    print(client)

    client = get_llm_client()
    print(client)

    client = get_llm_client("qwen-max", True)
    print(client)