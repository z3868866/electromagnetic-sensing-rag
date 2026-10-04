import sys
sys.path.insert(0, r"D:\RAG\YUANLAI")
from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from config.lm_config import lm_config
from processor.query_processor.prompt.item_name_confirm import (
    ITEM_NAME_EXTRACT_TEMPLATE, ITEM_NAME_EXTRACT_SYSTEM_PROMPT)

q = "基于深度卷积神经网络的协作频谱感知方法是如何缓解梯度消失的？"
llm = ChatOpenAI(
    model=lm_config.item_model,
    api_key=lm_config.api_key,
    base_url=lm_config.base_url,
    temperature=lm_config.llm_temperature,
    model_kwargs={"response_format": {"type": "json_object"}}
)
up = ITEM_NAME_EXTRACT_TEMPLATE.format(history_text="", query=q)
resp = llm.invoke([
    SystemMessage(content=ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
    HumanMessage(content=up)
])
print("RAW CONTENT START")
print(resp.content)
print("RAW CONTENT END")
