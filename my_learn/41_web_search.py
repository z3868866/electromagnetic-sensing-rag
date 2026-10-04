import asyncio
import os
import json
from dotenv import load_dotenv
from agents.mcp import MCPServerStreamableHttp

load_dotenv(r"D:\RAG\YUANLAI\.env")


async def _mcp_call(query):
    server = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            "url": os.getenv("MCP_DASHSCOPE_BASE_URL"),
            "headers": {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            "timeout": 30,
        },
        cache_tools_list=True,
        max_retry_attempts=3,
    )
    try:
        await server.connect()
        result = await server.call_tool(
            tool_name="search_pro",
            arguments={"query": query},
        )
        return result
    finally:
        await server.cleanup()      # 无论成功/失败都关闭连接，防止资源泄漏


def node_web_search(query):
    if not query:
        return {}
    result = asyncio.run(_mcp_call(query))
    pages = json.loads(result.content[0].text).get("pages") or []

    docs = []
    for item in pages:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        if not snippet:            # 过滤无正文摘要的无效结果
            continue
        docs.append({"title": title, "url": url, "snippet": snippet})
    return {"web_search_docs": docs}


if __name__ == "__main__":
    out = node_web_search("基于深度卷积神经网络的协作频谱感知方法的网络结构是什么？")
    docs = out.get("web_search_docs", [])
    print("有效结果数：", len(docs))
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d['title']}")
        print(f"   url: {d['url']}")
        print(f"   snippet: {d['snippet'][:100]}")
