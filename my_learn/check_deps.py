"""检查 Web 集成（FastAPI + SSE + MongoDB）所需依赖是否已安装。"""
import importlib
import importlib.metadata as md

# 发行版名称 -> import 名称
PKGS = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "openai": "openai",
    "dashscope": "dashscope",
    "pymilvus": "pymilvus",
    "pymongo": "pymongo",
    "python-dotenv": "dotenv",
    "openai-agents": "agents",
    "motor": "motor",
    "sse-starlette": "sse_starlette",
    "httpx": "httpx",
}

if __name__ == "__main__":
    for dist, imp in PKGS.items():
        try:
            ver = md.version(dist)
            importlib.import_module(imp)
            print(f"OK    {dist:15} {ver}")
        except md.PackageNotFoundError:
            print(f"MISS  {dist:15} 未安装")
        except Exception as e:
            print(f"ERR   {dist:15} 已安装但导入失败: {e}")
