import os
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

load_dotenv(r"D:\RAG\YUANLAI\.env")
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      base_url=os.getenv("OPENAI_API_BASE"))
LLM = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")

app = FastAPI(docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.18.2/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.bootcdn.net/ajax/libs/swagger-ui/5.18.2/swagger-ui.min.css",
    )


@app.get("/answer_stream")
async def answer_stream(question: str):
    async def gen():
        stream = await aclient.chat.completions.create(
            model=LLM,
            messages=[{"role": "user", "content": question}],
            temperature=0.1,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield f"data: {json.dumps(delta, ensure_ascii=False)}\n\n"
        yield 'event: final\ndata: "done"\n\n'

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
