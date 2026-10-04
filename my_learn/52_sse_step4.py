import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class QueryRequest(BaseModel):   # 请求体模型
    session_id: str
    question: str


task_queues = {}


async def process_query(session_id: str, question: str):
    queue = asyncio.Queue()
    task_queues[session_id] = queue
    await queue.put(f"已收到问题：{question}")
    await asyncio.sleep(1)
    await queue.put("正在检索知识库并生成答案…")
    await asyncio.sleep(1)
    await queue.put(f"针对「{question}」的模拟答案")
    await queue.put(None)


@app.post("/query")
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_query, req.session_id, req.question)
    return {"session_id": req.session_id, "status": "任务已提交"}


@app.get("/stream/{session_id}")
async def stream_result(session_id: str):
    async def event_generator():
        while session_id not in task_queues:
            await asyncio.sleep(0.1)
        queue = task_queues[session_id]
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
