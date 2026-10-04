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


class QueryRequest(BaseModel):
    session_id: str
    question: str


task_queues = {}


async def query_task(session_id: str, question: str):
    queue = asyncio.Queue()
    task_queues[session_id] = queue
    await queue.put({"event": "progress", "data": "正在解析问题…"})
    await asyncio.sleep(1)
    await queue.put({"event": "progress", "data": "正在检索知识库…"})
    await asyncio.sleep(1)
    await queue.put({"event": "progress", "data": "正在Rerank精排…"})
    await asyncio.sleep(1)
    await queue.put({"event": "answer",
                     "data": "DCNN通过添加捷径连接（shortcut connection）实现恒等映射，"})
    await asyncio.sleep(0.5)
    await queue.put({"event": "answer",
                     "data": "以残差学习方式训练，使梯度可沿捷径直接回传，从而缓解梯度消失。"})
    await asyncio.sleep(0.5)
    await queue.put({"event": "complete", "data": "回答完成"})
    await queue.put(None)


@app.post("/query")
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(query_task, req.session_id, req.question)
    return {"session_id": req.session_id, "status": "已提交"}


@app.get("/stream/{session_id}")
async def stream_result(session_id: str):
    async def gen():
        while session_id not in task_queues:
            await asyncio.sleep(0.1)
        q = task_queues[session_id]
        while True:
            item = await q.get()
            if item is None:
                break
            yield f"event: {item['event']}\ndata: {item['data']}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
