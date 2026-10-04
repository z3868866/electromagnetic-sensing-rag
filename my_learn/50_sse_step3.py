import asyncio
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

task_queues = {}   # {session_id: asyncio.Queue}


async def long_task(session_id: str):
    queue = asyncio.Queue()
    task_queues[session_id] = queue
    for i in range(5):
        msg = f"会话{session_id}处理结果{i + 1}"
        await queue.put(msg)
        await asyncio.sleep(1)
    await queue.put(None)          # 结束哨兵


@app.get("/submit/{session_id}")
async def submit_task(session_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(long_task, session_id)
    return {"message": "任务已启动", "session_id": session_id}


@app.get("/stream/{session_id}")
async def stream_result(session_id: str):
    async def event_generator():
        while session_id not in task_queues:
            await asyncio.sleep(0.1)          # 等队列创建（竞态保护）
        queue = task_queues[session_id]
        while True:
            msg = await queue.get()           # 空时异步挂起
            if msg is None:
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
