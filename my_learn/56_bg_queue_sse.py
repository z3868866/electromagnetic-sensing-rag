import asyncio
import json
import queue
import time

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

app = FastAPI()

# 1. 全局"传菜窗口"字典：session_id -> queue.Queue
_queues = {}


# 2. 后台任务：同步函数，FastAPI 在后台工作线程执行（厨师）
def background_run(sid: str):
    q = _queues[sid]
    q.put({"event": "progress", "data": {"done_list": ["确认问题文献"]}})
    time.sleep(1)
    for word in ["捷径", "连接", "以残差学习缓解梯度消失"]:
        q.put({"event": "delta", "data": {"text": word}})
        time.sleep(1)
    q.put({"event": "final", "data": {"answer": "通过捷径连接以残差学习缓解梯度消失"}})
    q.put({"event": "__close__", "data": {}})  # 关闭信号


class QueryBody(BaseModel):
    query: str


# 3. POST /query：建窗口、注册后厨任务、立即返回
@app.post("/query")
def submit(req: QueryBody, bg: BackgroundTasks):
    sid = f"learn56_{int(time.time() * 1000)}"
    _queues[sid] = queue.Queue()
    bg.add_task(background_run, sid)
    return {"session_id": sid, "message": "处理中"}


def pack(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# 4. GET /stream：事件循环线程，派帮工取窗口，每秒自查断开
@app.get("/stream/{sid}")
async def stream(sid: str, request: Request):
    async def gen():
        q = _queues.get(sid)
        if q is None:
            return
        loop = asyncio.get_running_loop()
        yield pack("ready", {})
        while True:
            if await request.is_disconnected():      # 服务员亲自看顾客在不在
                break
            try:
                # 派帮工盯窗口，最多 1 秒
                msg = await loop.run_in_executor(None, q.get, True, 1.0)
            except queue.Empty:
                continue                            # 空手回，重新检查
            event = msg["event"]
            if event == "__close__":
                break
            yield pack(event, msg["data"])
        _queues.pop(sid, None)                      # 收拾窗口

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )


# 5. 极简前端页面
@app.get("/demo", response_class=HTMLResponse)
def demo():
    return """
    <!doctype html><meta charset="utf-8">
    <button id="b">提问</button>
    <div id="out" style="margin-top:12px;line-height:1.8;"></div>
    <script>
    b.onclick = async () => {
      out.innerHTML = "";
      const r = await fetch("/query", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({query: "DCNN"})
      });
      const {session_id} = await r.json();
      const es = new EventSource("/stream/" + session_id);
      ["ready", "progress", "delta", "final"].forEach(ev =>
        es.addEventListener(ev, e => {
          out.innerHTML += "<b>" + ev + "</b>: " + e.data + "<br>";
        })
      );
      es.onerror = () => es.close();
    };
    </script>"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)   # 用 8002，避开成品 8001
