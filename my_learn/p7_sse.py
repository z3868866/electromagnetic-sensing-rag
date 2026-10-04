"""SSE 最小场景（自包含）：
- GET /        返回前端页面（用 EventSource 接收）
- GET /stream  SSE 接口：异步生成器每隔 1 秒 yield 一条 data: 消息
"""
import asyncio
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# ---------- SSE 数据生成：异步生成器 ----------
async def event_generator():
    for i in range(5):
        # SSE 协议：data: 内容，消息之间必须用“两个换行 \n\n”分隔
        yield f"data: 第 {i + 1} 条消息（{time.strftime('%H:%M:%S')}）\n\n"
        await asyncio.sleep(1)          # 模拟“每隔一秒产生一条”，此时服务器可服务别人
    yield "data: [DONE]\n\n"            # 自定义结束标记


@app.get("/stream")
async def stream():
    # StreamingResponse：不一次性返回，而是每 yield 一段就立刻推给客户端
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def page():
    return """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SSE 演示</title></head>
<body>
  <h3>SSE 流式演示</h3>
  <button onclick="start()">开始连接 SSE</button>
  <div id="out"></div>
  <script>
    function start() {
      const es = new EventSource("/stream");   // 浏览器发起并维持长连接
      es.onmessage = (e) => {                   // 每收到一条 data 消息触发
        if (e.data === "[DONE]") { es.close(); return; }  // 收到结束标记，关闭连接
        document.getElementById("out").innerHTML += e.data + "<br>";
      };
    }
  </script>
</body></html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8020)
