import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号

'''
SSE（Server-Sent Events） 是一种"服务器主动推消息给浏览器"的技术。
它解决一个很实际的问题：**RAG 问答要跑十几个节点、耗时十几秒，不能让用户对着转圈干等**。这段代码工程含量不低，是简历里 "全栈 RAG" 的好素材。
'''
# 全局 SSE 会话队列存储
# Key: session_id, Value: queue.Queue

_session_stream: Dict[str, "queue.Queue"] = {}

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _session_stream[session_id] = q
    return q

def remove_sse_queue(session_id: str):
    """移除指定 session 的队列"""
    _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})

async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        # 如果没有对应的队列，直接结束
        return

    loop = asyncio.get_running_loop()
    try:
        # 发送连接建立信号
        yield _sse_pack("ready", {})

        while True:             #只要连接还在，就一直监听队列。
            # 若客户端断开，尽快退出
            if await request.is_disconnected():
                break

            try:
                # 使用 run_in_executor 避免阻塞 async 事件循环
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            # 特殊关闭事件
            if event == "__close__":
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消/对端断开：静默退出
        return
    finally:
        # 清理资源
        remove_sse_queue(session_id)

'''


**Q1：为什么用 SSE 而不是普通 HTTP 一次性返回？它和 WebSocket 有什么区别？**

A：RAG 问答链路很长，要经过产品确认、向量检索、融合、重排、LLM 生成，可能耗时十几秒。
如果用普通 HTTP 憋到最后一次性返回，用户只能对着加载圈干等，体验很差也不知道系统是否在工作。
SSE（Server-Sent Events）是服务器单向推送技术，连接建立后服务器可以持续把"正在确认产品/正在检索/正在重排"的节点进度，以及 LLM 的流式 token 一条条推给前端，实现进度条加打字机效果。
它和 WebSocket 的区别是：SSE 是单向的（只有服务器到浏览器）、基于普通 HTTP、轻量、浏览器原生支持；WebSocket 是全双工双向通信、协议更重。本场景只需要服务器往外推，所以 SSE 更合适。

---

**Q2：为什么要给每个 session 建一个队列、存进全局字典？**

A：为了多用户并发隔离。
系统会同时服务多个提问的用户，如果共用一个队列，A 用户会收到 B 用户的检索进度和答案，会话内容互相串。
我用 `{session_id: queue.Queue}` 的全局字典管理，每个会话拥有独立队列，后台流水线只往当前 session_id 的队列推事件，SSE 生成器只消费自己会话的队列，实现多用户之间的通道隔离。
队列同时承担了跨线程传递消息的作用。

---

**Q3：queue.Queue.get() 是阻塞的，为什么不能直接在 async 生成器里调用？run_in_executor 解决了什么？**

A：因为 asyncio 事件循环用一个线程并发服务所有用户，而 queue.Queue.get() 在队列没消息时会阻塞当前线程。
如果在 async 生成器里直接调用，事件循环就会卡在这一次等待上，导致其他所有用户的请求都被堵死，并发能力直接失效。
解决办法是用 `await loop.run_in_executor(None, queue.get, True, 1.0)`，把这个阻塞操作丢到线程池里执行，事件循环本身立刻释放、继续服务别的请求；
同时设置 1 秒超时，借超时回到循环定期检查 request.is_disconnected()，一旦用户关闭页面就及时退出并清理资源。
核心原则是：async 代码里绝不能直接写会长时间阻塞的同步调用，否则一个慢请求会拖垮所有请求。

---

**Q4：SSE 一条消息的格式长什么样？结尾为什么要两个换行？**

A：SSE 是纯文本协议，一条消息由 event 行和 data 行组成，格式是 `event: 事件名\ndata: JSON字符串\n\n`。
data 里放 JSON 序列化后的业务数据，比如 `event: delta\ndata: {"text":"HAK180"}\n\n`。
结尾必须是两个换行（一个空行），这是 SSE 协议规定的消息分隔符，浏览器的 EventSource 靠这个空行判断一条消息完整结束、可以分发给前端回调；
少了这个空行，浏览器会认为消息还没收完，一直缓冲不触发事件。

---


'''