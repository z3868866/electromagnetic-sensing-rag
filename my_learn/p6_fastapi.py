"""FastAPI 最小场景：定义 GET / GET带查询参数 / POST请求体 三个接口。
启动后用浏览器和 /docs 自动文档访问。
"""
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="最小 FastAPI 演示")  # FastAPI 应用对象（框架）


@app.get("/")                      # 路由：把 GET + 路径 "/" 绑定到下面函数
def home():
    return {"message": "你好，FastAPI"}   # 返回 dict，FastAPI 自动序列化为 JSON


@app.get("/hello")
def hello(name: str = "world"):    # 查询参数：/hello?name=DCNN，默认 world
    return {"hello": name}


class AddRequest(BaseModel):       # 请求体模型：Pydantic 自动校验类型
    a: int
    b: int


@app.post("/add")
def add(req: AddRequest):
    return {"a": req.a, "b": req.b, "sum": req.a + req.b}


if __name__ == "__main__":
    # Uvicorn 是真正监听端口、运行 app 的 ASGI 服务器
    uvicorn.run(app, host="127.0.0.1", port=8010)
