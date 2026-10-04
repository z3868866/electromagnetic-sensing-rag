import asyncio
import time

async def task(name):
    print(f"{name} 开始")
    await asyncio.sleep(2)   # 异步等待：暂停本任务、让出控制权
    print(f"{name} 结束")

async def main():
    start = time.time()
    await asyncio.gather(task("A"), task("B"), task("C"))  # 三个任务并发调度
    print("总耗时：", round(time.time() - start, 1), "秒")

asyncio.run(main())
