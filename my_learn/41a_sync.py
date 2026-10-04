import time

def task(name):
    print(f"{name} 开始")
    time.sleep(2)          # 模拟一次耗时2秒的网络IO
    print(f"{name} 结束")

start = time.time()
task("A")
task("B")
task("C")
print("总耗时：", round(time.time() - start, 1), "秒")
