import os
import time
from dotenv import load_dotenv
from pymongo import MongoClient, DESCENDING

load_dotenv(r"D:\RAG\YUANLAI\.env")

# 1. 建立连接：MongoClient 读取连接串，db 指向 .env 指定的数据库
client = MongoClient(os.getenv("MONGO_URL"))
db = client[os.getenv("MONGO_DB_NAME")]
col = db["chat_message"]              # 对话集合

# 2. 唯一会话 ID，避免与已有数据混淆
sid = "learn54_" + str(int(time.time()))

# 3. 写入两条对话（user / assistant），ts 显式递增
col.insert_one({"session_id": sid, "role": "user",
                "text": "DCNN 是如何缓解梯度消失的？", "ts": time.time()})
time.sleep(0.01)
col.insert_one({"session_id": sid, "role": "assistant",
                "text": "通过捷径连接以残差学习方式缓解。", "ts": time.time()})

# 4. 取最近消息：先按 ts 降序取最新 N 条，再 reverse 成时间正序（从旧到新）
msgs = list(col.find({"session_id": sid}).sort("ts", DESCENDING).limit(10))
msgs.reverse()
print("读回条数：", len(msgs))
for m in msgs:
    print(f"[{m['role']}] {m['text']}")

# 5. 删除本次练习数据，保持集合干净
print("已清理条数：", col.delete_many({"session_id": sid}).deleted_count)
