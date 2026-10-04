"""MongoDB 最小场景：连接 -> 选择库/集合 -> 插入文档 -> 查询。
数据在本脚本结束后保留，便于你观察；update/delete 在下一个脚本。
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pprint import pprint

load_dotenv()
# 1. 连接 MongoDB：地址从 .env 的 MONGO_URL 读取，默认本机 localhost
client = MongoClient(
    os.getenv("MONGO_URL", "mongodb://localhost:27017"),
    serverSelectionTimeoutMS=3000,
)
print("连接是否成功：", client.admin.command("ping"))

# 2. 选择数据库与集合（库/集合不存在时无需手动建，第一次写入会自动创建）
db = client["practice_db"]      # 数据库 database
col = db["papers"]              # 集合 collection（类似“表”）

# 3. 插入一条文档：文档 document 就是一个 Python dict，可任意嵌套
doc = {
    "title": "基于深度卷积神经网络的协作频谱感知方法",
    "year": 2021,
    "tags": ["频谱感知", "DCNN", "残差学习"],
    "author": {"name": "张三", "org": "某大学"},
}
result = col.insert_one(doc)
print("插入文档自动生成的 _id：", result.inserted_id)

# 再插入多条，体会同一集合里文档结构可以不同（无固定模式 schema-free）
col.insert_many([
    {"title": "分布式异构频谱感知", "year": 2024},
    {"title": "一篇没有年份字段的文档", "note": "我没有 year 字段也能存"},
])
print("当前集合文档总数：", col.count_documents({}))

# 4. 查询
print("\n--- find_one：按条件查单条 ---")
pprint(col.find_one({"year": 2021}))

print("\n--- find：按条件查多条（这里查 tags 里含 DCNN 的）---")
for d in col.find({"tags": "DCNN"}):
    pprint(d)

print("\n--- find 全量，只返回 title 字段（投影 projection，1 显示 / 0 排除）---")
for d in col.find({}, {"_id": 0, "title": 1}):
    print(d)
