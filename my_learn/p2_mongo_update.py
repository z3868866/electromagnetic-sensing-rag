"""MongoDB 最小场景（二）：更新 update 与删除 delete，最后清理练习库。"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pprint import pprint

load_dotenv()
client = MongoClient(
    os.getenv("MONGO_URL", "mongodb://localhost:27017"),
    serverSelectionTimeoutMS=3000,
)
col = client["practice_db"]["papers"]

# 1. 更新 update_one(条件, 更新操作)
# $set：设置字段；字段不存在就新增（这里给“分布式异构”补 year 和 venue）
col.update_one(
    {"title": "分布式异构频谱感知"},
    {"$set": {"year": 2025, "venue": "某期刊"}},
)
# $inc：数值自增；$push：向数组字段追加一个元素
col.update_one(
    {"title": "基于深度卷积神经网络的协作频谱感知方法"},
    {"$inc": {"citation": 1}, "$push": {"tags": "协作"}},
)
# $unset：删除某个字段
col.update_one(
    {"title": "一篇没有年份字段的文档"},
    {"$unset": {"note": ""}},
)

print("--- 更新后的全部文档 ---")
for d in col.find({}, {"_id": 0}):
    pprint(d)

# 2. 删除 delete_many(条件)；{} 表示删除全部
res = col.delete_many({})
print("删除文档条数：", res.deleted_count)

# 3. 删掉整个练习数据库，保持环境干净
client.drop_database("practice_db")
print("practice_db 已删除，剩余库：", client.list_database_names())
