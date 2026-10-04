"""MinIO 清理：删除 practice-bucket 内所有对象，再删除空桶。"""
import os
from dotenv import load_dotenv
from minio import Minio

load_dotenv()
client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False,
)
bucket = "practice-bucket"

# 1. 删除桶内所有对象（recursive=True 列出含“子目录”前缀的全部对象）
for obj in client.list_objects(bucket, recursive=True):
    client.remove_object(bucket, obj.object_name)
    print("删除对象：", obj.object_name)

# 2. 删除桶（桶必须为空，否则报错）
client.remove_bucket(bucket)
print("桶已删除")
print("剩余桶：", [b.name for b in client.list_buckets()])
