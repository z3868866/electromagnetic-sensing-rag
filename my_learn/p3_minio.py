"""MinIO 最小场景：连接 -> 建 bucket -> 上传对象 -> 列出 -> 下载读回 -> 生成临时URL。
对象与 bucket 在本脚本后保留，便于你用浏览器/Console 查看；清理单独做。
"""
import io
import os
from dotenv import load_dotenv
from minio import Minio

load_dotenv()
# 1. 创建客户端：endpoint / 凭据从 .env 读取，默认本机 localhost
client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False,
)

# 2. 创建 bucket（桶：对象的顶层容器，名字全局唯一、不能有下划线/大写）
bucket = "practice-bucket"
if not client.bucket_exists(bucket):
    client.make_bucket(bucket)
print("bucket 是否存在：", client.bucket_exists(bucket))

# 3. 上传对象 object
# 方式A：上传内存中的字节（put_object 需要类文件对象 + length）
data = "hello MinIO 你好".encode("utf-8")
client.put_object(bucket, "hello.txt", io.BytesIO(data), length=len(data))
# 方式B：上传本地文件（fput_object = file put）
client.fput_object(bucket, "scripts/p2_mongo_crud.py", "my_learn/p2_mongo_crud.py")

# 4. 列出对象（object 的 key 可以含 /，用“前缀”模拟目录，但底层是扁平的）
print("\n--- 对象列表 ---")
for obj in client.list_objects(bucket, recursive=True):
    print(f"{obj.object_name}  ({obj.size} bytes)")

# 5. 下载并读回验证
resp = client.get_object(bucket, "hello.txt")
content = resp.read()
resp.close()
resp.release_conn()
print("\n读回 hello.txt：", content.decode("utf-8"))

# 6. 生成预签名临时 URL（presigned URL）：带签名、限时可直接访问，无需账号
url = client.presigned_get_object(bucket, "hello.txt")
print("\n临时访问 URL（复制到浏览器可直接下载）：\n", url)
