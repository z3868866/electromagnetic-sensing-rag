import json
import logging

from minio import Minio

from config.minio_config import minio_config
from processor.import_processor.base import setup_logging

setup_logging()

try:
    # 1.创建客户端链接对象
    minio_client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        #是否强制启用HTTPS的加密链接方式，False：使用http；True：使用https
        secure=False
    )

    # 2.判断bucket是否存在，不存在则创建
    found = minio_client.bucket_exists(minio_config.bucket_name)
    if not found:
        minio_client.make_bucket(minio_config.bucket_name)

    # 3. 定义当前bucket的访问权限
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "s3:GetObject",
                "Resource": f"arn:aws:s3:::{minio_config.bucket_name}/*",
            },
        ],
    }
    # 4. 设置当前bucket的访问权限
    minio_client.set_bucket_policy(minio_config.bucket_name, json.dumps(policy))

except Exception as e:
    logging.error(f"MinIO连接失败，错误原因：{e}")

def get_minio_client():
    return minio_client

if __name__ == '__main__':
    get_minio_client()


'''


**Q1：bucket 是什么？为什么要先 bucket_exists 判断再决定是否创建？**

A：bucket（桶）是对象存储里的顶层容器，相当于一个"盘"或项目根目录，所有图片都存在指定桶里，再按文档名分子目录。
先判断再创建是幂等初始化：程序第一次运行时桶不存在就自动创建，以后重复运行时桶已经存在就直接跳过，不会因为"桶已存在"报错。这样程序无论启动多少次都能安全执行，不会产生重复资源。

幂等初始化 = "不管调几次，效果都一样"。核心手段就是"先检查，再操作"——已经做过的就跳过，没做过的才执行。
---

**Q2：桶默认是私有的，如果不设置 bucket policy，用户浏览器访问图片 URL 会怎样？**

A：会被拒绝访问，返回 403 Forbidden，答案里的图片全部显示成裂图。
因为 MinIO 桶默认需要带账号密钥鉴权才能访问，而用户浏览器不可能携带我的服务端密钥。
所以必须通过 set_bucket_policy 给桶设置匿名可读策略，让图片的 HTTP URL 能被浏览器不带凭证直接 GET 到，这也是为什么要把 Markdown 里的本地图片路径替换成 MinIO URL。

---

**Q3：策略里 Principal 设为所有人（*），但 Action 只给 s3:GetObject，这样安全吗？为什么不给写权限？**

A：安全，这正是最小权限原则。
Principal 为 * 表示任何人可以匿名访问，但 Action 严格限定为 s3:GetObject，即只能读取、下载对象；没有授予 s3:PutObject 上传、DeleteObject 删除等写权限，Resource 也限定在本桶范围内。
所以匿名用户"能看不能改"，图片可以公开显示，而仓库不会被匿名上传垃圾文件或删除数据。如果是生产环境还会配合 HTTPS、独立桶和更细的目录策略进一步加固。

---

**Q4：这段代码在 MinIO 服务没启动时，get_minio_client 会出什么问题？怎么修？**

A：会有一个隐患：连接代码包在 try 里，如果 Minio() 构造或 bucket_exists 抛异常，except 只打了一条日志，而 minio_client 这个变量根本没有被成功赋值；
之后任何地方调用 get_minio_client()，return 一个不存在的变量会直接抛 NameError，错误被延迟、还很难排查。
修法是先用 _minio_client = None 占位，try 成功才赋值，get_minio_client 失败时返回 None，由调用方判空并降级（比如跳过图片处理但不崩主流程），而不是让程序在远处莫名崩溃。
更优雅的方式是改成懒加载，第一次使用时才建立连接。

|                        | 饿加载         |                懒加载 |

| 什么时候初始化           | 模块导入时（`import` 那一行） |    第一次调用时（`get_xxx()` 那一行） |
| MinIO 没启动会怎样      | 整个程序崩掉 |                   程序正常启动，用到 MinIO 时才报错 |
| 适合场景                 | 这个服务一定会被用到 |           这个服务可能不会被用到 |


---


'''