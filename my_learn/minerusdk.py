# 你的手写版 ≈ 60 行：拼 URL、两层校验、while 轮询、sleep、解压
# SDK 版长这样：
from mineru import MinerU

client = MinerU()                       # 自动读 MINERU_TOKEN 环境变量
result = client.parse("test_5g.pdf")    # 上传、轮询、等待，一行搞定
markdown = result.markdown              # 直接拿到 Markdown 文本