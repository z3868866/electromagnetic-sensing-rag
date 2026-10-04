# 默认名字是：root
import os
import logging
from logging.handlers import TimedRotatingFileHandler

import colorlog

# 初始化全局日志对象
logger = logging.getLogger()
# 设置日志的默认的级别
logger.setLevel(logging.DEBUG)

# 统一日志格式
_FMT = '%(asctime)s - %(filename)s : %(lineno)d - %(levelname)s - %(message)s'
_DATEFMT = '%Y-%m-%d %H:%M:%S'

# 1) 控制台彩色日志处理器
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s' + _FMT,
    datefmt=_DATEFMT,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',  # INFO 显示为绿色
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))
logger.addHandler(handler)

# 2) 文件日志处理器（按天轮转，保留 15 天，UTF-8，避免颜色码写入文件）
LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs"
)
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"),
    when="midnight",
    backupCount=15,
    encoding="utf-8",
)
file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
file_handler.setLevel(logging.INFO)  # 文件只记录 INFO 及以上
logger.addHandler(file_handler)
