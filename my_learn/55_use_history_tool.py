import os
import sys
# 把项目根加入模块搜索路径，使 import utils 可用（脚本在 my_learn 子目录）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mongo_history_utils import (
    save_chat_message, get_recent_messages, clear_history
)

sid = "learn55_demo"

# 1. 用成品封装写入 user / assistant
uid = save_chat_message(sid, "user", "DCNN 如何缓解梯度消失？",
                        rewritten_query="DCNN 如何缓解梯度消失？",
                        item_names=[])
aid = save_chat_message(sid, "assistant", "通过捷径连接以残差学习方式缓解。")
print("写入主键：", uid, "|", aid)

# 2. 用成品封装读回
msgs = get_recent_messages(sid, limit=10)
print("读回条数：", len(msgs))
for m in msgs:
    print(f"[{m['role']}] {m['text']} | item_names={m.get('item_names')}")

# 3. 清空该会话
print("清理条数：", clear_history(sid))
