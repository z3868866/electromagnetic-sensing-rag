
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING

'''
这段代码就是对话记录的 CRUD（增删改查）：save 存消息、update 补商品名、get 查最近对话、clear 清空历史。
底层用 MongoDB 存储，用 session_id 区分不同会话，
用 ts 时间戳排序。整个类用单例模式管理数据库连接，索引优化了"按会话查最新消息"的查询性能。
'''

# 加载.env文件中的环境变量，使os.getenv能读取到配置
load_dotenv()

# 封装 MongoDB 连接（只干一件事：建连接、建索引）
class HistoryMongoTool:
    """
    MongoDB 历史对话记录读写工具类 (基于原生 PyMongo 实现)
    核心功能：封装MongoDB的连接、集合初始化、索引创建，为上层提供统一的数据库操作入口
    扩展功能：支持与LangChain消息对象的格式转换（原代码预留能力）
    """
    def __init__(self):
        """
        类初始化方法：完成MongoDB的连接、数据库/集合获取、索引创建
        初始化失败会抛出异常并记录错误日志，确保程序感知连接问题
        """
        try:
            # 从环境变量读取MongoDB连接地址（敏感配置，不硬编码）
            self.mongo_url = os.getenv("MONGO_URL")
            # 从环境变量读取要使用的数据库名称
            self.db_name = os.getenv("MONGO_DB_NAME")

            # 创建MongoDB客户端实例，建立与数据库的连接
            self.client = MongoClient(self.mongo_url)
            # 获取指定名称的数据库对象
            self.db = self.client[self.db_name]
            # 获取对话记录的集合（相当于关系型数据库的表），集合名：chat_message
            self.chat_message = self.db["chat_message"]

            # 为chat_message集合创建复合索引，提升查询性能
            # 索引规则：session_id升序 + ts降序，适配"按会话查最新记录"的核心查询场景
            # create_index自带幂等性：索引已存在时不会重复创建，无需额外判断
            self.chat_message.create_index([("session_id", 1), ("ts", -1)])

            # 记录成功日志，确认数据库连接和初始化完成
            logging.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            # 捕获所有初始化异常，记录详细错误日志
            logging.error(f"Failed to connect to MongoDB: {e}")
            # 重新抛出异常，让调用方感知初始化失败，避免使用未初始化的实例
            raise


# 定义全局变量：存储HistoryMongoTool的单例实例
# 目的：将数据库连接的初始化提前到模块加载阶段，避免第一次调用接口时才建立连接（提升首次响应速度）
_history_mongo_tool = HistoryMongoTool()

def get_history_mongo_tool() -> HistoryMongoTool:
    """
    获取HistoryMongoTool的单例实例（懒加载模式）
    核心逻辑：全局实例为空时创建，不为空时直接返回，保证整个程序只有一个数据库连接实例
    :return: HistoryMongoTool的单例实例
    """
    # 声明使用全局变量，避免函数内视为局部变量
    global _history_mongo_tool
    # 懒加载：仅当全局实例为空时，才创建新的实例
    if _history_mongo_tool is None:
        _history_mongo_tool = HistoryMongoTool()
    # 返回单例实例
    return _history_mongo_tool

def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: List[str] = None,
        image_urls: List[str] = None,
        message_id: str = None
) -> str:
    """
    写入/更新单条会话记录到MongoDB
    支持两种模式：无message_id时新增记录，有message_id时更新已有记录
    :param session_id: 会话唯一标识，关联对话所属的会话
    :param role: 消息角色，固定值：user（用户）/assistant（助手）
    :param text: 对话核心内容，用户的提问或助手的回答
    :param rewritten_query: 重写后的查询语句（可选，用于检索增强等场景，默认空字符串）
    :param item_names: 关联的商品名称列表（可选，支持多商品，默认None）
    :param image_urls: 关联的图片URL列表（可选，默认None）
    :param message_id: 记录主键ID（可选，有值则更新，无值则新增）
    :return: 插入/更新的记录唯一标识（新增返回ObjectId字符串，更新返回传入的message_id）
    """
    # 生成当前时间的时间戳（秒级），用于记录消息的创建时间，后续用于排序和查询
    ts = datetime.now().timestamp()

    # 构造要插入/更新的文档数据（MongoDB的基本数据单元是文档，类似Python字典）
    document = {
        "session_id": session_id,  # 会话ID，关联维度
        "role": role,  # 消息角色
        "text": text,  # 消息内容
        "rewritten_query": rewritten_query or "",  # 问题优化后的改写，空值处理为空字符串
        "item_names": item_names,  # 关联商品名称列表
        "image_urls": image_urls,  # 关联图片URL列表
        "ts": ts  # 时间戳，排序和时间筛选维度
    }

    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    # 判断是否传入主键ID，区分更新/新增逻辑
    '''
    新增：你在聊天软件里发了一条新消息，系统给你生成一个消息ID，存起来
    更新：你编辑了一条已发送的消息（比如微信的"撤回重发"），用原来的消息ID找到它，改内容
    '''
    if message_id:
        # 有message_id：执行更新操作（根据主键更新）
        result = mongo_tool.chat_message.update_one(
            {"_id": ObjectId(message_id)},  # 更新条件：主键匹配（需将字符串转为ObjectId类型）
            {"$set": document}  # 更新操作：$set表示只更新指定字段，保留其他字段
        )
        # 更新操作返回传入的message_id作为标识
        return message_id
    else:
        # 无message_id：执行新增操作
        result = mongo_tool.chat_message.insert_one(document)
        # 新增操作返回插入的ObjectId并转为字符串，便于上层使用（避免直接返回ObjectId对象）
        return str(result.inserted_id)


def update_message_item_names(ids: List[str], item_names: List[str]) -> int:
    """
    对话过程中，系统识别出用户提到了哪些商品，然后回头更新之前的对话记录，把商品名补上去。

    批量更新历史会话记录的关联商品名称
    :param ids: 要更新的记录主键ID列表（字符串类型）
    :param item_names: 要设置的新商品名称列表
    :return: 实际更新的文档数量，更新失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    try:
        # 将字符串类型的主键列表转为MongoDB的ObjectId类型（数据库中主键是ObjectId类型）
        object_ids = [ObjectId(i) for i in ids]
        # 执行批量更新操作
        result = mongo_tool.chat_message.update_many(
            # 更新条件：复合条件，同时满足
            {
                "_id": {"$in": object_ids}# 主键在指定的ID列表中（批量筛选） $in 就是"在……里面"的意思。相当于 SQL 的 WHERE _id IN (id1, id2, id3)。一次更新多条记录。
            },
            {"$set": {"item_names": item_names}}  # 更新操作：设置新的商品名称列表
        )
        # 记录更新成功日志，包含更新数量和新的商品名称
        logging.info(f"Updated {result.modified_count} records to item_names: {item_names}")
        # 返回实际更新的数量（modified_count：真正被修改的文档数，区别于matched_count）
        return result.modified_count
    except Exception as e:
        # 捕获批量更新异常，记录错误日志
        logging.error(f"Error updating history item_names: {e}")
        # 异常时返回0，标识更新失败
        return 0

def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近N条对话记录，返回原始字典格式
    结果按时间正序排列，可直接喂给LLM作为上下文    打个比方：就像你把聊天记录截图发给朋友看，肯定是从上往下（从旧到新）截，不会从最新的开始截。
    :param session_id: 会话唯一标识，用于筛选指定会话的记录
    :param limit: 条数限制，默认返回最近10条
    :return: 对话记录列表（字典格式），查询失败返回空列表
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式
    mongo_tool = get_history_mongo_tool()
    try:
        # 构造查询条件：仅查询指定session_id的记录
        query = {"session_id": session_id}

        # 执行查询：按时间戳升序排序，限制返回条数
        # find(query)：获取符合条件的游标（惰性加载，不立即查询）
        # sort("ts", ASCENDING)：按ts字段升序（从旧到新），适配LLM上下文顺序
        # limit(limit)：限制返回的最大条数
        cursor = mongo_tool.chat_message.find(query).sort("ts", DESCENDING).limit(limit)
        # 将游标转为列表，触发实际数据库查询，获取所有符合条件的文档
        messages = list(cursor)
        messages.reverse()
        # 返回查询结果列表
        return messages
    except Exception as e:
        # 捕获查询异常，记录错误日志
        logging.error(f"Error getting recent messages: {e}")
        # 异常时返回空列表，避免上层处理None报错
        return []

def clear_history(session_id: str) -> int:
    """
    清空指定会话的所有历史对话记录
    :param session_id: 会话唯一标识，用于筛选要删除的记录
    :return: 实际删除的文档数量，删除失败返回0
    """
    # 获取全局的HistoryMongoTool实例，使用单例模式避免重复创建数据库连接
    mongo_tool = get_history_mongo_tool()
    try:
        # 执行批量删除操作：删除所有session_id匹配的文档
        result = mongo_tool.chat_message.delete_many({"session_id": session_id})
        # 记录删除成功日志，包含删除数量和会话ID，便于问题排查
        logging.info(f"Deleted {result.deleted_count} messages for session {session_id}")
        # 返回实际删除的数量（delete_many的返回对象包含deleted_count属性）
        return result.deleted_count
    except Exception as e:
        # 捕获删除异常，记录错误日志，包含会话ID
        logging.error(f"Error clearing history for session {session_id}: {e}")
        # 异常时返回0，标识删除失败
        return 0



if __name__ == "__main__":
    # # 测试代码
    # save_chat_message(
    #     "test_session_id_001", "user", "请问这个怎么用？", "请问HAK180烫金机如何使用？"
    # )

    # 测试会话，用于确认商品名称是否能正确的提取
    sid = "test_session_002"
    # 1. 写入用户消息
    save_chat_message(sid, "user", "你好，有烫金机吗？")
    # 2. 写入助手回复
    save_chat_message(sid, "assistant", "你好！请问你想询问哪个型号？")
    # 3. 写入带关联商品的用户消息
    save_chat_message(sid, "user", "brother的HAK180烫金机")
    save_chat_message(sid, "assistant", "有的")


'''
## 1. ts 排序的代码改法

把 `get_recent_messages` 改成"**先降序取最新，再反转成正序**"：

```python
def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询指定会话的最近N条对话记录
    先按时间降序取最新N条,再反转为时间正序(从旧到新),便于直接喂给LLM
    """
    mongo_tool = get_history_mongo_tool()
    try:
        query = {"session_id": session_id}

        # ① 降序排序 + limit:拿到的是【最新】的N条
        cursor = (
            mongo_tool.chat_message.find(query)
            .sort("ts", -1)          # -1=降序(从新到旧)
            .limit(limit)
        )
        latest_messages = list(cursor)   # 触发真实查询

        # ② 反转成时间正序(从旧到新),符合LLM阅读对话的顺序
        messages = latest_messages[::-1]
        return messages
    except Exception as e:
        logging.error(f"Error getting recent messages: {e}")
        return []
```

关键就两处：`sort("ts", -1)` 降序截最新，`[::-1]` 反转回正序。ASCENDING 那个常量也不用再 import 了。

---

## 2. 完整面试问答（可直接复制）

**Q1：RAG 问答为什么要存对话历史？不存会怎样？**

A：为了支持多轮对话的上下文理解和指代消解。
用户第二轮常说"那它怎么复位""这个多少钱"，里面的"它""这个"指代上一轮提到的产品，脱离上下文根本无法检索。
我按 session_id 把每轮消息存到 MongoDB，下一轮先取最近几条拼进提示词，LLM 才能把"那它怎么复位"还原成"HAK180 怎么复位"再去检索。
要注意对话历史和知识库是两回事：知识库是 Milvus 里长期、跨用户共享的手册切片，导入线构建；对话历史是 MongoDB 里临时、按会话隔离的聊天记录，聊天不会往知识库里写数据。

---

**Q2：对话历史为什么选 MongoDB，而不是 MySQL 或 Milvus？**

A：因为对话消息是典型的文档型数据：一条消息就是一个 JSON，字段灵活，有列表（item_names、image_urls）、有可选项，结构还可能演进，用文档数据库最自然，和 Python dict 天然互转，不用维护固定表结构。
MySQL 强关系、固定 schema，对这种多变结构偏重；Milvus 是向量库，只在需要相似度检索时用，而历史消息只按会话和时间查询，不需要向量搜索。所以 MongoDB 是最合适的选择。

---

**Q3：为什么给 chat_message 建 (session_id:1, ts:-1) 复合索引？**

A：因为这个集合的核心查询模式永远是"按 session_id 等值筛选出某个会话，再按 ts 时间排序取最近 N 条"，对应 WHERE session_id=? ORDER BY ts DESC LIMIT N。
复合索引第一列 session_id 支持快速等值定位，不用全表扫描；第二列 ts 在索引内已经有序，数据库可以直接按索引顺序读出结果，避免额外的内存排序。
一个索引同时覆盖了过滤和排序两个操作。create_index 自带幂等，重复执行不会报错。

---

**Q4：get_recent_messages 想取"最近 10 条"，代码按 ts 升序再 limit 有什么问题？怎么修？**

A：这是个真实 bug。升序排序后 limit(10) 取的是时间最靠前、也就是最旧的 10 条；当会话有 50 条消息时，永远返回第 1~10 条，最新对话反而取不到，记忆就错乱了。
正确做法是两步：先按 ts 降序排序再 limit，拿到最新的 N 条，然后用切片 [::-1] 反转成时间正序——因为喂给 LLM 的对话顺序必须从旧到新。这样既保证取到的是最近消息，又保证了上下文顺序正确。
这个点也说明写查询时要想清楚 limit 是在有序结果的哪一端截取。

---


'''