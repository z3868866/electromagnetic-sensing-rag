from typing import Dict, List
from .sse_utils import push_to_session

# ---------------------------
# 内存态任务追踪（单进程）
# 这段代码是任务进度追踪系统——记录每个任务当前跑到哪个节点了、哪些节点已经完成、任务状态是什么，并且通过 SSE 把进度实时推送给前端显示。
# ---------------------------
# key: task_id
# value: 节点名列表（原始英文/节点ID）
_tasks_running_list: Dict[str, List[str]] = {}#每个任务正在运行的节点列表
_tasks_done_list: Dict[str, List[str]] = {}#每个任务已完成的节点列表

# key: task_id
# value: status 字符串（如 pending/processing/completed/failed）
_tasks_status: Dict[str, str] = {}#每个任务的状态

# key: task_id
# value: 任务结果（例如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}#每个任务的结果


TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（用于前端展示）
# 说明：这里的 key 应与 LangGraph 的 add_node("xxx", ...) 中的节点名一致。
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",  
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_item_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点（kb/query_process/main_graph.py）---
    "node_item_name_confirm": "确认问题文献",
    "node_answer_output": "生成答案",
    "node_rerank": "精排序",
    "node_rrf": "RRF融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "文献切片检索",
    "node_search_embedding_hyde": "假设文档检索",
    "node_multi_search": "多路检索",
    "node_join": "多路检索合并",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    if task_id not in _tasks_running_list:
        _tasks_running_list[task_id] = []
    if task_id not in _tasks_done_list:
        _tasks_done_list[task_id] = []
    if task_id not in _tasks_result:
        _tasks_result[task_id] = {}


def _to_cn(node_name: str) -> str:
    """将节点名转换为中文展示名；若无映射则返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream:bool = False) -> None:#标记某节点正在运行
    """
    添加“正在运行”的节点任务。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    # 避免重复追加
    if node_name not in running:
        running.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream:bool = False) -> None:#标记某节点已完成
    """
    添加“已完成”的节点任务。

    注意：添加已完成任务时，会把同名的“正在运行”任务删除。

    参数：
    - task_id: 任务ID
    - node_name: 节点名称(节点ID)
    """
    _ensure_task(task_id)

    # 1) 从 running 中移除同名节点（可能出现重复，移除所有）
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]

    # 2) 追加到 done（保持完成顺序），避免重复
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)

    if is_stream:
        task_push_queue(task_id)



def set_task_result(task_id: str, key: str, value: str) -> None:#存储任务结果字段（如 answer / error）
    """
    存储任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:#获取任务结果字段（如 answer / error）
    """
    获取任务结果字段（如 answer / error）。
    """
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str ) -> str:#获取当前任务状态
    """
    获取当前任务状态。

    参数：
    - task_id: 任务ID

    返回：
    - str: 状态名称；如果未设置过则返回空字符串
    """
    return _tasks_status.get(task_id, "")



def get_done_task_list(task_id: str) -> List[str]:#获取已完成节点列表（中文展示）
    """
    获取已完成节点列表（中文展示）。

 
    """
    _ensure_task(task_id)
    done = _tasks_done_list.get(task_id, [])
    return [ _to_cn(n)  for n in done]


def get_running_task_list(task_id: str) -> List[str]:#获取正在运行节点列表（中文展示）
    """
    获取正在运行节点列表（中文展示）。
 
    """
    _ensure_task(task_id)
    running = _tasks_running_list.get(task_id, [])
    return [ _to_cn(n)  for n in running]



def update_task_status(task_id: str, status_name: str, push_queue:bool = False) -> None:#更新任务状态
    """
    更新任务状态。

    参数：
    - task_id: 任务ID
    - status_name: 状态名称（字符串）
    """
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)

def task_push_queue(task_id: str):#通过 SSE 推送任务进度给前端
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })
#
def clear_task(task_id: str):#清空任务数据
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)

    '''

**Q1：task_utils 和 sse_utils 各负责什么？它们怎么配合实现进度条？**

A：sse_utils 是"推送管道"，负责 SSE 连接、按 session_id 隔离的队列和把消息发给浏览器，它不关心任务进行到哪一步；
task_utils 是"进度数据源"，按 task_id 在内存维护四张表：正在运行的节点列表、已完成的节点列表、任务状态和任务结果。
LangGraph 节点生命周期回调在节点开始时登记 running、结束时移到 done，每次状态变化就调用 task_push_queue，把状态加 running/done 两个中文名列表组装成 progress 事件，
交给 sse_utils 的 push_to_session 推给前端，前端据此画出"哪步在转圈、哪步打勾"的步骤条。两者是数据源和传输管道的关系。

---

**Q2：一个节点从开始到结束，running 和 done 两个列表怎么变化？**

A：节点开始时调用 add_running_task，把节点名加入 running 列表（带去重，避免回调重复触发）；
节点结束时调用 add_done_task，把该节点从 running 列表移除，再追加到 done 列表。done 列表严格按完成先后顺序保存，这样前端步骤条能按真实执行顺序逐个打勾。
正常情况下 running 此消、done 彼长，直到所有节点完成，running 清空、done 包含全部节点。

---

**Q3：为什么要做节点英文名到中文名的映射？查不到映射时为什么返回原名而不是报错或空？**

A：代码里节点必须用英文 ID，和 LangGraph 的 add_node 注册名保持一致；但给终端用户看的是中文步骤名，所以用一张映射表做展示层解耦，代码逻辑不受文案影响。
查不到时用 dict.get 返回原英文名，是防御性设计：将来新增节点如果忘了配置中文映射，线上表现是"至少显示英文节点名"，而不是返回 None 导致前端空白或报错。对未知输入永远给一个安全兜底值，保证系统健壮性。

---

**Q4：进度状态存在内存全局字典里，多进程部署或服务重启时会有什么问题？**

A：有两个问题。第一，多 worker 进程部署时，各进程的内存不共享：
任务在 A 进程登记的进度，如果 SSE 连接被分到 B 进程，B 进程的全局字典里没有这个 task_id，前端会看到进度时有时无甚至一直为空，这是分布式状态不一致问题，不是内存占用问题。
第二，服务重启后所有进行中的任务进度全部丢失。生产环境的解法是把进度状态放到 Redis 这类所有进程共享、且可持久化的外部存储，而不是放在进程内存里。另外内存态还要配合任务结束清理和 TTL 过期，防止僵尸任务长期占用内存。

---


    '''