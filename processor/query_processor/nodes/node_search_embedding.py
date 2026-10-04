# processor/query_processor/nodes/node_search_embedding.py
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase, T
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.milvus_utils import get_milvus_client, online_hybrid_search
from utils.task_utils import add_done_task


class NodeSearchEmbedding(NodeBase):
    """
    节点功能：基于已确认文献名 + 改写后的用户问题，在 kb_chunks 集合执行在线混合检索。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding"

    def process(self, state: T) -> T:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        try:
            # 1、改写后的问题与已确认文献名
            query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2. 过滤表达式：仅在已确认文献的切片范围内检索
            expr = f'chunk_level=="child" and item_name in {item_names}'

            # 3. 在线混合检索 kb_chunks
            client = get_milvus_client()
            collection_name = milvus_config.chunks_collection
            search_res = online_hybrid_search(
                client=client,
                collection_name=collection_name,
                query_text=query,
                expr=expr,
                limit=10,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name", "parent_uid", "parent_title"]
            )

            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
            return {"embedding_chunks": search_res[0] if search_res else []}

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}


if __name__ == "__main__":

    init_state = {
        "session_id": "n2_test",
        "rewritten_query": "基于深度卷积神经网络的协作频谱感知方法是如何缓解梯度消失的？",
        "item_names": ["基于深度卷积神经网络的协作频谱感知方法"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))
