# processor/query_processor/nodes/node_search_embedding_hyde.py
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.search_embedding_hyde import HYDE_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client, online_hybrid_search, build_item_name_expr
from utils.task_utils import add_done_task


class NodeSearchEmbeddingHyde(NodeBase):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        HyDE 检索节点。
        执行步骤：
        1. 提取改写后的查询（rewritten_query）和已确认文献名（item_names）。
        2. 调用 LLM 生成假设性理想回答（HyDE 文档）。
        3. 将“重写问题 + 假设文档”作为检索文本，在 kb_chunks 中执行在线混合检索（带文献名过滤）。
        4. 返回检索切片与假设文档。
        """
        try:
            # 1、改写后的问题与已确认文献名
            rewritten_query = state.get("rewritten_query")
            item_names = state.get("item_names")

            # 2. 生成假设性文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            # 3、用“重写问题 + 假设文档”检索切片
            res = self._step_2_search_embedding_hyde(
                rewritten_query=rewritten_query,
                hyde_doc=hyde_doc,
                item_names=item_names
            )

            # 4. 封装返回结果
            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
            return {
                "hyde_embedding_chunks": res,
                "hyde_doc": hyde_doc
            }

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            return {}

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """
        阶段1：让 LLM 根据查询生成假设性文档（Hypothetical Document），
        缓解短查询与长文档在语义空间不匹配的问题。
        """
        try:
            llm = get_llm_client()
            gyde_prompt = HYDE_PROMPT.format(rewritten_query=rewritten_query)
            hyde_doc = llm.invoke(gyde_prompt).content
            return hyde_doc
        except Exception as e:
            logger.exception(f"LLM调用失败: {e}")
            raise e

    def _step_2_search_embedding_hyde(
            self,
            rewritten_query: str,
            hyde_doc: str,
            item_names=None
    ):
        """
        阶段2：用“重写问题 + 假设性文档”作为检索文本，在 kb_chunks 中在线混合检索。

        :param rewritten_query: 改写后的查询
        :param hyde_doc: Step 1 生成的假设性文档
        :param item_names: 文献名称列表，用于元数据过滤 (item_name in [...])
        :return: 检索结果列表
        """
        try:
            # 1. 拼接字符串
            combined_text = rewritten_query + " " + hyde_doc

            # 2. 过滤表达式：仅在已确认文献的切片范围内检索（名称统一转义，防注入）
            expr = build_item_name_expr(item_names)

            # 3. 在线混合检索 kb_chunks
            client = get_milvus_client()
            collection_name = milvus_config.chunks_collection
            search_res = online_hybrid_search(
                client=client,
                collection_name=collection_name,
                query_text=combined_text,
                expr=expr,
                limit=10,
                ranker_weights=(0.8, 0.2),
                output_fields=["chunk_id", "content", "item_name", "parent_uid", "parent_title"]
            )

            return search_res[0] if search_res else []

        except Exception as e:
            logger.exception(f"向量搜索失败: {e}")
            raise e


if __name__ == "__main__":

    init_state = {
        "rewritten_query": "基于深度卷积神经网络的协作频谱感知方法是如何缓解梯度消失的？",
        "item_names": ["基于深度卷积神经网络的协作频谱感知方法"]
    }
    node_search_embedding_hyde = NodeSearchEmbeddingHyde()
    result = node_search_embedding_hyde(init_state)
    logger.info(serialize_json(result, indent=4))
