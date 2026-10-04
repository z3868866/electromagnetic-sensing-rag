# processor/query_processor/state.py

from typing import TypedDict, List

'''
QueryGraphState 是查询流程的"共享白板"：定义了从用户提问到最终答案的整个过程中，所有节点需要读写的数据字段。
每个节点从白板上读自己需要的数据，处理完后把结果写回白板，下一个节点再读取。数据像流水线一样一步步往下传。
'''

class QueryGraphState(TypedDict):
    """
    查询流程图状态
    包含整个查询流程中传递的所有数据。
    """

    session_id: str  # 会话ID
    message_id: str  # 消息ID

    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # 已向量化的假设性问题切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的文献名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出

    # 链路追踪
    trace_spans: list  # 各节点 span（节点名/起止/耗时/状态），累积后供链路追踪落库

'''

**Q1：检索线为什么把三路检索结果分成三个字段存，而不是边检索边合并？**

A：因为后面的 RRF 融合算法依赖"每一路的独立排名"。RRF 要计算每条结果在普通向量检索路排第几、在 HyDE 路排第几、在网络搜索路排第几，再按名次倒数加权融合。
如果检索时就把三路结果混成一个列表，"这条结果来自哪一路、在那一路里排第几名"的信息就丢失了，RRF 无法计算。
所以用 embedding_chunks、hyde_embedding_chunks、web_search_docs 三个字段分别保留来源和各路排名，到融合节点再统一处理，同时也避免不同来源的数据互相污染。

---

**Q2：original_query 和 rewritten_query 有什么区别？为什么必须做问题改写？**

A：original_query 是用户的原始提问，原样保留用于前端展示和存入对话历史；rewritten_query 是结合多轮历史改写后的完整、无指代的问题，用于实际检索和喂给大模型。
必须改写是因为多轮对话里用户常说"那它怎么复位""这个多少钱"，里面的"它""这个"指代上一轮提到的产品，脱离上下文直接拿去向量检索根本命中不了。
系统先从 MongoDB 取出该会话历史，让 LLM 把指代词还原成具体实体，比如把"那它怎么复位"改写成"HAK180 烫金机怎么复位"，再用这个完整问题去检索，召回才准确。

---

**Q3：检索线 7 个节点的先后顺序是什么？**

A：整体是"分诊 → 三路搜 → 融合 → 精排 → 作答"：第一个是产品确认节点，识别问题涉及哪款产品并完成问题改写；之后三路并行检索——普通混合向量检索、HyDE 假设文档检索、网络搜索，结果分别存入三个独立字段；三路汇合后由 RRF 节点按排名倒数融合，再交给 Rerank 节点用交叉编码器做精排、取 Top-K；最后答案输出节点组装 Prompt，基于精排后的切片生成带出处引用的最终答案。在 LangGraph 里三路检索是并行分支，在融合节点汇合。

---

**Q4：history 从哪来？session_id 在检索线有哪些作用？**

A：history 是按 session_id 从 MongoDB 对话历史集合里取出的最近 N 条消息，按时间正序排列，用于多轮指代消解和给 LLM 提供上下文。
session_id 在检索线有三个作用：第一，作为查询条件从 MongoDB 取该会话的历史消息；
第二，作为 SSE 队列的 key，把节点进度和流式 token 精准推送给对应用户的浏览器；
第三，配合 message_id 把这一轮问答和识别出的产品名存回 MongoDB。
它是贯穿"历史记忆、进度推送、消息持久化"三条线的会话标识。

---


'''