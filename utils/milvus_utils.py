from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker

from config.milvus_config import milvus_config
from tool.logger import logger

_milvus_client = None


# 获取 Milvus 连接（单例缓存）
# 和之前 BGE-M3、LLM 客户端一模一样的套路——单例缓存。Milvus 连接创建一次就够了，不用每次都重新连接。
def get_milvus_client():

    global _milvus_client
    if _milvus_client is not None:
        return _milvus_client

    _milvus_client = MilvusClient(milvus_config.milvus_url)
    return _milvus_client


def online_hybrid_search(
        client,
        collection_name,
        query_text,
        expr=None,
        limit=10,
        ranker_weights=(0.8, 0.2),
        output_fields=None,
        consistency_level=None):
    """
    在线方案混合检索（替代本地 BGEM3 双向量）：
      - 稠密路：query_text 经在线 text-embedding-v4 生成 dense 向量，按 COSINE 检索；
      - 稀疏路：直接传 query_text 原文，由集合的 BM25 Function 自动转换并按 BM25 检索。
    返回 pymilvus hybrid_search 原始结果（按集合分组的列表）。
    """
    from utils.embedding_utils import get_online_embedder

    embedder = get_online_embedder()
    dense_vector = embedder.embed_query(query_text)

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        expr=expr,
        limit=limit
    )
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field="sparse_vector",
        param={"metric_type": "BM25"},
        expr=expr,
        limit=limit
    )

    ranker = WeightedRanker(ranker_weights[0], ranker_weights[1])

    kwargs = dict(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=ranker,
        limit=limit,
        output_fields=output_fields if output_fields else ["item_name"]
    )
    if consistency_level:
        kwargs["consistency_level"] = consistency_level

    return client.hybrid_search(**kwargs)


# 字符串转义（防注入）
'''
Milvus 的过滤表达式长这样：
expr = 'item_name == "HAK180"'
如果产品名称里本身就包含引号呢？比如产品名叫 HAK"180：

# ❌ 不转义：表达式被截断，Milvus 解析报错
expr = 'item_name == "HAK"180"'

# ✅ 转义后：引号被加了反斜杠，Milvus 知道这是内容的一部分
expr = 'item_name == "HAK\"180"'
打个比方：就像你在网上填表单，名字里有个引号，系统会让你加个反斜杠转义，不然系统会把你的引号当成表单的分隔符，导致格式错乱。
'''

def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


# 构建搜索请求（准备"搜查令"）
def create_hybrid_search_requests(
        dense_vector,
        sparse_vector,
        dense_params=None,
        sparse_params=None,
        expr=None,
        limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量

    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector", # 在 Milvus 表的 dense_vector 字段里搜
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",# 在 Milvus 表的 sparse_vector 字段里搜
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


# 执行混合搜索（真正去查）
# 这个函数把上面两个搜索结果加权融合，得到最终排名。
def hybrid_search(
        client,
        collection_name,
        reqs,
        ranker_weights=(0.5, 0.5),
        norm_score=True,
        limit=5,
        output_fields=None,
        search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.4
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None






'''
**Q1：一次混合搜索实际发了几路检索请求？分别在哪个字段、用什么度量？**

A：两路。查询文本经 BGE-M3 生成稠密和稀疏两个向量后，分别构造成两个 AnnSearchRequest：稠密路在 dense_vector 字段上用 COSINE 余弦相似度检索，负责语义相近；稀疏路在 sparse_vector 字段上用 IP 内积检索，负责关键词精确匹配。两路各自取 topK 候选，最后再交给融合器统一排序。

---

**Q2：WeightedRanker 的 norm_score 为什么必须开启？不开会怎样？**

A：因为两路分数的量纲不同——余弦相似度通常在 0 到 1 之间，而稀疏内积的数值可能是 0 到几十，直接加权求和时，数值大的那一路会碾压另一路，0.5/0.5 的权重就形同虚设。norm_score=True 会先把两路分数各自归一化到 0~1 的同一尺度，再加权求和，这样权重才能真实反映"语义和关键词各占多少"。这个权重本身也是可调优的超参，型号手册类场景可以调高稀疏权重。

---

**Q3：混合搜索里的 expr 参数是干什么的？产品分诊结果怎么影响检索？**

A：expr 是 Milvus 的标量过滤表达式，和向量相似度是两回事：向量检索负责"找内容最像的"，expr 负责"先圈定在哪个范围里找"。产品分诊节点识别出用户问的是哪款产品后，传入类似 item_name=='H3C LA2608企业网关' 的 expr，向量搜索就只在这款产品的切片里进行，其他产品手册即使语义很像也不参与，从根本上避免多产品串号。这就是导入时给每个 chunk 都存 item_name 标量字段的价值。

---

**Q4：WeightedRanker 依赖实际分数，如果两路分数量纲完全不可比，还有什么融合办法？**

A：可以用 RRF（Reciprocal Rank Fusion，倒数排名融合）。它完全不看原始分数，只看每条结果在每一路里的排名名次，按 1/(k+名次) 这样的倒数公式打分再相加。因为不依赖分数绝对值，所以天然规避了不同检索器分数量纲不一致的问题，鲁棒性更强；代价是丢失了分数里的置信度信息。项目里 Milvus 库内融合用 WeightedRanker，多路（向量检索、网络搜索等）外部结果融合则单独实现了 RRF。

---

检索线地基的问答齐了。把 **main_graph.py** 发我，正式开始检索流程第一站。
'''