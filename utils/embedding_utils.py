import os

import numpy as np
from dotenv import load_dotenv


_bge_m3_ef = None
_online_embedder = None


def get_online_embedder():
    """
    获取全局单例的在线稠密向量客户端（阿里云百炼 text-embedding-v4）。
    用于替代本地 BGEM3 的稠密向量：低配置机器不下载本地模型，统一走在线 API。
    """
    global _online_embedder
    if _online_embedder is not None:
        return _online_embedder

    load_dotenv(r"D:\RAG\YUANLAI\.env")
    from langchain_openai import OpenAIEmbeddings

    _online_embedder = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        check_embedding_ctx_length=False
    )
    return _online_embedder

def get_bge_m3_ef():
    """
    获取全局单例的BGEM3EmbeddingFunction对象
    :return:  BGEM3EmbeddingFunction对象
    """
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    # 惰性导入：仅在使用本地 BGEM3 时才加载（依赖 transformers，在线方案不触发）
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
    from config.embedding_config import embedding_config

    model_name = embedding_config.bge_m3_path
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    _bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,
        use_fp16=use_fp16
    )

    return _bge_m3_ef


def generate_embeddings(texts):
    """
    为文本生成向量嵌入
    :param texts: 要生成嵌入的文本列表
    :return: 包含dense和sparse向量的字典
    """

    # BGE - M3返回的稀疏向量是CSR格式（CompressedSparseRow），这是一种紧凑的存储方式

    model = get_bge_m3_ef()
    embeddings = model.encode_documents(texts)

    processed_sparse = []
    for i in range(len(texts)):

        sparse_obj = embeddings["sparse"]
        sparse_indices = sparse_obj.indices[
            sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]].tolist()

        sparse_data = sparse_obj.data[
            sparse_obj.indptr[i]:sparse_obj.indptr[i + 1]].tolist()

        sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
        processed_sparse.append(sparse_dict)

    return {
        "dense": [emb.tolist() for emb in embeddings["dense"]],
        "sparse": processed_sparse
    }


def normalize_sparse_vector(sparse_vec):
    """
    对稀疏向量做 L2 归一化（仅处理非零维度，不影响零维度）
    :param sparse_vec: 原始稀疏向量（dict 格式：{维度: 数值}）
    :return: 归一化后的稀疏向量
    """
    if not sparse_vec:  # 空向量直接返回
        return sparse_vec

    # 提取非零维度的数值
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # 计算 L2 范数（避免除以 0）
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # 范数接近 0 时，直接返回原向量（避免除零错误）
        return sparse_vec

    # 归一化：每个数值除以 L2 范数
    normalized_values = values / l2_norm
    # normalized_values = (values / l2_norm).astype(np.float32)  # 统一转为 float32
    # 重建稀疏向量 dict
    return dict(zip(sparse_vec.keys(), normalized_values))

'''
下面是第 10 课（BGE-M3 向量工具）的完整面试问答，可直接复制。

---

**Q1：BGE-M3 模型为什么要用单例加载？不缓存会怎样？**

A：BGE-M3 模型体积约 2GB，从磁盘加载权重到显存需要几十秒，属于很重的初始化对象。而一个服务进程要反复调用它做成千上万次向量化，如果每次调用都重新加载，时间和显存都无法接受。
我用模块级全局变量加判空的方式实现单例：第一次调用时加载模型并缓存，之后所有请求复用同一个实例。
这和项目里 Milvus 客户端、MinIO 客户端、LangGraph 编译图的懒加载是同一个设计思想——重对象只初始化一次。

---

**Q2：BGE-M3 输出的稀疏向量是 CSR 格式，它的 indptr、indices、data 三个数组各存什么？怎么取出第 i 条文本？**

A：CSR 是压缩稀疏行格式，用三个一维数组紧凑存储所有非零元素：data 存所有非零的权重值，indices 存每个非零值对应的词 id（列索引），indptr 是行指针，标记每条文本在 data/indices 里的起止位置。
取第 i 条文本就是切片 data[indptr[i] : indptr[i+1]] 和 indices[indptr[i] : indptr[i+1]]，即"从我的起点到下一条的起点"，再把两个切片 zip 成字典。

---

**Q3：为什么要把 CSR 转成 {词id:权重} 字典？是为了省空间吗？**

A：不是为了省空间——CSR 本身就只存非零元素，字典也只存非零元素，两者空间开销相当。真正原因是格式契约：CSR 是 SciPy 模型内部的计算格式，方便矩阵运算；
而 Milvus 的 SPARSE_FLOAT_VECTOR 字段要求的插入格式是 {整数词id: 浮点权重} 字典，插入时它只认这种结构。所以这一步是"模型输出格式到数据库存储格式"的适配转换，是两个系统之间的接口对接。

---

**Q4：稀疏向量为什么要做 L2 归一化？数学上做了什么？不归一会有什么问题？**

A：L2 归一化就是把向量每个分量除以它的 L2 范数（各分量平方和开根号），使向量长度变为 1，比如 {0:3, 4:4} 的范数是 5，归一化后变成 {0:0.6, 4:0.8}。
稀疏向量索引用的是内积 IP，内积同时受方向和长度影响，长文档命中的词多、向量更长，会天然刷高内积分，造成排序不公平。
归一化后所有向量长度都是 1，内积就等价于余弦相似度，只比语义方向、不比词多词少。实现上只处理字典里的非零维度（零分量归一化后还是零），并对范数接近零的空向量做防除零保护。




'''