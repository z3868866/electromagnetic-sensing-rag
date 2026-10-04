# processor/import_processor/nodes/node_bge_embedding.py
import json
import logging
from typing import Dict, List

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import get_online_embedder

# 把 state 里所有 chunks 的文本内容，逐个转成向量（稠密 + 稀疏），然后绑定回每个 chunk 上

class NodeBGEEmbedding(BaseNode):
    """
    向量化节点：使用在线 text-embedding-v4 将文本转换为 dense 向量；
    sparse 向量由 Milvus BM25 Function 在入库时依据 content 自动生成。
    """

    name: str = "node_bge_embedding"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：BGE-M3文本向量化处理
        流程总览：
            1. 输入校验：验证chunks有效性，核心数据缺失则终止当前节点
            2. 批量向量化：分批拼接文本、生成双向量，为切片绑定向量字段
            3. 状态更新：将带向量的chunks更新回全局状态，供下游Milvus入库节点使用

        必要参数：chunks
        更新参数：chunks字段新增dense_vector/sparse_vector

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据校验
        chunks = self._step_1_validate_input(state)

        # 步骤2：批量生成双向量，为切片绑定向量字段
        output_data = self._step_2_generate_embeddings(chunks)

        # 步骤3：更新全局状态，将带向量的chunks回传下游
        state['chunks'] = output_data
        return state

    def _step_1_validate_input(self, state: ImportGraphState) -> List[Dict]:
        """
        步骤 1：输入数据有效性校验
        核心作用：
            1. 从全局状态提取待向量化的chunks切片列表
            2. 严格校验chunks类型和非空性，无有效数据则终止向量化
        参数：
            state: ImportGraphState - 流程全局状态对象
        返回：
            List[Dict] - 校验通过的文本切片列表
        异常：
            若chunks非列表/为空，抛出ValueError，终止当前向量化流程
        """

        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return chunks

    def _step_2_generate_embeddings(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        步骤 2: 批量生成向量（核心业务逻辑）
        核心逻辑：
            1. 分批处理：避免一次性处理过多数据导致显存溢出（OOM）。
            2. 文本构造：将 item_name 和 content 拼接，增强语义（商品名作为核心特征前置）。
            3. 向量生成：调用模型批量生成 Dense（稠密）和 Sparse（稀疏）向量。
        参数：
            chunks: List[Dict] 待向量化的文本切片列表
        返回：
            List[Dict]: 包含向量字段（dense_vector/sparse_vector）的文本切片列表
        """

        # 在线 API 批量大小：分批调用以控制单次请求体大小、便于限流退避（不占用本地显存）
        batch_size = 10
        # 初始化一个空列表
        output_data = []

        # 在线 embedder（text-embedding-v4，单例）
        embedder = get_online_embedder()

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            # 批量构造文本：item_name 作为元数据前置，参与 dense 语义表示
            texts = [f"{chunk.get('item_name', '')} \n{chunk['content']}" for chunk in batch_chunks]

            # 在线批量生成 dense 向量；sparse 不在客户端生成，由 Milvus BM25 Function 入库时自动计算
            dense_vectors = embedder.embed_documents(texts)

            for j in range(len(batch_chunks)):
                chunk = batch_chunks[j]
                chunk["dense_vector"] = dense_vectors[j]
                output_data.append(chunk)

        return output_data



if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_bge_embedding = NodeBGEEmbedding()
    result = node_bge_embedding(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))

    '''
    下面是第 7 课 4 道检查点的**标准面试答案**，按口述标准写，可直接背。

---

## 第 1 题：这个节点碰 Milvus 吗？向量去哪、谁入库？

**标准回答：**

> 不碰。这个节点是纯计算节点，不调任何外部服务，只负责用本地 BGE-M3 给每个 chunk 一次性生成稠密和稀疏双向量，然后把向量绑回 chunk 字典、更新到 state。它不负责持久化——带向量的 chunks 由下一个节点 save_to_milvus 批量写入 Milvus 的 kb_chunks 集合。这样"算向量"和"存数据"职责分离，本节点可以脱离数据库单独测试。

---

## 第 2 题：为什么分批？batch_size 调大调小受什么约束？

**标准回答：**

> 因为文档可能有上万个切片，一次性全部喂给 BGE-M3，GPU 显存要同时容纳模型权重和这一批文本的中间激活结果，会直接 OOM 爆显存。所以用 batch 分批推理，每批只处理几条，算完释放显存再来下一批。
>
> batch_size 是"显存换吞吐"的旋钮：调大，GPU 并行度高、整体吞吐量高，但显存占用大，超过显存上限就 OOM；调小，显存安全但推理次数变多、总耗时长。实际取值要根据显卡显存和文本长度压测，比如 8G 显存跑 BGE-M3 可以设 16 到 32，工程上不会硬编码，而是做成配置项。

---

## 第 3 题：为什么向量化文本是 item_name + content，而不是只用 content？

**标准回答：**

> 这是把文档级元数据注入文本的设计。切片正文里不一定每个 chunk 都出现产品名，比如不同产品手册里都有"长按复位键 10 秒"这种几乎相同的操作段落，如果只编码正文，它们的向量会非常接近，检索时容易跨产品串号。
>
> 把产品名拼在正文最前面一起编码，等于强制每个 chunk 的语义向量都带上"产品身份"这个强特征：一方面，内容相同但产品不同的切片，向量距离被拉开；另一方面，用户带产品名提问时，产品名和正文语义双重命中，召回更准。这比只把产品名存成标量过滤字段更进一步，让它直接参与了语义表示。

---

## 第 4 题：为什么主体识别节点 Milvus 挂了降级继续，这个节点模型挂了却直接崩？

**标准回答：**

> 这是关键路径和非关键路径的区别，不是语法习惯。
>
> 主体识别节点少存一条产品名，属于非关键路径故障——切片照常可以入库，系统还能部分运转，所以用 try-except 降级、打 warning 继续，保住主流程。
>
> 而向量化节点如果失败，chunk 算不出向量，硬着头皮入库的全是没有向量的废数据，检索时根本搜不到；更糟的是问题会被藏到检索阶段才暴露，很难排查。所以它属于关键路径，要 fail-fast 快速失败，当场抛异常，绝不让脏数据流到下游。
>
> 我的原则是：非关键路径出故障要容错降级，关键路径出故障要立刻暴露——判断依据是"这个节点的数据废了，整条链还有没有意义"。

---

四题里**第 3 题（元数据注入）和第 4 题（fail-fast 取舍）**最能体现设计能力，建议重点背。准备好了把 **save_to_milvus.py** 贴过来，给导入线收尾。
    '''