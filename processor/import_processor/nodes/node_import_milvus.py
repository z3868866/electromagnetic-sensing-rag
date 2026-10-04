# processor/import_processor/nodes/node_import_milvus.py
import json
import logging
from typing import Dict, Any, List

from pymilvus import DataType, Function, FunctionType

from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.milvus_utils import get_milvus_client, escape_milvus_string


# 把上游传来的、已经带好了向量的 chunks，真正写入 Milvus 向量数据库，实现数据持久化。

class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
            2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
            3. 幂等清理：删除同file_title旧数据，避免重复存储
            4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
            5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用

        异常处理：
            任一步骤失败抛出异常，终止节点执行，保证数据不脏写

        必要参数：chunks
        更新参数：chunks字段回填chunk_id

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = self._step_1_check_input(state)

        # 步骤2：Milvus客户端连接+集合准备（自动建表）
        client = self._step_2_prepare_collection(vector_dimension)

        # 步骤3：幂等性处理 - 清理同file_title旧数据
        self._step_3_clean_old_data(client, chunks_json_data)

        # 步骤4：批量插入数据+主键chunk_id回填
        updated_chunks = self._step_4_insert_data(client, chunks_json_data, state)

        # 步骤5：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks

        return state

    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 提取向量维度，为集合创建/索引构建提供依据
        参数：
            state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：
            tuple - (校验通过的切片列表, 稠密向量维度)
        异常：
            任一校验项不通过，抛出ValueError终止入库流程，避免脏数据处理

        """

        # 校验1：chunks非空
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 说明：sparse_vector 不由客户端提供，入库时由 BM25 Function 依据 content 自动生成

        # 提取向量维度
        vector_dimension = len(first_chunk['dense_vector'])
        return chunks, vector_dimension

    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        步骤2：Milvus客户端连接+集合准备
        核心逻辑：
            1. 获取Milvus单例客户端，验证连接有效性
            2. 集合不存在则自动创建（Schema+索引），存在则直接复用
        参数：
            vector_dimension: int - 稠密向量维度（步骤1提取）
        返回：
            MilvusClient - 已连接、集合准备完成的客户端实例
        异常：
            客户端获取失败/集合名称未配置，抛出异常终止流程
        """

        # 1. 获取milvus客户端对象
        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        # 2. 集合不存在则创建
        collections_name = milvus_config.chunks_collection
        if not milvus_client.has_collection(collections_name):
            self._create_chunks_collection(collections_name, milvus_client, vector_dimension)

        return milvus_client

    def _create_chunks_collection(self, collections_name, milvus_client, vector_dimension):

        # 1. 创建 schema
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        # 2. 字段（content 启用中文分词分析器，供 BM25 使用）
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535,
                         enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="chunk_level", datatype=DataType.VARCHAR, max_length=32)
        schema.add_field(field_name="parent_uid", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)

        # 3. 注册 BM25 Function：content → sparse_vector
        schema.add_function(Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_vector"]
        ))

        # 4. 索引：dense 用 COSINE；BM25 Function 输出字段必须用 BM25 度量
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        # 5. 创建集合
        milvus_client.create_collection(
            collection_name=collections_name,
            schema=schema,
            index_params=index_params
        )

    def _step_3_clean_old_data(self, client, chunks_json_data):

        """
        幂等清理
        基于每个片段的file_title进行旧数据的清理
        :param client: milvus客户端
        :param chunks_json_data: chunks数据
        :return:
        """

        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 执行幂等清理
        self._clear_chunks_by_file_title(client, file_title)

    def _clear_chunks_by_file_title(self, client, file_title):

        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=milvus_config.chunks_collection,
                filter=f"file_title=='{file_title}'")
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败: {str(e)}")
            raise MilvusError(f"Milvus 数据删除失败: {str(e)}")

    def _step_4_insert_data(self, client, chunks_json_data, state):
        """
        步骤4：父子分别入库 + 子块主键回填
            父块：完整章节，dense_vector 以零向量占位，不参与检索；
            子块：带真实 dense_vector，chunk_level=child，用于检索；
            sparse_vector 均由 BM25 Function 依据 content 自动生成。
        """
        item_name = state.get("item_name", "")
        parents = state.get("parents", [])
        dim = len(chunks_json_data[0]["dense_vector"])

        # 父块行（零向量占位；空章节不入库）
        parent_rows = []
        for p in parents:
            if not p.get("content"):
                continue
            parent_rows.append({
                "content": p.get("content", ""),
                "title": p.get("parent_title", ""),
                "parent_title": p.get("parent_title", ""),
                "part": 0,
                "file_title": p.get("file_title", ""),
                "item_name": item_name,
                "chunk_level": "parent",
                "parent_uid": p.get("parent_uid", ""),
                "dense_vector": [0.0] * dim,
            })

        # 子块行（带真实向量）
        child_rows = []
        for it in chunks_json_data:
            child_rows.append({
                "content": it.get("content", ""),
                "title": it.get("title", ""),
                "parent_title": it.get("parent_title", ""),
                "part": it.get("part", 0),
                "file_title": it.get("file_title", ""),
                "item_name": it.get("item_name", "") or item_name,
                "chunk_level": "child",
                "parent_uid": it.get("parent_uid", ""),
                "dense_vector": it["dense_vector"],
            })

        if parent_rows:
            client.insert(collection_name=milvus_config.chunks_collection, data=parent_rows)
        result = client.insert(
            collection_name=milvus_config.chunks_collection,
            data=child_rows
        )

        # 回填子块 chunk_id（auto_id，插入前未知）
        inserted_ids = result.get("ids")
        for idx, it in enumerate(chunks_json_data):
            it["chunk_id"] = inserted_ids[idx]

        return chunks_json_data

if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\基于深度卷积神经网络的协作频谱感知方法\state_vector.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))


'''
下面是第 8 课 4 道面试题的完整问答，可直接复制。

---

**Q1：向量库为什么不只存向量，还要存 content、title、item_name 这些标量字段？各有什么用？**

A：向量只负责"相似度找候选"，但一个可落地的 RAG 系统还需要标量字段承担三个职责：第一是元数据过滤，比如用 item_name、file_title 在检索时加 WHERE 条件，把搜索范围限定在指定产品内，避免多产品手册串号；第二是取正文，向量本身不是答案，命中后必须把 content 取出来塞进提示词喂给 LLM 生成回答；第三是答案溯源，用 title、parent_title、chunk_id 标识这条内容出自哪本手册哪一章，支持引用展示和可信度背书。一句话总结：向量负责"找得准"，标量负责"滤得对、取得出、说清出处"。

---

**Q2：稠密向量维度为什么从第一个 chunk 动态取，而不是写死 1024？**

A：因为不同 Embedding 模型输出维度不同，比如 BGE-M3 是 1024 维，有的模型是 768 或 1536 维。建集合时 dim 必须和实际向量一致，否则 Milvus 直接拒收。从第一个 chunk 的向量长度动态推断 dim，就能让 schema 自动适配数据，将来更换 Embedding 模型时这段建表代码不用改，符合"schema 适配数据"的设计原则。

---

**Q3：insert 之后为什么要把返回的 ids 回填到每个 chunk？**

A：因为 chunk_id 是 Milvus 的 auto_id 自增主键，插入前应用层不知道 id 是多少，它是数据库生成的。insert 返回的 ids 按插入顺序和 chunks 一一对应，把它写回每个 chunk 字典后，state 里的切片就有了数据库"身份证"。后续做答案溯源引用、按条删除或更新某条切片，都需要这个主键来精确定位。

---

**Q4：（拔高）"先删后插"如果在删完、还没插完时程序崩了会怎样？怎么改进？**

A：会出现数据丢失：旧数据已经删掉、新数据没插完，这本手册在库里就残缺甚至全空了，而 Milvus 没有跨请求事务，删和插无法原子回滚。改进方向是改成"先插后删 + 版本隔离"：给每批数据加一个 batch_id（导入批次号）字段，先插入带新批次号的数据并确认成功，再删除 file_title 相同但 batch_id 不等于当前批次的旧数据。这样删除的永远是上一版，即使中途崩溃，旧版本仍然完整可用，实现了安全的灰度替换。

---

导入线四课的标准问答到此齐了。需要的话我可以把**导入线全部 8 课的面试题汇总成一份文档**（Markdown，方便你打印背诵），或者直接开始检索线第 11 课骨架代码，你说。
'''
