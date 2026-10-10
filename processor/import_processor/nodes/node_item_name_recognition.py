# processor/import_processor/nodes/node_item_name_recognition.py
import json
import logging
from http.client import responses
from typing import Tuple, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pyexpat.errors import messages
from pymilvus import DataType, Function, FunctionType

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.prompt.item_name_recognition import ITEM_NAME_RECOGNITION_TEMPLATE, \
    ITEM_NAME_RECOGNITION_SYSTEM_PROMPT
from processor.import_processor.state import ImportGraphState
from utils.embedding_utils import get_online_embedder
from utils.alias_utils import extract_aliases, build_match_text
from utils.milvus_utils import get_milvus_client, escape_milvus_string
# 把文档切分成很多个chunk，取5个chunk只是为了知道这个chunk是和什么有关？然后提取出和什么有关之后，把后面的95个chunk一起打上标签

# # --- 配置参数 (Configuration) # --- 配置参数 (Configuration) ---
# # 大模型识别文献名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
# DEFAULT_ITEM_NAME_CHUNK_K = 5
# # 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
# CONTEXT_TOTAL_MAX_CHARS = 2500


class NodeItemNameRecognition(BaseNode):
    """
    主体识别节点：主体识别与标签提取
    """

    name = "node_item_name_recognition"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph 核心节点：文献主体名称识别
        流程总览：
            1. 提取输入
            2. 构建大模型上下文
            3. 调用大模型识别文献名称
            4. 回填文献名称到状态和切片
            5. 生成文献名称的稠密/稀疏向量
            6. 将数据存入Milvus向量数据库

        必要参数：task_id、file_title, chunks
        更新参数：item_name

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：提取并校验输入
        file_title, chunks = self._step_1_get_inputs(state)

        # 步骤2：构建大模型识别的上下文
        context = self._step_2_build_context(chunks)

        # 步骤3：调用大模型识别文献名称
        item_name = self._step_3_call_llm(file_title, context)

        # 步骤4：回填文献名称到状态和切片
        self._step_4_update_chunks(state, chunks, item_name)

        # 步骤4b：抽取别名/缩写，构造“标准名+别名”可检索文本 match_text
        aliases = extract_aliases(item_name, chunks)
        match_text = build_match_text(item_name, aliases)

        # 步骤5：为 match_text 生成在线 dense 向量（sparse 由 BM25 Function 自动生成）
        dense_vector = self._step_5_generate_vector(match_text)

        # 步骤6：将标准名、别名、match_text 与 dense 向量存入 kb_item_names
        self._step_6_save_to_milvus(state, file_title, item_name, dense_vector,
                                    aliases=aliases, match_text=match_text)

        # 打印识别结果
        self.logger.info(f"--- 识别完成: {item_name} ---")

        return state

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, List[Dict]]:
        """
        步骤 1: 接收并校验流程输入
        核心作用：
            1. 从流程状态中提取文件标题、文本切片核心数据
            2. 基础数据类型校验，保证下游流程输入有效性
        依赖的状态数据（上游节点产出）：
            - state["file_title"]: 上游提取的文件标题
            - state["chunks"]: 文本切片列表
        返回值：
            Tuple[str, List[Dict]]: (处理后的文件标题, 校验后的文本切片列表)
        """

        # 1、参数校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        return file_title, chunks

    def _step_2_build_context(self, chunks: List[Dict]) -> str:
        """
        步骤 2: 构造大模型文献名称识别的标准化上下文
        核心作用：
            1. 限制切片数量：仅取前k个切片，避免上下文过长
            2. 限制字符长度：总上下文字符限制，适配大模型输入上限
            3. 格式化内容：带序号的结构化格式，提升大模型识别精度
        参数说明：
            chunks: 文本切片列表
        返回值：
            str: 格式化后的上下文字符串
        """

        parts: List[str] = []
        total_chars = 0
        for idx, chunk in enumerate(chunks[:self.config.item_name_chunk_k], start = 1):

            # 1. 提取前k的切片，避免上下文过长
            chunk_title = chunk.get("title").strip()
            chunk_content = chunk.get("content").strip()

            # 2. 格式化切片
            piece = f"【切片{idx}】\n标题{chunk_title}\n内容：{chunk_content}"
            parts.append(piece)

            # 3. 计算累计的字符数
            total_chars += len(piece)

            # 4. 判断是否需要继续切分
            if total_chars > self.config.item_name_chunk_size:
                self.logger.warning(f"累计字符数{total_chars}已超过限制{self.config.item_name_chunk_size}，停止切分")
                break

        # 5. 使用换行符对切分后的片段进行连接
        context = "\n\n".join(parts).strip()

        # 6. 对返回结果进行二次截断
        final_context = context[:self.config.item_name_chunk_size]

        return final_context

    def _step_3_call_llm(self, file_title: str, context: str) -> str:
        """
        步骤 3: 调用大模型实现文献名称精准识别
        核心逻辑：
            1. 上下文为空 → 直接返回file_title（兜底，无需调用大模型）
            2. 上下文非空 → 加载标准化prompt模板，构建大模型对话消息
            3. 调用大模型后对返回结果做清洗，过滤无效字符
            4. 大模型返回空/调用异常 → 均返回file_title兜底，保证流程不中断
        核心特性：
            - 提示词解耦：通过load_prompt加载本地模板，无需硬编码
            - 格式兼容：兼容不同LLM客户端返回格式，防止属性报错
            - 异常兜底：全异常捕获，大模型服务不可用时不影响主流程
        参数：
            file_title: 处理后的文件标题（异常/空值时的兜底值）
            context: 步骤2构建的结构化切片上下文（大模型识别的核心依据）
        返回值：
            str: 清洗后的文献名称（异常/空值时返回原始file_title）
        """

        # 1. 上下文为空，返回file_title
        if not context:
            return file_title

        # 2. 大模型调用
        try:
            # 2.1 加载提示词模板
            prompt = ITEM_NAME_RECOGNITION_TEMPLATE.format(
                file_title = file_title,
                context = context
            )
            # 2.2 初始化大模型客户端
            llm = ChatOpenAI(
                model=lm_config.llm_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                extra_body={"enable_thinking": False}
            )

            # 2.3 创建消息对象
            messages = [
                SystemMessage(content=ITEM_NAME_RECOGNITION_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]

            # 2.4 模型调用
            response = llm.invoke(messages)
            item_name = response.content

            # 2.5 数据清洗
            item_name = (item_name.replace(" ", "")
                         .replace("\n", "")
                         .replace("\t", "")
                         .replace("\r", ""))

            # 2.6 空值或占位符（LLM 无主体时可能按 prompt 返回字面“空字符串”）→ 兜底为 file_title
            if not item_name or item_name.strip() in ("空字符串", "无", "none", "None", "null"):
                return file_title

            return item_name

        except Exception as e:
            self.logger.error(f"大模型调用异常：{e}")
            return file_title

    def _step_4_update_chunks(self, state: ImportGraphState, chunks: List[Dict[str, str]], item_name: str):
        """
        步骤 4: 回填文献名称到流程状态和所有文本切片
        核心作用：
            1. 全局状态更新：将item_name存入state，供下游所有节点直接使用
            2. 切片数据补全：为每个切片添加item_name字段，保证数据一致性
            3. 状态同步：更新state中的chunks，确保切片修改全局生效
        设计思路：
            所有切片关联同一文献名称，保证后续向量入库、检索时的维度一致性
        参数：
            state: 流程状态对象
            chunks: 校验后的文本切片列表
            item_name: 步骤3识别并清洗后的文献名称
        """

        state["item_name"] = item_name

        for chunk in chunks:
            chunk["item_name"] = item_name

        state["chunks"] = chunks

    def _step_5_generate_vector(self, text: str):
        """
        步骤 5：使用在线 text-embedding-v4 为文献名生成 dense 向量。
        sparse 向量不在客户端生成，由 kb_item_names 的 BM25 Function
        依据 item_name 字段在入库时自动计算。
        :param item_name: 步骤 3 识别的文献名
        :return: dense 向量（list[float]）；item_name 为空时返回 None
        """
        if not text:
            return None
        return get_online_embedder().embed_query(text)

    def _step_6_save_to_milvus(self, state: ImportGraphState, file_title: str, item_name: str, dense_vector, aliases=None, match_text=""):
        """
        步骤 6: 将文献名称、文件标题、双向量持久化到 Milvus 向量数据库
        核心逻辑：
            1. 客户端获取：获取单例 Milvus 客户端，连接失败则跳过
            2. 集合初始化：无集合则创建（定义 Schema+索引），有集合则直接使用
            3. 幂等性处理：删除同名文献数据，避免重复存储
            4. 数据插入：构造符合 Schema 的数据，非空向量才添加
        参数：
            state: 流程状态对象，用于最终状态同步
            file_title: 处理后的文件标题
            item_name: 识别后的文献名称（主键去重依据）
            dense_vector: 步骤 5 生成的稠密向量（1024 维列表）
            sparse_vector: 步骤 5 生成的稀疏向量（字典格式）
        """


        # 1. 获取 Milvus 客户端
        milvus_client = get_milvus_client()
        if not milvus_client:
            # 降级处理
            self.logger.warning("Milvus 连接失败，跳过设备名称保存")
            return
        try:
            # 2. 集合初始化
            collection_name = milvus_config.item_name_collection

            # 判断集合是否已经被创建
            if not milvus_client.has_collection(collection_name):
                self._create_item_name_collection(collection_name, milvus_client)

            # 3. 幂等性处理：删除同名文献数据，避免重复存储
            file_title = escape_milvus_string(file_title)#对文件名进行转义。如果文件名是 O'Reilly.pdf，里面的单引号会破坏 filter 查询语句，所以必须先处理一下。
            milvus_client.delete(collection_name=collection_name, filter=f"file_title=='{file_title}'")

            # 4. 数据插入：只传 item_name 与 dense_vector；
            #    sparse_vector 由 BM25 Function 自动生成，禁止客户端传入
            data = {
                "file_title": file_title,
                "item_name": item_name,
                "aliases": " | ".join(aliases or []),
                "match_text": match_text or item_name,
            }
            if dense_vector is not None:
                data["dense_vector"] = dense_vector

            milvus_client.insert(collection_name=collection_name, data=data)


        except Exception as e:
            # 降级处理
            self.logger.warning(f"数据存入Milvus失败，跳过设备名称保存，错误原因:{str(e)}")

    def _create_item_name_collection(self, collection_name, milvus_client):

        # 1. 创建 schema
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

        # 2. 字段：item_name 启用中文分词分析器，供 BM25 使用
        schema.add_field(
            field_name="name_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )
        schema.add_field(
            field_name="item_name",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"}
        )
        schema.add_field(field_name="aliases", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(
            field_name="match_text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"}
        )
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

        # 3. 注册 BM25 Function：match_text（标准名+别名）→ sparse_vector
        schema.add_function(Function(
            name="bm25_function",
            function_type=FunctionType.BM25,
            input_field_names=["match_text"],
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
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25"
        )

        # 5. 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )


if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\基于深度卷积神经网络的协作频谱感知方法\chunks.json"
    with open(json_path, "r", encoding="utf-8") as f:
        chuncks_json = f.read()

    chuncks = json.loads(chuncks_json)

    init_state = {
        "chunks": chuncks,
        "file_title": "基于深度卷积神经网络的协作频谱感知方法"
    }

    # 执行核心处理流程
    node_item_name_recognition = NodeItemNameRecognition()
    result = node_item_name_recognition(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))

    '''
        稠密向量 dense	                    稀疏向量 sparse
长相	    1024 个浮点数，密密麻麻几乎都非 0	    25 万维词表里只点亮几个：{词id: 权重}
本质	    一句话的 "意思指纹"	                一句话的 "关键词账本"
擅长	    懂同义："CNN"≈"卷积神经网络"	        精确命中：缩写 "DCNN" 必须字面出现
短板	    可能 "脑补"、跑偏	                    不懂同义词、错一个字就不认
    '''

    '''
    下面是这 4 题的**标准回答**，按面试口述的标准写，因果链完整，你可以直接背、也能写进简历项目描述。

---

## 第 1 题：文献名为什么单独建集合？服务检索线什么功能？

**标准回答：**

> 文献名单独建集合，是因为它和切片回答的是两个不同的问题：文献名册回答"知识库里有哪些文献"，切片库回答"具体内容在哪"。
>
> 它服务于检索线的**文献名确认节点**，做检索前的"分诊路由"：
用户提问后，先把问题向量只在几百条文献名里做匹配——匹配到唯一文献，就用文献名作为元数据过滤条件，把后续切片检索锁定在该文献范围内，避免不同文献互相干扰；
匹配到多个或模糊时，反问用户"您问的是哪一篇"；一个都匹配不上，说明知识库没有该文献，直接拒答或引导。
> 如果从切片库查，几万个 chunk 每个都重复带文献名，既慢又要去重聚合，而且拿问题去匹配章节内容，命中的是"系统模型""实验结果"这类章节，根本回答不了"有哪些文献"。
独立小集合又快又省，业务语义也清晰。

---

## 第 2 题：稠密/稀疏向量的本质，为什么混合用？

**标准回答：**

> 稠密向量是 BGE-M3 输出的 1024 维浮点向量，几乎每一维都非零，本质是文本的"语义指纹"，刻画深层含义，擅长同义、近义匹配——"卷积神经网络"和"CNN"字面不同，但稠密向量距离很近。
>
> 稀疏向量是在模型约 25 万维词表上只点亮少数几个词，本质是 `{词id: 权重}` 的"关键词账本"，刻画精确字面匹配，哪个词出现就计谁。
>
> 两者要混合用，是因为它们的短板正好互补：稠密懂语义但可能"脑补"跑偏，稀疏精确但不懂同义词。
>
> "只有稀疏能救场"的例子：用户搜缩写 **"DCNN 如何缓解梯度消失"**。
稠密向量会觉得 DCNN、CNN、"深度神经网络"语义都差不多近，可能召回别的文献切片，张冠李戴；而稀疏向量直接按"DCNN"这个 token 精确倒排，只召回真正出现这串字符的切片，一字不差。
缩写、公式符号、参数编号这类"字面必须精确、语义反而帮倒忙"的场景，稀疏不可替代。反过来用户说"卷积神经网络的结构"、文档写"CNN 的结构"，字面不同稀疏匹配不上，靠稠密救场。
所以项目用稠密+稀疏做混合检索，兼顾语义理解和精确命中。

---

## 第 3 题：为什么只喂前 3 个切片、卡 2500 字符？

**标准回答：**

> 这是一个"上下文预算"的省钱设计。文献名、标题与摘要天然出现在文档的封面、目录和开头章节，前 3 个切片通常已经包含足够信息，LLM 完全能识别。
>
> 把全文喂进去有三个坏处：一是 token 成本高、响应慢，文献名识别这种轻任务不需要；二是长上下文会稀释关键信息，反而可能被后面的无关内容带偏；
三是可能撞上模型的输入长度上限。所以代码做了三重限制：只取前 3 个切片、累计字符超 2500 就停止拼接、最后再硬截断一次兜底。用最小的上下文成本拿到稳定结果。

---

## 第 4 题：IVF_FLAT 的 nlist=128 在干什么？不建索引会怎样？

**标准回答：**

> nlist=128 是在建库时把所有向量聚类成 128 个"桶"（簇），"有 1 个索引，索引内部把向量空间划分成 128 个区域（簇).
每个桶有一个聚类中心。查询时不是全库扫描，而是先把查询向量和这 128 个中心比，只打开最近的若干个桶、在桶内做精确比对，其余桶直接跳过，
用极少量精度损失换取大幅提速——这就是"先聚类分桶、桶内暴力检索"的 IVF_FLAT。
>
> 如果向量字段不建索引，就退化成 FLAT 暴力扫描：查询向量要和集合里每一条向量逐一计算相似度再排序，数据量小还能接受，上到几万、几十万条时延迟会无法接受。
索引本质是趁入库时预先建好的"提速目录"，稠密向量用 IVF_FLAT+余弦，稀疏向量用倒排索引+内积，都是为了让相似度检索在大规模数据下仍然可用。

--

    '''

'''
我再说一下我的理解，就是假如我有五个文档，都存进知识库。
但是我的知识库只有一个，里面有一个索引，任何入库的文档都要进行chunk切分并且转为向量，
之后要通过k-means算法计算所有chunk的相似度，就是说我128个簇一早就有了，然后chunk分配给离他语义近的簇

'''