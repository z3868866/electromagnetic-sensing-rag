import json
from typing import List, Dict

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config.lm_config import lm_config
from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase, T
from processor.query_processor.prompt.item_name_confirm import ITEM_NAME_EXTRACT_TEMPLATE, \
    ITEM_NAME_EXTRACT_SYSTEM_PROMPT
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.milvus_utils import get_milvus_client, online_hybrid_search
from utils.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from utils.task_utils import add_done_task


class NodeItemNameConfirm(NodeBase):

    name: str = "node_item_name_confirm"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        必要参数：session_id、original_query
        更新参数：history、rewritten_query、item_names、answer

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：校验参数
        session_id, original_query = self._step_1_validate_param(state)
        logger.info(f"步骤1：参数校验通过")

        # 步骤2：获取历史记录
        history = get_recent_messages(session_id)
        # 更新状态
        state["history"] = history
        logger.info("步骤2：state信息更新成功,")

        # 步骤3：用户初始消息保存
        message_id = save_chat_message(session_id, "user", original_query)
        logger.info(f"步骤3：用户消息已初始保存, ID: {message_id}")

        # 步骤4：提取信息
        extract_res = self._step_4_extract_info(original_query, history)
        
        item_names = extract_res.get("item_names")
        # 用 or 兜底：键缺失或为 None/空串时都退回原始问题，保证下游拿到的改写问题永远非空
        rewritten_query = extract_res.get("rewritten_query") or original_query
        # 更新状态
        state["rewritten_query"] = rewritten_query

        # 4b 别名精确匹配（词典优先）：命中别名/标准名字面相等 → 直接确认
        exact = self._step_4b_exact_alias_match(item_names)
        exact_confirmed = list(exact.get("confirmed", {}).values())
        remaining = exact.get("remaining", [])

        align_result = {"confirmed_item_names": list(exact_confirmed), "options": []}
        if len(remaining) > 0:
            # 词典未命中的指称，退到向量语义检索
            query_results = self._step_5_vectorize_and_query(remaining)
            vec_align = self._step_6_align_item_names(query_results)
            align_result["confirmed_item_names"] += vec_align.get("confirmed_item_names", [])
            align_result["options"] = vec_align.get("options", [])
        elif len(item_names) == 0:
            logger.info("Node: 未提取到名称，跳过检索")

        # 7. 检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 8. 写入最终历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)

        add_done_task(state.get("session_id"), self.name, state.get("is_stream"))
        return state

    # def _test_exception(self):
    #     try:
    #         a = 9 / 0
    #         return 666
    #     except Exception as e:
    #         logger.exception(f"{self.name} 执行异常: {e}")
    #         # return 999
    #         raise
    def _step_1_validate_param(self, state):

        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("参数 original_query 不能为空")

        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("参数 session_id 不能为空")

        return session_id, original_query

    def _step_4_extract_info(self, original_query, history):

        """
       利用LLM从当前问题以及历史会话中提取出主要询问的文献名称item_names（可多个，JSON列表形式）
       若文献名不够明确则返回空列表，同时根据上下文重新改写问题，保证问题独立完整
       :param query: 字符串 - 用户当前原始查询问题（如："这个多少钱？"）
       :param history: 列表[字典] - 近期会话历史，每条消息含role/text等字段，
                       格式：[{"role": "user/assistant", "text": "消息内容", "_id": "消息ID"}, ...]
       :return: 字典 - 提取结果，固定包含2个字段，格式：
                {
                    "item_names": ["文献名1", "文献名2", ...],  # 提取的文献名列表，无则空列表
                    "rewritten_query": "改写后的完整问题"       # 包含文献名的独立问题，无则返回原始query
                }
       """

        try:
            # 1. 准备提示词
            chant_model = ChatOpenAI(
                model = lm_config.item_model,
                api_key=lm_config.api_key,
                base_url=lm_config.base_url,
                temperature=lm_config.llm_temperature,
                # 开启JSON标准输出模式，强制模型返回可解析的json_object
                model_kwargs={
                    "response_format": {"type": "json_object"}
                }
            )

            # 2. 组装历史记录
            history_text = ""
            for msg in history:
                role = msg.get("role")
                content = msg.get("text")
                history_text += f"{role}: {content}\n"

            user_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
                history_text =  history_text,
                query = original_query
            )

            # 3. 组装message
            messages = [
                SystemMessage(content = ITEM_NAME_EXTRACT_SYSTEM_PROMPT),
                HumanMessage(content = user_prompt)
            ]


            # 4. 调用模型
            response = chant_model.invoke(messages)
            content = response.content

            # 5. 数据清洗：去掉可能的代码围栏（```json 和 ```）
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")

            # 6. JSON数据的解析（将JSON字符串转换为字典）
            result = json.loads(content)

            # 7. 健壮性判断：确保返回结果包含item_names和rewritten_query字段
            if "item_names" not in result:
                logger.warning("大模型返回结果缺少item_names字段")
                result["item_names"] = []

            if "rewritten_query" not in result:
                logger.warning("大模型返回结果缺少rewritten_query字段")
                result["rewritten_query"] = original_query

            result["item_names"] = [
                name
                .replace(" ", "")
                .replace("\n", "")
                .replace("\t", "")
                .replace("\r", "")
                for name in result["item_names"]
            ]

            return result

        except Exception as e:
            # 捕获所有异常（如LLM调用失败、JSON解析失败等），记录错误日志
            logger.error(f"大模型调用异常：{e}")
            # 异常时返回默认结果：空文献名列表+原始查询（字段必须完整，禁止返回 None）
            return {"item_names": [], "rewritten_query": original_query}

    def _step_4b_exact_alias_match(self, item_names) -> dict:
        """
        别名词典精确匹配（优先于向量检索）。加载 kb_item_names 全量标准名与别名，
        对提取名做归一化（去空白、小写）后字面比较，命中即映射到标准名。
        :return: {"confirmed": {提取名: 标准名}, "remaining": [未命中提取名]}
        """
        milvus_client = get_milvus_client()
        coll = milvus_config.item_name_collection

        def norm(s):
            return ("" if s is None else str(s)).replace(" ", "").replace("　", "").lower()

        confirmed, remaining = {}, []
        # 分页拉全量词典：query 单次 limit 有上限，循环 offset 翻页避免静默截断；
        # 加安全上限防止异常情况下无限翻页
        ALIAS_PAGE_SIZE = 1000
        ALIAS_MAX_ROWS = 100_000
        rows = []
        try:
            offset = 0
            while len(rows) < ALIAS_MAX_ROWS:
                batch = milvus_client.query(
                    coll, filter="name_id >= 0",
                    limit=ALIAS_PAGE_SIZE, offset=offset,
                    output_fields=["item_name", "aliases"],
                    consistency_level="Strong",
                )
                rows.extend(batch)
                if len(batch) < ALIAS_PAGE_SIZE:
                    break
                offset += len(batch)
        except Exception as e:
            logger.warning(f"别名词典加载失败，退到向量检索：{e}")
            rows = []

        key2std = {}
        for r in rows:
            std = r.get("item_name", "")
            keys = [std] + [a.strip() for a in (r.get("aliases", "") or "").split("|") if a.strip()]
            for k in keys:
                nk = norm(k)
                if nk and nk not in key2std:
                    key2std[nk] = std

        for name in item_names:
            std = key2std.get(norm(name))
            if std:
                confirmed[name] = std
            else:
                remaining.append(name)

        logger.info(f"别名精确命中 {len(confirmed)} 个，转向量 {len(remaining)} 个")
        return {"confirmed": confirmed, "remaining": remaining}

    def _step_5_vectorize_and_query(self, item_names) -> List[Dict]:
        """
        对提取出的 item_names 逐个在 kb_item_names 集合执行在线混合检索，获取匹配评分。
          - 稠密向量由在线 text-embedding-v4 生成；
          - 稀疏侧传原文，由 BM25 Function 自动计算。
        :return: [{"extracted_name": str, "matches": [{"item_name": str, "score": float}]}]
        """
        milvus_client = get_milvus_client()
        collection_name = milvus_config.item_name_collection

        results = []
        if not milvus_client:
            logger.warning("Milvus 获取失败，跳过检索")
            return results

        for name in item_names:
            search_res = online_hybrid_search(
                client=milvus_client,
                collection_name=collection_name,
                query_text=name,
                limit=5,
                output_fields=["item_name"],
                consistency_level="Strong"
            )

            matches = []
            if search_res and len(search_res) > 0:
                for hit in search_res[0]:
                    matches.append({
                        "item_name": hit.get("entity", {}).get("item_name"),
                        "score": hit.get("distance")
                    })

            results.append({"extracted_name": name, "matches": matches})

        return results

    def _step_6_align_item_names(self, query_results) -> dict:
        """
        6 根据Milvus搜索评分，逐个对齐step4提取的item_names，生成「确认文献名」和「候选文献名」以及 [没有可选文献]
        对齐规则（优先级a>b>c>d）：
                # a  如果只有一个匹配结果评分高于0.85 → 直接确认该文献名
                # b  如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
                优化 ab 所有评分高于0.85的都可以直接确认
                c  如果无0.85分以上结果 → 取分数≥0.6的最高前5个作为候选
                d  如果无0.6分及以上结果 → 不返回任何文献名（确认+候选均为空）
        :param query_results: 列表[字典] - step5的返回结果，每个文献名的搜索匹配数据（格式同step5返回值）
        :return: 字典 - 文献名对齐结果，包含确认列表和候选列表，格式：
                 {
                     "confirmed_item_names": ["确认文献名1", "确认文献名2"],  # 去重后的确认文献名，无则空列表
                     "options": ["候选文献名1", "候选文献名2", ...]          # 去重后的候选文献名，无则空列表
                 }
        """
        # 1. 初始化文献名列表
        # 确认的
        confirmed_item_names: List[str] = []
        # 候选的
        options: List[str] = []

        for res in query_results:

            # 提取上一步组装的信息
            extracted_name = res.get("extracted_name")
            matches = res.get("matches")
            # 如果没有匹配的结果，则跳过对齐步骤
            if not matches:
                continue


            # 筛选高置信度等分的结果： > 0.85
            high = [m for m in matches if m.get("score") > 0.85]
            mid = [m for m in matches if m.get("score") >= 0.65]

            # 优化 ab 所有评分高于0.85的都可以直接确认
            if len(high) > 0:
                for m in high:
                    confirmed_item_names.append(m.get("item_name"))
                continue
            # 筛选高置信度等分的结果： >= 0.65
            # # a  如果只有一个匹配结果评分高于0.85 → 直接确认该文献名
            # if len(high) == 1:
            #     confirmed_item_names.append(high[0].get("item_name"))
            #     continue
            #
            # # b  如果多条匹配结果评分超过0.85 → 优先取与原始提取名相同的，无则取分数最高的
            # if len(high) > 1:
            #     picked = None
            #     if extracted_name:
            #
            #         # 优先取与原始提取名相同的
            #         for m in high:
            #             if m.get("item_name") == extracted_name:
            #                 picked = m
            #                 break
            #
            #     if not picked:
            #         # 无则取分数最高的
            #         picked = high[0]
            #
            #     confirmed_item_names.append(picked.get("item_name"))
            #     continue

            # c  如果无0.85分以上结果 → 取分数≥0.6的最高前3个作为候选
            if len(mid) > 0:
                for m in mid[:3]:
                    options.append(m.get("item_name"))

            # d  如果无0.6分及以上结果 → 不返回任何文献名（确认+候选均为空）
            # pass


        # 去重
        return {
            "confirmed_item_names": list(set(confirmed_item_names)),
            "options": list(set(options))
        }

    def _step_7_check_confirmation(self, state, align_result, history):
        """
        7 检查step6对齐后的文献名状态，分3种分支更新state，并同步更新历史消息的文献名关联
        :param state: 字典 - 原始会话状态，包含session_id/original_query等核心字段
        :param align_result: 字典 - step6的对齐结果
        :param history: 列表[字典] - 近期会话历史
        :return: 字典 - 更新后的会话状态，包含item_names/answer
        """

        # 1. 从对齐结果中获取确认的文献名列表，无，则返回空列表
        confirmed = align_result.get("confirmed_item_names", [])

        # 2. 从对齐结果中获取候选的文献名列表，无，则返回空列表
        options = align_result.get("options", [])

        # **分支A（有确认文献）**：更新 State 中的 `item_names`，并批量回填历史消息中缺失的文献名关联。
        if confirmed:

            # 获取历史记录的id列表
            ids_to_update = []
            for msg in history:
                if not msg.get("item_names"):
                    msg_id = msg.get("_id")
                    ids_to_update.append(str(msg_id))

            # 根据id列表做批量更新
            if ids_to_update:
                update_message_item_names(ids_to_update, confirmed)

            # 更新会话状态
            state["item_names"] = confirmed
            state["answer"] = ""

            return state

        # **分支B（有候选选项）**：生成澄清反问句（如“您是指...吗？”），写入 State 的 `answer` 字段，清空 `item_names`。
        if options:
            options_str = "，".join(options)
            state["answer"] = f"您是想问以下哪篇文献或方法：{options_str}？请明确具体名称。"
            state["item_names"] = []

            return state

        # **分支C（无结果）**：生成拒识回复（如“未找到相关文献...”），写入 State 的 `answer` 字段。
        state["answer"] = "未找到相关文献，请提供准确的文献或方法名称。"
        state["item_names"] = []
        return state

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        """
         8 把本次处理的核心信息（用户问题、助手答案、文献名、改写查询）写入MongoDB的会话历史
         包含2个核心操作：1. 写入助手答案（若有）；2. 更新用户原始问题的关联信息
         :param state: 字典 - step6更新后的会话状态，包含answer/item_names等字段
         :param session_id: 字符串 - 会话唯一标识
         :param rewritten_query: 字符串 - step3改写后的完整问题
         :param message_id: 字符串 - 本次用户问题的消息唯一ID
         :return:
         """
        # 若会话状态中有助手答案（分支B/C），写入助手消息到历史
        if state.get("answer"):
            save_chat_message(
                session_id = session_id,
                role = "assistant",
                text = state.get("answer"),
                item_names=state.get("item_names")
            )

        # 强制更新本次用户原始问题的关联信息
        save_chat_message(
            session_id=session_id,
            role="user",
            text=state.get("original_query"),
            item_names=state.get("item_names"),
            rewritten_query=rewritten_query,
            message_id=message_id
        )

if __name__ == "__main__":

    # 初始化状态
    init_state = {
        "session_id": "test_session_002",
        "original_query": "怎么调节转印温度？"
    }

    node_item_name_confirm = NodeItemNameConfirm()
    # result = node_item_name_confirm.process(init_state)
    result = node_item_name_confirm(init_state)

    # json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logger.info(result)
    # logger.info(serialize_json(result, indent=4))

