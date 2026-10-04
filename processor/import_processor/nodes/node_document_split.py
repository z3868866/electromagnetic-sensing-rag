# processor/import_processor/nodes/node_document_split.py
import json
import logging
import re
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from processor.import_processor.parent_split_utils import build_parent_child

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError
from processor.import_processor.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """
    节点: 文档切分 (node_document_split)
    将长文档切分成小的 Chunks (切片) 以便检索。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_document_split"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        节点：文档切分（node_document_split）
        整体流程：加载输入→按MD标题初切→长切短合→统计输出→结果备份
        核心目的：将长MD文档切分为长度适中的Chunk，适配大模型上下文窗口和向量检索
        后续扩展点：可在各步骤间新增Chunk元信息补充、自定义切分规则、向量入库前置处理等

        必要参数：md_content、file_title
        更新参数：chunks

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：加载并标准化输入数据
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：构建父子两级（父块=规范章节完整原文，子块=约500字检索片段）
        parents, children = build_parent_child(content)

        # 无标题兜底：整篇无标题时构造单一父块，再切子块
        if not parents:
            parents = [{"parent_uid": "pc_0", "parent_title": "无标题", "content": content}]
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=80,
                separators=["\n\n", "\n", "。", "！", "？", " ", ""])
            children = []
            for j, t in enumerate(fallback_splitter.split_text(content), 1):
                t = t.strip()
                if t:
                    children.append({"parent_uid": "pc_0", "parent_title": "无标题",
                                     "title": "无标题", "content": t, "part": j})

        # 统一补 file_title
        for p in parents:
            p["file_title"] = file_title
        for c in children:
            c["file_title"] = file_title

        # 统计
        lines_count = len(content.split("\n"))
        self.logger.info("=" * 20 + " 父子切分统计 " + "=" * 20)
        self.logger.info(f"MD原始行数：{lines_count} | 父块：{len(parents)} | 子块：{len(children)}")

        # 备份（自动建目录）
        try:
            out_dir = Path("D:/output") / file_title
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "parents.json", "w", encoding="utf-8") as f:
                json.dump(parents, f, ensure_ascii=False, indent=2)
            with open(out_dir / "children.json", "w", encoding="utf-8") as f:
                json.dump(children, f, ensure_ascii=False, indent=2)
            self.logger.info(f"父子切分已备份：{out_dir}")
        except Exception as e:
            self.logger.error(f"父子切分备份失败：{str(e)}")

        # 写入状态
        state["parents"] = parents
        state["chunks"] = children
        return state


    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict[str, str]]) -> None:
        """
        【步骤6】Chunk结果本地JSON备份（便于调试/问题排查，保留处理结果）
        :param state: 项目状态字典，需包含md_dir（备份目录）
        :param sections: 最终处理后的Chunk列表
        """

        try:
            # 拼接备份文件路径：固定文件名，便于查找
            backup_path = Path("D:/output") / state.get("file_title") / "chunks.json"
            # 写入JSON文件：保留中文/格式化缩进，便于人工查看
            with open(backup_path, "w", encoding="utf-8") as f:
                """
                sections是Python 嵌套数据结构（List[Dict[str, str]]，列表里装字典，字典里可能嵌套字符串 / 数字等），而普通文件写入
                （如f.write(sections)）仅支持写入字符串，直接写 Python 数据结构会报错。
                json.dump的核心作用就是：将 Python 原生数据结构（列表、字典、字符串、数字等）直接序列化并写入 JSON 文件，无需手动转换为字符串，
                同时保证数据格式规范、可跨语言 / 跨场景读取，完美适配「Chunk 列表备份」的需求。
                """
                json.dump(
                    sections,
                    f,
                    #开启 True："title": "\u4e00\u7ea7\u6807\u9898"（乱码，无法直接看）；
                    #开启 False："title": "一级标题"（正常中文，人工可直接阅读）。
                    ensure_ascii=False,  # 保留中文，不转义为\u编码
                    indent=2             # 格式化缩进，便于阅读
                )
            self.logger.info(f"步骤6：Chunk结果备份成功，备份文件路径：{backup_path}")
        except Exception as e:
            # 备份失败仅记录日志，不终止主流程
            self.logger.error(f"步骤6：Chunk结果备份失败，错误信息：{str(e)}", exc_info=False)

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict[str, str]]) -> None:
        """
        【步骤5】输出文档切分统计信息（日志记录，便于监控/调试）
        :param lines_count: MD原始文本总行数
        :param sections: 最终处理后的Chunk列表
        """
        chunk_num = len(sections)
        # 输出核心统计信息：原始行数/最终Chunk数/首个Chunk预览
        self.logger.info("-" * 50 + " 文档切分统计信息 " + "-" * 50)
        self.logger.info(f"MD原始文本总行数：{lines_count}")
        self.logger.info(f"最终生成Chunk数量：{chunk_num}")

    def _split_long_section(self, section: Dict[str, str]) -> List[Dict[str, str]]:
        """
        【辅助函数】超长章节二次切分（核心适配LangChain分割器）
        功能：单个章节内容超限时，按「段落→句子→空格」从粗到细切分，保留语义
        切分规则：1.先按空行(段落) 2.再按换行 3.最后按中英文标点/空格
        :param section: 原始章节字典，必须包含content键，可选title/file_title等
        :return: 切分后的子章节列表，每个子章节带父标题/序号等元信息
        """
        content = section.get("content")

        # 1. 长度未超限，无需切分，直接返回
        if len(content) <= self.config.max_content_length:
            return [section]

        # 2. 切割文本
        # 章节标题
        title = section.get("title")
        #计算标题的长度
        prefix = f"{title}\n\n" if title else ""

        # 计算正文的长度
        available_len = self.config.max_content_length - len(prefix)

        if available_len <= 0:
            self.logger.warning("章节标题过长，无法切分")
            return [section]


        # 去掉title部分，便于以后判断
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title) :].lstrip()

        # 对正文进行切割
        # langchain扽文本切割器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len, # 切片长度（扣除标题）
            chunk_overlap=0, #重叠部分的长度
            separators=[
                "\n\n",
                "\n",
                "。",
                "？",
                "！",
                "；",
                ".",
                "?",
                "!",
                ";",
                " ",
            ],
        )

        sub_sections = []
        chunks = splitter.split_text(body)
        for idx, chunk in enumerate(chunks, start=1):
            text = chunk.strip()
            if not text:
                continue

            full_text = (prefix + text).strip()
            new_title = f"{title}-{idx}"

            sub_sections.append(
                {
                    "parent_title": title,
                    "title": new_title,
                    "content": full_text,
                    "part": idx,
                    "file_title": section.get("file_title")
                }
            )

        return sub_sections


    def _step_4_refine_chunks(self, sections: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
       【步骤4】Chunk精细化处理（核心：长切短合，适配大模型/检索）
       执行流程：1.切分超长章节 2.合并过短章节 3.父标题兜底（适配Milvus向量库schema）
       :param sections: 步骤3处理后的章节列表
       :return: 长度适中、低碎片化的最终Chunk列表
       """
        # 阶段1：超长段落切分，控制在最大字符数内（self.config.max_content_length）
        refined_split = []
        for section in sections:
            sub_sections = self._split_long_section(section)
            refined_split.extend(sub_sections)

        # 阶段2：短段落合并，减少内容碎片化
        final_sections = self._merge_short_sections(refined_split)

        # 阶段3：父标题兜底
        # 兜底规则：无parent_title则用自身title，title也无则填空字符串
        for sec in final_sections:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or ""

        return final_sections


    def _merge_short_sections(self, sections: List[Dict[str,str]]) -> List[Dict[str,str]]:

        #边界处理：空列表直接范湖
        if not sections:
            return []


        #合并后的结果
        merged_sections = []
        # 保存当前要合并的chunk
        current_chunk = None

        for section in sections:
            #获取第一个section赋值给current_chunk
            if current_chunk is None:
                current_chunk = section
                continue

            #段落是否过短
            is_current_short = len(current_chunk["content"]) < self.config.min_content_length
            is_same_parent = current_chunk.get("parent_title") == section.get("parent_title")

            if is_current_short and is_same_parent:
                #合并前清理每个content前面重复的parent_title
                parent_title = section.get("parent_title")
                section_content = section.get("content")
                if parent_title and section_content.startswith(parent_title):
                    section_content = section_content[len(parent_title):].lstrip()

                #合并段落
                current_chunk["content"] += "\n\n" + section_content

                #处理part
                if "part" in section:
                    current_chunk["part"] = section["part"]

            else:
                #保存当前段落
                merged_sections.append(current_chunk)
                current_chunk = section

        if current_chunk is not None:
            merged_sections.append(current_chunk)

        return merged_sections


    def _step_3_handle_no_title(self, content: str, sections: List[Dict[str, str]], title_count: int,
                                file_title: str) -> List[Dict[str, str]]:
        """
        【步骤3】无标题兜底处理
        功能：若MD中未识别到任何标题，将全文作为一个整体处理，避免后续逻辑异常
        :param content: 标准化后的MD完整内容
        :param sections: 步骤2切分后的章节列表
        :param title_count: 步骤2识别的有效标题数量
        :param file_title: 所属文件标题
        :return: 兜底后的章节列表
        """
        if title_count == 0:

            # 将content交给LLM，总结一个标题出来
            # 提示词：根据content和file_title总结一个段落标题，不要有空格，不要换行，控制在20个字符以内

            # 无标题处理
            return [{
                "title": "无标题",
                "content": content,
                "file_title": file_title
            }]

        return sections

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict[str, str]], int, int]:
        """
        【步骤2】按Markdown标题初次切分（核心：按#分级切分，跳过代码块内标题）
        LangChain前置预处理：将整份MD按标题拆分为独立章节，为后续精细化切分做基础
        :param content: 标准化后的MD完整内容（字符串）
        :param file_title: 所属文件标题，用于标记章节归属
        :return: 切分后的章节列表/有效标题数量/原始文本总行数
        """

        # 1、定义标题正则
        # 正则匹配Markdown 1-6级标题（核心规则，适配缩进/标准格式）
        # ^\s*：行首允许0/多个空格/Tab（兼容缩进的标题）
        # #{1,6}：匹配1-6个#（对应MD1-6级标题）
        # \s+：#后必须有至少1个空格（区分#是标题还是普通文本）
        # .+：标题文字至少1个字符（避免空标题）
        title_pattern = r'\s*#{1,6}\s+.+'

        # 2、初始化需要的数据
        lines = content.split("\n")
        sections = []  # 章节列表
        title_count = 0  # 标题数量
        current_title = ""  # 当前章节的标题
        current_lines = []  # 当前标题和下一个标题之间的文本内容
        in_code_block = False  # 代码块标记：False当前没在代码块中，True当前在代码块中

        # 3、定义内部函数组装sections列表
        def _flush_section():
            """内部辅助函数：将当前缓存的章节写入sections，空缓存则跳过"""
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                # 每段使用 \n换行区分
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        # 4、逐行遍历，识别标题和普通行以及代码快
        for line in lines:
            # 去空格
            striped_line = line.strip()

            # # 4.1 识别代码快边界 ``` 或 ~~~
            # if striped_line.startswith("```") or striped_line.startswith("~~~"):
            #     in_code_block = not in_code_block
            #     current_lines.append(striped_line)
            #     continue
            # 4.1 识别代码块边界（至少 3 个连续 ` 或 ~）
            # 围栏开始：可带语言标识（如 ```txt、```python），正则放开 info string
            fence_open_match = re.match(r'^(`{3,}|~{3,})(.*)$', striped_line)
            # 围栏结束：必须是纯围栏，不含语言标识
            fence_close_match = re.match(r'^(`{3,}|~{3,})$', striped_line)

            if not in_code_block and fence_open_match:
                # 进入代码块，记录开始围栏（字符与长度）
                in_code_block = True
                code_block_start_marker = fence_open_match.group(1)
                current_lines.append(line)
                continue

            if in_code_block and fence_close_match:
                close_marker = fence_close_match.group(1)
                # 结束围栏需与开始同字符、长度不小于开始围栏
                if close_marker[0] == code_block_start_marker[0] and \
                        len(close_marker) >= len(code_block_start_marker):
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(line)
                continue

            # 4.2 表示标题
            is_valid_title = (not in_code_block) and re.match(title_pattern, line)
            if is_valid_title:
                #遇到标题行则先将上一个片段写入section
                _flush_section()
                current_title = striped_line
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(striped_line)

        _flush_section()

        return sections, title_count, len(lines)


    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        """
        【步骤1】获取并预处理输入数据
        功能：从状态字典中提取MD内容/文件标题/最大长度，做基础标准化
        :param state: 项目状态字典（ImportGraphState），包含md_content等核心键
        :return: 标准化后的MD内容/文件标题（无内容则返回None,None）
        """

        # 中间节点也可以只做轻量级的防御性判断，不做完整的磁盘级别的检查 => 前置校验 + 信任状态

        # 1、非空校验
        file_title = state.get("file_title")
        if not file_title:
            raise StateFieldError(field_name="file_title", message="文件标题不能为空", expected_type=str)

        md_content = state.get("md_content")
        if not md_content:
            raise StateFieldError(field_name="md_content", message="文件内容不能为空", expected_type=str)

        # 2、基础标准化：统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        return md_content, file_title

if __name__ == "__main__":
    setup_logging()

    md_path = r"D:\output\hak180产品安全手册\hak180产品安全手册_new.md"
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    init_state = {
        "md_path": md_path,
        "md_content": md_content,
        "file_title": "hak180产品安全手册"
    }

    # 执行核心处理流程
    node_document_split = NodeDocumentSplit()
    result = node_document_split(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))
'''
    1. 为什么不能整篇文档直接向量化检索？说出 "语义稀释" 和"噪声" 两个角度。
    语义稀释（Semantic Dilution）：向量化是把文本转成高维空间的“坐标点”。整篇文档包含“安装指南”、“故障代码”、“安全警告”等十几个主题。最终生成的向量，是这十几个主题的“平均大杂烩”。当你搜索“设备红灯闪烁怎么办”时，这个平均向量会把“安装步骤”和“保修政策”都掺和进来，导致检索出来的内容像隔着一层毛玻璃，模糊不清，精准命中率极低。
    噪声（Noise）：整篇文档里有大量“无关杂音”——比如页眉页脚、版本修订记录、表格格式符、甚至代码注释。如果直接向量化，这些噪声会像沙子一样混进珍珠里，污染向量表征。当你做相似度匹配时，匹配上的很可能不是核心业务知识，而是那些高频出现的噪声词（如“版权所有”、“参见第X章”），导致召回结果全是垃圾。
    
    2.为什么代码块里的`  # ` 要特殊处理？靠哪个变量、什么机制实现？
这里的 `#` 是 Python 注释（C 语言里还有 `#include`、`#define`），**不是 Markdown 标题**。不屏蔽的话，扫描器看到 "# 这行是代码注释" 会误以为新章节开始，把一段完整代码拦腰切断。

机制：`in_code_block` 布尔标记，遇到 `\`\`\`` 就翻转 True/False（这就是状态机），**只有 `in_code_block=False` 时的 `#` 才认作标题**。    

    3.每个子切片为什么都要重复带上标题？不带会发生什么具体问题？
    想象你是仓库分拣员，手里拿着一堆撕碎的小纸条（子切片），要分门别类放进不同的筐里（向量库）。

场景A：纸条上不带标题（不带parent_title）

你捡起一张纸条，上面写着：“电压不能超过220V”。

你满头问号：这是电饭煲的说明书？还是高压电线的警告牌？还是电脑电源的适配器？

结果：你只能胡乱把它扔进离你最近的筐里。后续用户问“电饭煲怎么用”时，这张“高压电线”的纸条被搜出来，AI拿着它乱回答，彻底翻车。

场景B：纸条上贴了标签，写着“父标题：电饭煲安全须知”

你看到纸条内容“电压不能超过220V”，再看看标签。

你一秒都不犹豫，精准扔进“电饭煲”筐。用户查电饭煲时，这条信息完美命中。

再重点看代码里的“合并短段落”（_merge_short_sections）：
如果两张短纸条没带“父标签”，机器根本不敢把它们粘在一起，因为不知道它们是不是讲同一件事。带了标签，机器才能放心地说：“哦，你们两个都来自‘电饭煲’，太短了，我给你们粘成一张长纸条。”

一句话人话：不带标题的切片，就像没写收件人地址的快递，永远送不到正确的人手里。
    
    4.
    `RecursiveCharacterTextSplitter`
    拿到分隔符列表
    `["\n\n", "\n", "。", " "]`，它为什么按这个顺序用？一个
    3000
    字、段落都很短的章节，会主要靠哪个分隔符切？
    
    没问题！我们这次不聊代码，直接拿出一篇**真实的3000字“短段落”文档**，把 `RecursiveCharacterTextSplitter` 按顺序切分的过程，一步一步推演给你看。

---

### 📄 现实文档样例（3000字，段落极短）

假设这是《咖啡机安全手册》的一部分，全文3000字，但每段只有1~2句话（**到处都是空行 `\n\n`**）：

```
咖啡机使用前请阅读本手册。
本产品仅限室内使用。

请勿将机身浸入水中。
清洁前务必拔掉电源插头。

水箱容量为1.5升。
请使用纯净水或过滤水。

（中间还有几百个这样的短段落，直到凑满3000字）...

长时间不使用请拔掉电源。
产品保修期为一年。
```

**关键特征**：每段文字大约30~50字，段与段之间都有**空行（`\n\n`）**。

---

### ⚙️ 机器设定的阈值（`chunk_size`）

代码里配置的 `max_content_length`（也就是 `chunk_size`）假设是 **500个字符**。

---

### 🔪 第1轮切分：使用第一个分隔符 `\n\n`（按空行/段落切）

机器的操作：

1. 扫描全文，遇到 `\n\n`（空行）就砍一刀。
2. 因为全文段落极短，每个被砍出来的“块”，**最长也就50个字**。
3. 机器挨个检查这些块的长度：`50 < 500（阈值）`。

**机器的内心OS（判断逻辑）**：
> “哦？我按空行切出来的每一块，都远远小于500字。说明这些块已经很完美了！完全没有必要再往下切了。”

**结论**：切分**立即终止**。机器直接返回这几十个“短段落块”作为最终结果。

---

### ❌ 第2、3、4轮（`\n`、`。`、` `）发生了什么？

它们**全程都在睡大觉，一次都没被执行**。

因为递归切割器的铁律是：
> **“只有当一个块超过阈值时，我才拿下一个分隔符去切它。如果上一轮切出来的块全都合格，我就直接收工，绝不多切一刀。”**

---

### 🔄 换个极端场景，你就明白“递归”的意思了

如果文档**只有一个超长段落，且没有空行**（例如3000字都在一段里）：

- **第1轮（`\n\n`）**：找不到空行，切不动，整段还是3000字 → 超过500字，不合格。
- **第2轮（`\n`）**：按换行切。假如这超长段落里有换行，切成几块。如果每块还是超过500字，继续降级。
- **第3轮（`。`）**：按句号切。如果某个句子还超过500字（比如超长无标点英文），继续降级。
- **第4轮（` `）**：按空格切。这是保底，暴力把单词拆开。

---

### 🎯 给你一个铁定的规律，以后遇到直接套用

- **如果文档里空行密集（段落短）** → **仅靠 `\n\n` 一刀切完**，后面的分隔符全是摆设。
- **如果文档里没有空行** → `\n\n` 失效，退到 `\n`。
- **如果文档连换行都没有** → 退到 `。` 切句子。
- **如果文档连标点符号都没有（如长串英文无空格）** → 最后退到 ` `（空格）暴力拆词。

针对你问的“3000字短段落章节”，**现实中的真实结果就是：全靠 `\n\n`（空行）切分**。你可以把这篇文档复制到记事本里，数一下空行的数量，机器就会切成多少个块。

这个现实例子够直观吗？如果还不够，我可以**虚构一份具体的前后对照文本**，把“切之前的原文”和“切之后的Chunk列表”直接打印出来给你看。😊
    
    
'''
