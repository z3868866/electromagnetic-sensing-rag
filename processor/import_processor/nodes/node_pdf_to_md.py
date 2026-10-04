# processor/import_processor/nodes/node_pdf_to_md.py
import hashlib
import json
import logging
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, FileProcessingError, ConfigurationError, \
    PdfConversionError
from processor.import_processor.state import ImportGraphState


def _safe_stem(name: str, max_len: int = 60) -> str:
    """
    将原始文件名（无后缀）转换为适合落盘的安全短名：
    1. 替换 Windows 非法文件名字符并压缩空白；
    2. 追加基于原名的 8 位短哈希，避免截断后重名；
    3. 截断到 max_len，防止目录/文件路径叠加超过 Windows MAX_PATH(260)。
    原始完整名称通过 state 元数据保留，不用于构造路径。
    """
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    short_hash = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    keep = max_len - len(short_hash) - 1
    if len(cleaned) > keep:
        cleaned = cleaned[:keep].rstrip()
    return f"{cleaned}_{short_hash}"


class NodePDFToMD(BaseNode):
    """
    PDF 转 Markdown 节点：PDF结构化解析
    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):   #这是子类必须实现的核心方法，被 BaseNode.__call__ 调用。
        """

        :param state:
        :return:
        """

        # 1. 参数校验并返回Path结果
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2. 将PDF上传到MinerU并轮询结果最后得到解压文件路径
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)
        self.logger.info(zip_url)

        # 3. 下载ZIP包并解压并且得到md的绝对路径
        md_path = self._step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem)

        # 4. 将MD文件的内容都取出来
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5. 返回结果
        state["md_content"] = md_content
        state["md_path"] = md_path
        return state
        # return {
        #     "md_content": md_content,
        #     "md_path": md_path
        # }

    def _step_3_download_and_extract(self, zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
        """
        步骤3：下载MinerU解析结果ZIP包并解压，提取目标MD文件
        核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件 → 重命名统一为PDF同名
        参数：zip_url-ZIP包下载链接；output_dir_obj-输出目录Path；pdf_stem-PDF无后缀纯名称
        返回：最终MD文件的字符串格式绝对路径
        异常：RuntimeError(下载失败)
        """

        # 1.下载zip
        response = requests.get(zip_url)

        if response.status_code != 200:
            raise FileProcessingError("下载失败")

        # 落盘统一使用安全短名，避免长文件名导致路径超过 Windows MAX_PATH(260)
        safe_stem = _safe_stem(pdf_stem)

        # 定义ZIP包的保存路径
        zip_save_path = output_dir_obj / f"{safe_stem}_result.zip"
        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        self.log_step("ZIP下载", "下载成功")

        # 2. 解压ZIP
        # 解压目录
        extract_target_dir = output_dir_obj / safe_stem

        # 删除已有目录
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)

        # 创建新目录
        extract_target_dir.mkdir(parents=True, exist_ok=True)

        # 解压ZIP
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        self.log_step("ZIP下载", "解压完成")

        # 3. 在解压结果中递归查找 Markdown 文件（不写死 full.md，兼容 MinerU 命名差异）
        md_candidates = list(extract_target_dir.rglob("*.md"))
        if not md_candidates:
            raise FileProcessingError("解压结果中未找到 Markdown 文件")
        target_md_file = md_candidates[0]

        # 统一重命名为安全短名，控制最终路径长度
        new_md_path = target_md_file.with_name(f"{safe_stem}.md")
        if target_md_file != new_md_path:
            target_md_file.rename(new_md_path)

        return str(new_md_path.absolute())


    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        步骤2：上传PDF至MinerU并轮询解析任务状态
        核心流程：配置校验 → 获取上传链接 → 文件上传 → 任务轮询（直至完成/失败/超时）
        参数：pdf_path_obj-已校验的PDF Path对象
        返回：解析结果ZIP包下载链接full_zip_url
        异常：ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
        """
        # 1、配置文件校验
        if not self.config.mineru_base_url:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_BASE_URL 参数")
        if not self.config.mineru_api_token:
            raise ConfigurationError("MinerU配置缺失：请在 .env 文件中正确配置 MINERU_API_TOKEN 参数")

        # 2. 调用MinerU的远程API接口，获取上传链接
        # 2.1 组织上传需要的数据
        token = self.config.mineru_api_token
        url = f"{self.config.mineru_base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm"
        }

        # 2.2 发送请求
        response = requests.post(url, headers=header, json=data)

        # 2.3 获取响应结果
        if response.status_code != 200:
            raise PdfConversionError(
                '获取上传链接失败. 响应状态:{} ,响应结果:{}'.format(response.status_code, response))

        result = response.json()
        if result["code"] != 0:
            raise PdfConversionError('获取上传链接失败. 返回数据：{}'.format(result["msg"]))

        # 2.4 获取到上传链接
        # 批量提取任务 id，可用于批量查询解析结果
        batch_id = result["data"]["batch_id"]
        # 文件上传链接
        urls = result["data"]["file_urls"]

        # 3. 文件上传
        with open(pdf_path_obj, 'rb') as f:
            res_upload = requests.put(urls[0], data=f)
            if res_upload.status_code != 200:
                raise PdfConversionError(f'{urls[0]}文件上传失败')

            self.logger.info('上传文件成功!')

        # 4. 获取解析结果
        poll_url = f"{self.config.mineru_base_url}/extract-results/batch/{batch_id}"

        # 轮询获取
        start_time = time.time()  # 记录当前时间
        timeout_seconds = 600  # 最大超时时间
        poll_interval = 3  # 轮询间隔
        self.log_step(
            step_name="轮询开始",
            message=f"轮询间隔: {poll_interval}s, 超时时间: {timeout_seconds}s，batch_id:{batch_id}"
        )
        while True:
            #已消耗时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
              raise TimeoutError(f"[任务轮询]超时，已消耗时间: {int(elapsed_time)}s，batch_id:{batch_id}")

            # 获取任务结果
            try:
                poll_res = requests.get(poll_url, headers=header, timeout=10)
            except Exception as e:
                self.logger.warning(f"网络请求异常，{poll_interval}s后重试，batch_id:{batch_id}")
                self.logger.warning(f"异常信息：{str(e)}")
                time.sleep(poll_interval)
                continue

            # print(poll_res.status_code)
            # print(poll_res.json())
            # print(poll_res.json()["data"])
            if poll_res.status_code != 200:
                raise PdfConversionError(f'[任务轮询]失败，状态码：{poll_res.status_code}')

            poll_res_json = poll_res.json()
            if poll_res_json["code"] != 0:
                raise PdfConversionError(f'[任务轮询]失败，错误码：{poll_res_json["code"]}')

            extract_results = poll_res_json["data"]["extract_result"]
            extract_result = extract_results[0]
            # 获取任务的状态值
            data_state = extract_result["state"]
            if data_state == "done":
                self.log_step(
                    step_name="任务轮询",
                    message=f"解析完成s, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                full_zip_url = extract_result["full_zip_url"]
                self.log_step(
                    step_name="任务轮询",
                    message=f"获取全量zip地址成功: {full_zip_url}, 总耗时: {int(elapsed_time)}s，batch_id:{batch_id}"
                )

                return full_zip_url

            elif data_state == "failed":
                raise PdfConversionError(f'[任务轮询]失败，错误信息: {extract_result["err_msg"]}')

            else:
                self.log_step(
                    step_name="任务轮询",
                    message=f"处理中...... 已耗时: {int(elapsed_time)}s，状态：{data_state}，batch_id:{batch_id}"
                )
                time.sleep(poll_interval)


    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        步骤1：校验PDF文件路径和输出目录
        核心职责：参数非空校验 | 路径转换 | PDF文件有效性校验 | 输出目录自动创建
        返回：合法的PDF文件Path对象、输出目录Path对象
        异常：StateFieldError(参数缺失)、FileNotFoundError(文件无效)
        """

        # 1.参数的非空校验
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", message="PDF路径不能为空", expected_type=str)

        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", message="输出路径不能为空", expected_type=str)

        # 2. 转换为Path对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 3.pdf是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(f"文件{pdf_path_obj.name}不存在")

        # 4.输出目录不存在则创建
        if not file_dir_obj.exists():
            self.logger.info(f"输出目录{file_dir_obj.absolute()}不存在，开始创建...")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj


if __name__ == "__main__":
    # 激活日志
    setup_logging()

    init_state = {
        "pdf_path": r"D:\doc\联想海豚用户手册.pdf",
        "file_dir": r"D:\output"
    }
    node_pdf_to_md = NodePDFToMD()
    # 使用 对象() 的方式相当于调用了 对象的__call__()
    result = node_pdf_to_md(init_state)

    # 将返回的图状态进行json序列化
    json_state = json.dumps(result, ensure_ascii=False, indent=4)
    # 输出
    logging.getLogger().info(json_state)

'''
检查点

1. 用自己的话讲：为什么不能一个 POST 请求把 PDF 发过去然后等结果，非要 "申请 URL → PUT → 轮询" 三步？
2. 轮询时收到 HTTP 200，能不能直接当成功？还要检查什么？
3. 如果同一份 PDF 连续导入两次，代码靠哪一行保证不会因 "解压目录已存在" 报错？
4. 节点最后 `return {"md_path": ...}`（只返回一个字段）LangGraph 会不会把 state 里其他字段弄丢？为什么？


来，逐题批改——这几题恰恰戳中了两个最重要的机制，错得有价值，讲透它。

## 批改

**第 1 题：只答到了皮毛（60 分）**

"能看到进度"只是**副产品**，根本原因是两个：

1. **PDF 解析太慢**（几十秒到几分钟）。如果一个 HTTP 请求发过去一直挂着等结果，连接早就超时了，服务器也扛不住成千上万个傻等的连接。所以必须**异步**：提交任务立刻拿到 `batch_id` 就断开，之后定期来问。
先必须访问API，然后api给你一个地址，你去访问OSS，把文件传给他
┌─────────────────────────────────────────────┐
│              MinerU 系统                      │
│                                               │
│   ┌──────────────┐      ┌──────────────────┐ │
│   │  API 服务器   │      │   对象存储(OSS)   │ │
│   │  (接收请求)   │      │  (存文件的地方)   │ │
│   └──────┬───────┘      └────────┬─────────┘ │
│          │                       │            │
│          │                       │            │
│   ┌──────┴───────────────────────┴─────────┐ │
│   │         解析引擎（干重活的工人）          │ │
│   └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
**第 2 题：答偏方向了**

"检查输出路径"是 step 1 的事，跟这里的轮询响应没关系。HTTP 200 之后还要过**两道**：
就是说要看业务层返回的json里面body字段值是什么？HTTP 200 → 看 code 是不是 0 → 看 state 是不是 done。三层都过关，才算真正成功。
| 层级 | 检查什么 | 反例 |
|---|---|---|
| HTTP 层 | `status_code == 200`（网络通、服务器回话了） | 500 服务器崩 |
| 业务层 | JSON 里 `code == 0` **且** `state == "done"` | 200 但 body 写着"Token 无效" |

记住这句话：**HTTP 200 只代表"电话打通了"，不代表"对方答应了你"。**

**第 3 题：补上答案**

```python
if extract_target_dir.exists():
    shutil.rmtree(extract_target_dir)   # ← 就是这行
```

解压前先把旧目录整个删掉再重建，所以同一份 PDF 导一百次，环境永远是干净的，不会因为"目录已存在/文件被占用"出错。这个性质叫**幂等**——同样的操作重复执行，结果一致，是后端工程的重要追求。

**第 4 题：结论对，理由没到点子上**

不是因为"代码已经存了"。真正的机制是 **LangGraph 会把节点返回值"合并"进总 state，而不是"替换"**：

> 你不用交回整张施工单，只交一张**变更条**，上面写着"md_path 改成 xx"。资料员只把这一栏贴改掉，单子上其他栏目（pdf_path、file_title…）原封不动。

- 你现在的写法：整张单子改完交回（`state["xx"]=...; return state`）
- 注释掉的写法：只交变更条（`return {"md_path": ...}`）

LangGraph 两种都认，第二种更"函数式"（不改输入、只产出变更），检索流水线会大量用。**关键词：状态合并（merge），不是覆盖。**

---

这 4 题对应的两个核心思想——**异步轮询**和**状态增量合并**——后面会反复出现，现在卡住比面试时卡住强。

都顺过来了的话，回个"继续"，我们进入 **node_md_img（多模态图片理解）**：MinerU 的 zip 里其实还躺着一堆图片，这一节点要调视觉大模型"看图说话"，并且解决一个很实际的工程问题——**API 有频率限制，15 张图不能瞬间全发出去，怎么做限流？**（这是简历上能写的第二个设计点）

'''