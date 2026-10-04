import os
import sys
import re

sys.path.insert(0, r"D:\RAG\YUANLAI")

from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from pymilvus import DataType, Function, FunctionType
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from utils.embedding_utils import get_online_embedder
from utils.milvus_utils import get_milvus_client, online_hybrid_search

MD = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\基于深度卷积神经网络的协作频谱感知方法_new.md"
FILE_TITLE = "基于深度卷积神经网络的协作频谱感知方法"
COLL = "kb_pc_test"
DIM = 1024
QUESTION = "DCNN 是如何缓解梯度消失问题的？"
DROP_AT_END = True
WHITELIST = {"参考文献", "摘要", "关键词"}

with open(MD, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

title_line_re = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
num_re = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")
exclude_re = re.compile(r"^(表|图|Table|Figure|Fig\.?)\s*\d")
fence_open_re = re.compile(r"^(`{3,}|~{3,})(.*)$")
fence_close_re = re.compile(r"^(`{3,}|~{3,})$")


def is_real_title(text, level):
    if exclude_re.match(text):
        return False
    if level == 1:
        return True
    if num_re.match(text):
        return True
    return text.replace(" ", "") in WHITELIST


# 1. 解析父块
in_code, start_marker = False, None
parents, cur_title, cur_body = [], None, []

def flush():
    if cur_title is not None:
        parents.append({"parent_title": cur_title, "content": "\n".join(cur_body).strip()})

for line in lines:
    s = line.strip()
    mo = fence_open_re.match(s)
    mc = fence_close_re.match(s)
    if not in_code and mo:
        in_code, start_marker = True, mo.group(1)
        if cur_title: cur_body.append(line)
        continue
    if in_code and mc:
        if mc.group(1)[0] == start_marker[0] and len(mc.group(1)) >= len(start_marker):
            in_code, start_marker = False, None
        if cur_title: cur_body.append(line)
        continue
    mt = title_line_re.match(line)
    if not in_code and mt and is_real_title(mt.group(2), len(mt.group(1))):
        flush(); cur_title, cur_body = s, []
    elif cur_title:
        cur_body.append(line)
flush()

for i, p in enumerate(parents):
    p["parent_uid"] = f"pc_{i}"

# 2. 父块切子块
sp = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80,
                                    separators=["\n\n", "\n", "。", "！", "？", " ", ""])
children = []
for p in parents:
    if not p["content"]:
        continue
    for j, t in enumerate(sp.split_text(p["content"])):
        t = t.strip()
        if t:
            children.append({"parent_uid": p["parent_uid"], "parent_title": p["parent_title"],
                             "content": t})
print(f"父块 {len(parents)}（非空 {sum(1 for p in parents if p['content'])}）| 子块 {len(children)}")

# 3. 子块向量化（每批10）
embedder = get_online_embedder()
texts = [c["content"] for c in children]
for i in range(0, len(texts), 10):
    vecs = embedder.embed_documents(texts[i:i+10])
    for k, v in enumerate(vecs):
        children[i+k]["dense_vector"] = v
print("子块向量化完成")

# 4. 建临时集合
client = get_milvus_client()
if client.has_collection(COLL):
    client.drop_collection(COLL)
schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
schema.add_field("chunk_id", DataType.INT64, is_primary=True, auto_id=True)
schema.add_field("content", DataType.VARCHAR, max_length=65535, enable_analyzer=True,
                 analyzer_params={"type": "chinese"})
schema.add_field("title", DataType.VARCHAR, max_length=65535)
schema.add_field("parent_title", DataType.VARCHAR, max_length=65535)
schema.add_field("part", DataType.INT8)
schema.add_field("file_title", DataType.VARCHAR, max_length=65535)
schema.add_field("item_name", DataType.VARCHAR, max_length=65535)
schema.add_field("chunk_level", DataType.VARCHAR, max_length=32)
schema.add_field("parent_uid", DataType.VARCHAR, max_length=64)
schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=DIM)
schema.add_function(Function(name="bm25_function", function_type=FunctionType.BM25,
                             input_field_names=["content"], output_field_names=["sparse_vector"]))
ip = client.prepare_index_params()
ip.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
ip.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
client.create_collection(collection_name=COLL, schema=schema, index_params=ip)
print("临时集合创建成功")

# 5. 插入父块（零向量占位）与子块
parent_rows = [{"content": p["content"], "title": p["parent_title"],
                "parent_title": p["parent_title"], "part": 0, "file_title": FILE_TITLE,
                "item_name": FILE_TITLE, "chunk_level": "parent", "parent_uid": p["parent_uid"],
                "dense_vector": [0.0]*DIM} for p in parents if p["content"]]
child_rows = [{"content": c["content"], "title": c["parent_title"],
               "parent_title": c["parent_title"], "part": 0, "file_title": FILE_TITLE,
               "item_name": FILE_TITLE, "chunk_level": "child", "parent_uid": c["parent_uid"],
               "dense_vector": c["dense_vector"]} for c in children]
client.insert(COLL, parent_rows)
client.insert(COLL, child_rows)
print(f"入库：父块 {len(parent_rows)} | 子块 {len(child_rows)}")

# 6. 只检索子块
res = online_hybrid_search(client, COLL, QUESTION, expr='chunk_level=="child"', limit=5,
                           output_fields=["parent_title", "parent_uid"])
hits = res[0]
print("\n=== 命中的子块（只检索 child）===")
hit_uids = []
for r, h in enumerate(hits, 1):
    e = h.get("entity", {})
    hit_uids.append(e.get("parent_uid"))
    print(f"  #{r} score={round(h.get('score',0),3)} | {e.get('parent_title','')[:30]}")

# 7. 回取父块
uniq = list(dict.fromkeys(hit_uids))
quoted = ",".join(f'"{u}"' for u in uniq)
fdata = client.query(COLL, filter=f'chunk_level=="parent" and parent_uid in [{quoted}]',
                     output_fields=["parent_title", "content"])
print("\n=== small-to-big 回取的父块 ===")
total = 0
for p in fdata:
    total += len(p.get("content", ""))
    print(f"  {p.get('parent_title','')[:40]} | {len(p.get('content',''))}字")
print(f"喂给 LLM 的父块上下文：{total} 字")

if DROP_AT_END:
    client.drop_collection(COLL)
    print("\n临时集合已清理（DROP_AT_END=True）")
