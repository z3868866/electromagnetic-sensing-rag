import sys
sys.path.insert(0, r"D:\RAG\YUANLAI")
from dotenv import load_dotenv
load_dotenv(r"D:\RAG\YUANLAI\.env")

from processor.query_processor.nodes.node_item_name_confirm import NodeItemNameConfirm

node = NodeItemNameConfirm()
q = "基于深度卷积神经网络的协作频谱感知方法是如何缓解梯度消失的？"
history = []

extract = node._step_4_extract_info(q, history)
print("STEP4 extract:", extract)

names = extract.get("item_names")
if names:
    qr = node._step_5_vectorize_and_query(names)
    for item in qr:
        print("STEP5 extracted:", repr(item["extracted_name"]))
        for m in item["matches"]:
            print("    match:", repr(m["item_name"]), round(m["score"], 4))
    align = node._step_6_align_item_names(qr)
    print("STEP6 align:", align)
print("DEBUG DONE")
