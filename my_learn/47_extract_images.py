import json
import re

IMG_EXT = ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg')
PAT = re.compile(r'!\[.*?\]\((.*?)\)')


def extract_images(docs):
    images, seen = [], set()
    for doc in docs:
        url = (doc.get("url") or "").strip()
        if url and url.lower().endswith(IMG_EXT) and url not in seen:
            seen.add(url); images.append(url)
        for m in PAT.findall(doc.get("content") or ""):
            m = m.strip()
            if m and m not in seen:
                seen.add(m); images.append(m)
    return images


if __name__ == "__main__":
    p = r"D:\RAG\YUANLAI\my_learn\output\基于深度卷积神经网络的协作频谱感知方法\chunks_with_vector.json"
    docs = json.load(open(p, encoding="utf-8"))
    imgs = extract_images(docs)
    print("切片数：", len(docs), "| 提取真实图片：", len(imgs))
    for u in imgs:
        print(u)
