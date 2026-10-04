import copy

# 原始状态，chunks 里先放两段
original = {
    "task_id": "t1",
    "chunks": ["第一段", "第二段"]
}

assigned = original              # 直接赋值
shallow = copy.copy(original)    # 浅拷贝
deep = copy.deepcopy(original)   # 深拷贝

print("=== 1. 看 chunks 的对象标识 id（相同=同一个列表）===")
print("original:", id(original["chunks"]))
print("assigned:", id(assigned["chunks"]))
print("shallow :", id(shallow["chunks"]))
print("deep    :", id(deep["chunks"]))

print()
print("=== 2. 只往 original.chunks 追加一段 ===")
original["chunks"].append("第三段")
print("assigned:", assigned["chunks"], "  <- 直接赋值，跟着变")
print("shallow :", shallow["chunks"], "  <- 浅拷贝，内层列表共享，跟着变")
print("deep    :", deep["chunks"], "  <- 深拷贝，完全独立，不变")

print()
print("=== 3. 改 original 的外层字段 task_id ===")
original["task_id"] = "changed"
print(assigned["task_id"])
print("shallow task_id:", shallow["task_id"], "  <- 浅拷贝的外层字典独立，不变")
print("deep task_id   :", deep["task_id"], "  <- 深拷贝也不变")

print()
print("=== 4. 用 id 验证：替换 vs 原地改 ===")
print("task_id 改前 id:", id(original["task_id"]))
original["task_id"] = "again"
print("task_id 改后 id:", id(original["task_id"]), " <- 重新绑定，id 变了（新对象）")

print("chunks 改前 id:", id(original["chunks"]))
original["chunks"].append("第四段")
print("chunks 改后 id:", id(original["chunks"]), " <- 原地修改，id 不变（同一对象）")
