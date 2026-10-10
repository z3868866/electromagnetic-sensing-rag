def check_confirmation(state, align_result):
    confirmed = align_result.get("confirmed", [])
    options = align_result.get("options", [])

    # 分支A：已确认主体，继续检索
    if confirmed:
        state["item_names"] = confirmed
        state["answer"] = ""
        return state, "→ 进入多路并行检索"

    # 分支B：有候选，生成反问，中断流程
    if options:
        options_str = "、".join(options[:3])
        state["answer"] = f"您是想问以下哪篇文献：{options_str}？请明确具体名称。"
        state["item_names"] = []
        return state, "→ 直接输出反问（中断检索）"

    # 分支C：无结果，拒识，中断流程
    state["answer"] = "抱歉，未找到相关文献，请提供准确的文献或方法名称。"
    state["item_names"] = []
    return state, "→ 直接输出拒识（中断检索）"


if __name__ == "__main__":
    cases = [
        ("分支A 已确认", {"confirmed": ["基于深度卷积神经网络的协作频谱感知方法"], "options": []}),
        ("分支B 有候选", {"confirmed": [], "options": ["候选文献A", "候选文献B"]}),
        ("分支C 无结果", {"confirmed": [], "options": []}),
    ]
    for label, align_result in cases:
        state = {"original_query": "测试问题", "item_names": [], "answer": ""}
        state, route = check_confirmation(state, align_result)
        print("=" * 52)
        print(label, route)
        print("  item_names:", state["item_names"])
        print("  answer:", state["answer"])
