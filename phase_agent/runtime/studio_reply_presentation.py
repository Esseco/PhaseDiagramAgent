"""Present a Studio turn as status and advice without changing scientific decisions."""

from phase_agent.runtime.chat_state_presentation import brief_chat_state


def format_turn_reply(reply, state, *, configuring=False):
    text = str(reply or "").strip()
    if not configuring and state.get("pending_execution_policies"):
        from phase_agent.runtime.plan_queries import pending_plan_reply

        # A rejected new proposal must not hide the existing approval object.
        if "已保存方案（编号：" not in text and ("建议：" not in text or "本轮未执行" in text):
            text += "\n" + pending_plan_reply(state)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    current = next((line for line in lines if line.startswith(("当前：", "当前阶段："))), None)
    if current:
        current = current.replace("当前阶段：", "当前状态：", 1).replace("当前：", "当前状态：", 1)
    else:
        current = brief_chat_state(state, configuring=configuring).replace(
            "当前：", "当前状态：", 1
        )
    if (
        not configuring
        and not state.get("tasks")
        and not state.get("pending_execution_policies")
        and not state.get("remote_finetune_jobs")
        and "无待运行任务" in current
    ):
        current = "当前状态：尚未开始搜索。"
    body = []
    for line in lines:
        if line.startswith(
            (
                "当前：",
                "当前阶段：",
                "当前项目状态：",
                "完整记录：",
                "详细记录：",
                "累计任务：",
                "剩余相对预算",
                "依据：",
                "仅查看已记录状态",
                "本次仅查看状态",
            )
        ):
            continue
        if line.startswith(("下一步：", "下一步方案：", "Agent建议：")):
            line = "建议：" + line.split("：", 1)[1]
        body.append(line)
    if not any(line.startswith("建议：") for line in body):
        if "暂不可批准：" in text:
            body.append("建议：先核对原任务与执行回执，当前方案暂不可批准。")
        elif (
            state.get("generation_history")
            and not state.get("tasks")
            and not any(word in text for word in ("本轮失败", "本轮未执行", "未能", "错误", "缺少"))
        ):
            body.append(
                "建议：是否现在审查已生成结构并准备 Relax 任务方案？确认后先展示方案，批准后生成输入与提交脚本。"
            )
        elif any(word in text for word in ("失败", "未执行", "缺少", "未能", "错误")):
            body.append("建议：先处理上述问题，再继续生成方案。")
        elif configuring:
            body.insert(0, "建议：按以下要求完成并确认初始配置。")
        elif state.get("pending_execution_policies"):
            body.insert(0, "建议：核对下方方案，回复‘同意’批准或提出修改。")
        else:
            body.append("建议：回复‘继续’，由 Agent 检查结果并提出下一步方案。")
    if not configuring and state.get("pending_execution_policies"):
        from phase_agent.runtime.plan_queries import pending_plan_reply

        pending_details = pending_plan_reply(state)
        if "暂不可批准：" in pending_details:
            # Persisted execution facts override any model-generated approval advice.
            body = [line for line in body if not line.startswith("建议：")]
            body.insert(
                0,
                "建议：先核对原任务与执行回执，当前方案暂不可批准。",
            )
            if "已保存方案（编号：" not in text:
                body.append(pending_details)
    # Only explicit output facts become results; never infer completion from a plan.
    result_prefixes = ("结果：", "已生成 ", "已完成：", "本轮失败：", "本轮未执行：")
    results = [
        line if line.startswith("结果：") else "结果：" + line
        for line in body
        if line.startswith(result_prefixes)
    ]
    advice = [line.replace("建议：", "下一步：", 1) for line in body if line.startswith("建议：")]
    details = [line for line in body if not line.startswith(("建议：", *result_prefixes))]
    return "\n".join([*results, current, *advice, *details])
