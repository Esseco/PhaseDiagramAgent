"""Present a Studio turn as status and advice without changing scientific decisions."""

from phase_agent.runtime.chat_state_presentation import brief_chat_state


def format_turn_reply(reply, state, *, configuring=False):
    text = str(reply or "").strip()
    if not configuring and not state.get("pending_execution_policies"):
        from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

        delivery = waiting_file_lines(state)
        if delivery and any(marker in text for marker in ("prepare_local_batch_files", "本轮提交文件夹：", "Relax+MC 上传包：", "上传 inputs 包")) and "本轮失败" not in text and "本轮未执行" not in text:
            return "\n".join([brief_chat_state(state).replace("当前：", "当前状态：", 1), *delivery])
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
        if "本轮未执行：" in text or "本轮失败：" in text:
            body.append("建议：先处理上述问题。")
        elif "暂不可批准：" in text:
            body.append("建议：先核对原任务与执行回执，当前方案暂不可批准。")
        elif configuring:
            body.insert(
                0,
                "建议：核对并确认本次配置草稿。"
                if state.get("confirmed_config_version")
                else "建议：按以下要求完成并确认初始配置。",
            )
        elif state.get("pending_execution_policies"):
            body.insert(0, "建议：核对下方方案，回复‘同意’批准或提出修改。")
        elif not state.get("pending_mc_regeneration"):
            from phase_agent.tools.state.task_waiting import active_pending_tasks
            from phase_agent.runtime.chat_state_presentation import waiting_training_job

            if active_pending_tasks(state.get("tasks") or []) or waiting_training_job(state):
                from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

                delivery = waiting_file_lines(state)
                body.extend(delivery or ["建议：等待当前任务完成，回传结果后告知已回传。"])
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
