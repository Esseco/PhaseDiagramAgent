"""Present a Studio turn as status and advice without changing scientific decisions."""

from phase_agent.runtime.chat_state_presentation import brief_chat_state
from phase_agent.runtime.review_file_presentation import inventory_location_lines


def format_turn_reply(reply, state, *, configuring=False):
    text = str(reply or "").strip()
    pending_cache = None

    def saved_plan_details():
        nonlocal pending_cache
        if pending_cache is None:
            from phase_agent.runtime.plan_queries import pending_plan_reply

            pending_cache = pending_plan_reply(state)
        return pending_cache

    model = state.get("active_model_version")
    error_batches = [
        batch
        for batch in (state.get("relax_mc_batches") or {}).values()
        if batch.get("return_error")
        and batch.get("status") != "cancelled"
        and (not model or batch.get("model_version") in (None, model))
    ]
    if not configuring and error_batches:
        from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

        lines = [
            "当前状态：回传核对受阻，本地待完成记录不能据此判定为尚未计算。",
            *waiting_file_lines(state),
        ]
        for batch in error_batches:
            error_line = "回传核对失败：" + str(batch["return_error"])
            if error_line not in lines:
                lines.append(error_line)
        if state.get("pending_execution_policies"):
            lines.append(saved_plan_details())
        return "\n".join(lines)
    if not configuring:
        from phase_agent.analysis.state.batch_return_facts import batch_return_facts

        returns = batch_return_facts(state)
        stale_claim = any(
            marker in text
            for marker in (
                "结果包仍未登记",
                "结果文件没有进入项目库",
                "请在接收端修复传输清单",
                "尚无任何 Relax/MC/DFT 结果回收",
            )
        )
        if returns["all_registered_returns_verified"] and stale_claim:
            counts = returns["completed_tasks_by_stage"]
            result = (
                f"结果：已有回传包已核验并入账，共 {returns['reported_returned_tasks']} 项；"
                f"Relax 已完成 {counts.get('relax_and_feature', 0)} 项，"
                f"MC 已完成 {counts.get('deep_search', 0)} 项。"
            )
            mlip = returns["phase_diagrams"].get("mlip") or {}
            if mlip.get("status") == "completed":
                result += f"MLIP 相图已有 {mlip['entry_count']} 条有效条目。"
            lines = [
                result,
                "说明：旧的大小限制报错已不代表当前回收状态，无需为该旧错误重新上传、回传或重算。",
            ]
            if state.get("pending_execution_policies"):
                lines.append(saved_plan_details())
            else:
                lines.append("下一步：可基于已回收结果评估后续方案；新动作仍需按原规则审阅。")
            return "\n".join(lines)
    if not configuring and not state.get("pending_execution_policies"):
        from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

        delivery = waiting_file_lines(state)
        if (
            delivery
            and any(
                marker in text
                for marker in (
                    "prepare_local_batch_files",
                    "本轮提交文件夹：",
                    "Relax+MC 上传包：",
                    "上传 inputs 包",
                )
            )
            and "本轮失败" not in text
            and "本轮未执行" not in text
        ):
            return "\n".join(
                [
                    brief_chat_state(state).replace("当前：", "当前状态：", 1),
                    *delivery,
                    *inventory_location_lines(text),
                ]
            )
    if not configuring and state.get("pending_execution_policies"):
        # A rejected new proposal must not hide the existing approval object.
        if (
            "确认范围：" not in text
            and "已保存方案（编号：" not in text
            and ("建议：" not in text or "本轮未执行" in text)
        ):
            text = "\n".join(
                line
                for line in text.splitlines()
                if not (
                    line.startswith(("下一步：", "建议："))
                    and "回复" in line
                    and "同意" in line
                    and not any(word in line for word in ("不要", "不能", "暂不可"))
                )
            )
            text += "\n" + saved_plan_details()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    current = next(
        (line for line in lines if line.startswith(("当前：", "当前阶段：", "当前状态："))), None
    )
    if current:
        current = current.replace("当前阶段：", "当前状态：", 1).replace("当前：", "当前状态：", 1)
    else:
        current = brief_chat_state(state, configuring=configuring).replace(
            "当前：", "当前状态：", 1
        )
    reported_current = current
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
        if line.startswith(("当前：", "当前阶段：", "当前状态：")):
            normalized = "当前状态：" + line.split("：", 1)[1]
            if normalized != reported_current:
                body.append("需注意：" + line.split("：", 1)[1])
            continue
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
                from phase_agent.runtime.waiting_files_presentation import (
                    waiting_file_lines,
                )

                delivery = waiting_file_lines(state)
                body.extend(delivery or ["建议：等待当前任务完成，回传结果后告知已回传。"])
    if not configuring and state.get("pending_execution_policies"):
        pending_details = saved_plan_details()
        if "暂不可批准：" in pending_details:
            # Persisted execution facts override any model-generated approval advice.
            body = [line for line in body if not line.startswith("建议：")]
            body.insert(
                0,
                next(
                    (
                        line.replace("下一步：", "建议：", 1)
                        for line in pending_details.splitlines()
                        if line.startswith("下一步：")
                    ),
                    "建议：先核对原任务与执行回执，当前方案暂不可批准。",
                ),
            )
            if pending_details not in text:
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
    return "\n".join(dict.fromkeys([*results, current, *advice, *details]))
