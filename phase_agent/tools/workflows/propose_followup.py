"""Prepare one next approval after a local milestone; never carry permission."""

from copy import deepcopy


def propose_after_generation(frame, result, *, event_loop):
    if result.get("module_preparation"):
        return result
    if frame.get("execution_mode") != "interactive" or not callable(frame.get("agent_client")):
        return result
    if result.get("status") != "completed":
        return result
    state = result.get("state") or {}
    if state.get("pending_execution_policies") or state.get("pending_tasks"):
        return result
    refresh = result.get("completed_refresh") or next(
        (
            event["completed_refresh"]
            for event in reversed(result.get("events") or [])
            if event.get("completed_refresh")
        ),
        None,
    )
    if refresh:
        from phase_agent.tools.state.task_waiting import active_pending_tasks

        if (
            refresh.get("model_version") != state.get("active_model_version")
            or (state.get("model_refresh") or {}).get("status") != "completed"
            or active_pending_tasks(state.get("tasks") or [])
        ):
            return result
        result = {**result, "completed_refresh": deepcopy(refresh)}
    completed = next(
        (
            event
            for event in reversed(result.get("events") or [])
            if (event.get("execution") or {}).get("status") == "completed"
            and (event.get("action") or event.get("final_action") or {}).get("tool")
            in {
                "generate_branches",
                "select_candidates",
                "adjust_strategy",
                "reevaluate_candidates",
            }
        ),
        None,
    )
    if completed is None and not refresh:
        return result
    completed = completed or {}
    tool = (
        "model_refresh"
        if refresh
        else (completed.get("action") or completed.get("final_action") or {})["tool"]
    )
    payload = (completed.get("execution") or {}).get("result") or {}
    summary = payload.get("summary") or {}
    if tool == "generate_branches" and not summary.get("registered_structures"):
        return result
    invocation = (
        str(frame.get("invocation_id") or completed.get("record_id") or "completed")
        + ":next-proposal"
    )
    context = {
        **(frame.get("context") or {}),
        "unified_dialogue": True,
        "user_message": f"上一项 {tool} 已实际完成。审查最新产出、合法性、去重与失败记录，在本轮主动提出下一步的一项具体可执行方案供人工审批。Branch方案获批后，生成branch并在同一批准内准备对应Relax+MLIP-MC输入；只停止在输入文件已生成和可人工提交的状态，不提交超算。若存在真实缺失条件，具体说明。禁止执行或提交下一项任务。",
    }
    if tool == "generate_branches":
        context["user_message"] += (
            "为已登记branch准备Relax+MLIP-MC输入时，使用已注册operation=prepare_relax_mc_batch，"
            "对应tool=prepare_local_batch_files、parameters.mode=relax_mc_inputs。"
            "这只形成一份新的待审方案，不继承上一项的批准，也不执行输入准备或提交。"
        )
    if refresh:
        context["user_message"] = (
            f"新模型结构刷新及相图核对已完成：已刷新 {refresh['refreshed']} 个独立结构，"
            f"{refresh['deferred']} 个未刷新，当前相图 {refresh['phase_diagram_entry_count']} 条。"
            "结合最新相图、覆盖缺口和预算，在本轮提出生成当前模型新branch的具体方案供人工审批。"
            "审批内容须同时展示区域、预计输入范围和不提交超算的边界；生成后回报实际数量、分布和输入包。"
            "不得把Relax/MC提交或复用旧branch混入本次批准。"
            "部分刷新不能作为全面覆盖或全局收敛证据。"
            "展示完成内容、下一步目标、范围、工作量、预算与审批边界。"
            "若存在真实阻塞或无需新增任务，具体说明原因与下一步。"
            "禁止执行或提交下一项任务。"
        )
    try:
        following = event_loop(
            state,
            frame["config_session"],
            registry=frame["effective_registry"],
            agent_client=frame["agent_client"],
            context=context,
            execution_mode="interactive",
            human_feedback=None,
            replay_record=None,
            max_steps=1,
            invocation_id=invocation,
            state_path=frame.get("state_path") or frame["effective_config"].get("state_path"),
            approval_directory=frame.get("approval_directory")
            or frame["effective_config"].get("approval_directory"),
        )
    except Exception as error:
        # Generation remains completed; failure to prepare the next approval
        # must not turn a successful scientific action into a retry request.
        return {**result, "followup_error": f"{type(error).__name__}: {error}"}
    if following.get("status") != "awaiting_approval":
        return {
            **result,
            "followup_note": following.get("reason")
            or following.get("answer")
            or "下一步方案尚未通过校验。",
            "state": following.get("state") or state,
        }
    combined = {**result, **following}
    combined["events"] = [*(result.get("events") or []), *(following.get("events") or [])]
    combined["steps_executed"] = result.get("steps_executed", 1)
    combined["completed_action"] = {"tool": tool, "summary": deepcopy(summary)}
    if tool == "generate_branches":
        combined["completed_generation"] = deepcopy(summary)
    combined["submitted"] = result.get("submitted", False)
    return combined
