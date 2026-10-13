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
    if completed is None:
        return result
    tool = (completed.get("action") or completed.get("final_action") or {})["tool"]
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
        "user_message": f"上一项 {tool} 已实际完成。审查最新产出、合法性、去重与失败记录，在本轮主动提出下一步的一项具体可执行方案供人工审批，不询问是否开始分析或准备方案。新生成 Branch 后，诊断已登记候选并提出 Relax+MLIP-MC 封装输入准备方案（prepare_relax_mc_batch，relax_mc_inputs），展示范围与工作量供审批。封装内部 MC 使用已确认规则；不得另行分配同批 MC。若存在真实缺失条件，具体说明。禁止执行或提交下一项任务。",
    }
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
