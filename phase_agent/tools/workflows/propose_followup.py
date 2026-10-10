"""Prepare one next approval after generation; never carry execution permission."""

from copy import deepcopy


def propose_after_generation(frame, result, *, event_loop):
    if frame.get("execution_mode") != "interactive" or not callable(frame.get("agent_client")):
        return result
    if result.get("status") != "completed":
        return result
    state = result.get("state") or {}
    if state.get("pending_execution_policies") or state.get("pending_tasks"):
        return result
    generated = next(
        (
            event
            for event in reversed(result.get("events") or [])
            if (event.get("execution") or {}).get("status") == "completed"
            and (event.get("action") or event.get("final_action") or {}).get("tool")
            == "generate_branches"
        ),
        None,
    )
    if generated is None:
        return result
    payload = (generated.get("execution") or {}).get("result") or {}
    summary = payload.get("summary") or {}
    if not summary.get("registered_structures"):
        return result
    invocation = (
        str(frame.get("invocation_id") or generated.get("record_id") or "generated")
        + ":next-proposal"
    )
    context = {
        **(frame.get("context") or {}),
        "unified_dialogue": True,
        "user_message": "结构生成已经完成。审查最新产出、合法性、去重与失败记录，主动提出下一步的一项具体方案供人工审批。尚无当前模型弛豫结果及有效相图时，只提出 Relax 输入和提交脚本准备方案；不要分配 MC 步数。禁止执行或提交下一项任务。",
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
    combined["completed_generation"] = deepcopy(summary)
    combined["submitted"] = result.get("submitted", False)
    return combined
