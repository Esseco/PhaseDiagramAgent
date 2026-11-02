"""After activation, propose refresh inputs through the existing workflow gate."""


def propose_refresh_after_activation(handler, result, previous_version):
    state = result["state"]
    active = state.get("active_model_version")
    refresh = state.get("model_refresh") or {}
    if (
        not active
        or active == previous_version
        or refresh.get("new_model_version") != active
        or refresh.get("status") != "approval_required"
    ):
        return None
    from phase_agent.graphs.invocation_context import new_invocation
    from phase_agent.graphs.dialogue.support import bind_presented_proposal
    from phase_agent.tools.step_runner.file_protocol import read_json
    from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply

    try:
        following = handler._run(
            new_invocation(),
            None,
            "新模型已按用户批准切换。请通过既有结构刷新入口，展示生成刷新文件的具体目标、范围及预算，"
            "并询问是否批准生成刷新文件。这里只提出方案，不继承模型切换的批准，不生成文件或提交计算。"
            "刷新结果回传分析完成后，再提出 branch 生成方案并接回既有搜索流程。",
        )
        reply = format_workflow_reply(following, handler.state_path)
        if following.get("status") == "awaiting_approval":
            bind_presented_proposal(handler, read_json(handler.state_path, {}) or {})
    except Exception as error:
        # Activation is already persisted: never turn a preview failure into a
        # failed activation that a caller could retry or roll back.
        return {
            "status": "refresh_proposal_failed",
            "activation_status": "activated",
            "reason": f"结果：新模型已切换。生成刷新文件方案失败：{type(error).__name__}: {error}。请修复后重新提出刷新方案；不要重复切换模型。",
        }
    return {
        "status": following.get("status"),
        "activation_status": "activated",
        "result": following,
        "reason": "结果：新模型已切换。\n" + reply,
    }
