"""Build one approved workflow call from a user turn; no HTTP dependencies."""

from phase_agent.tools.step_runner.file_protocol import read_json


def run_workflow_turn(
    handler,
    invocation_id,
    human_feedback,
    user_message,
    *,
    approve_config_migration=False,
    dft_recovery_decision=None,
    explicit_branch_request=False,
):
    workflow = handler.workflow
    if workflow is None:
        from phase_agent.runtime.main import run_workflow

        workflow = run_workflow
    kwargs = {
        key: value
        for key, value in handler.workflow_kwargs.items()
        if key
        not in {
            "state",
            "execution_mode",
            "human_feedback",
            "replay_record",
            "max_steps",
            "invocation_id",
            "state_path",
            "config_session_path",
        }
    }
    kwargs["user_message"] = user_message
    if getattr(handler, "unified_dialogue", False):
        kwargs["unified_dialogue"] = True
    if dft_recovery_decision is not None:
        kwargs["dft_recovery_decision"] = dft_recovery_decision
    if approve_config_migration:
        # This approval only authorizes a safe, versioned config migration;
        # it never approves a scientific action or remote submission.
        kwargs["approve_budget_extension"] = True
    base_agent_client = kwargs.get("agent_client")
    from phase_agent.decisions.agent.resolve_explicit_generation_request import (
        resolve_explicit_generation_request,
    )
    from phase_agent.tools.local.rebuild_relax_inputs import (
        is_relax_rebuild_request,
        plan_relax_rebuild,
    )

    if (
        callable(base_agent_client)
        or explicit_branch_request
        or is_relax_rebuild_request(user_message)
    ):

        def user_contextualized_agent(payload):
            if not getattr(handler, "unified_dialogue", False) and is_relax_rebuild_request(
                user_message
            ):
                plan = plan_relax_rebuild(
                    read_json(handler.state_path, {}) or {},
                    kwargs["run_config"]["upload_batches_directory"],
                )
                return {
                    "tool": "prepare_local_batch_files",
                    "task_key": f"rebuild-relax:{invocation_id}",
                    "target_ids": [],
                    "budget": 0.0,
                    "parameters": {
                        "mode": "relax_inputs",
                        "rebuild_inputs": True,
                        "cleanup_plan": plan,
                    },
                    "reason": f"请确认旧批次尚未在超算提交；批准后删除 {len(plan['directories'])} 个旧输入批次并重建，保留原结构、任务编号和预算。",
                    "expected_purpose": "重建Relax输入，每个作业最多100个结构；不提交计算。",
                }
            direct_action = (
                None
                if callable(base_agent_client)
                else resolve_explicit_generation_request(
                    user_message,
                    allowed_tools=payload.get("allowed_tools") or [],
                    state=payload.get("state") or {},
                )
            )
            if direct_action is not None:
                direct_action["task_key"] = f"generate_branches:user-request:{invocation_id}"
                return direct_action
            if not callable(base_agent_client):
                raise ValueError("DeepSeek Agent 未配置")
            from time import monotonic
            from phase_agent.runtime.turn_process import process_event, record_timing

            started = monotonic()
            outcome = "失败"
            try:
                response = base_agent_client(
                    {
                        **payload,
                        "user_instruction": user_message,
                        "conversation_facts": getattr(handler, "turn_context", {}),
                    }
                )
                outcome = "已返回"
                return response
            finally:
                elapsed = monotonic() - started
                record_timing(
                    "search_model",
                    elapsed,
                    category="cache"
                    if getattr(base_agent_client, "last_call_reused", False)
                    else "model",
                    status=outcome,
                    mode=payload.get("mode"),
                )
                process_event(
                    "模型调用耗时",
                    {
                        "mode": payload.get("mode"),
                        "耗时秒": round(elapsed, 3),
                        "结果": outcome,
                    },
                )

        kwargs["agent_client"] = user_contextualized_agent
        user_contextualized_agent.decision_backend = handler.decision_backend
    return workflow(
        **kwargs,
        execution_mode="interactive",
        human_feedback=human_feedback,
        max_steps=1,
        invocation_id=invocation_id,
        state_path=str(handler.state_path),
    )
