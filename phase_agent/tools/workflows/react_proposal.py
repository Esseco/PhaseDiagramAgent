"""One online decision, then immutable proposal validation and human approval."""

from copy import deepcopy
from phase_agent.tools.policy.validate_tool_action import validate_tool_action
from phase_agent.tools.state.state_manager import agent_state_summary


def prepare_react_proposal(frame):
    import importlib

    tools = importlib.import_module("phase_agent.tools.workflows.run_tool_step")
    current = frame["current"]
    context = frame.get("context") or {}
    key = frame["invocation_id"] or "__single_interactive_action__"
    stored = current["pending_execution_policies"].get(key)
    feedback = frame.get("human_feedback")
    decision = feedback.get("decision") if isinstance(feedback, dict) else feedback
    if stored and decision == "comment":
        from phase_agent.tools.workflows.training_plan_revision import revise_training_plan

        current, response = revise_training_plan(
            current,
            stored,
            str(feedback.get("comment") or context.get("user_message") or ""),
            frame["agent_client"],
            context.get("state_path"),
        )
        if response:
            return response
    summary = agent_state_summary(current)
    summary["user_message"] = str(context.get("user_message") or "")
    if stored:
        summary.setdefault("decision_context", {})["pending_proposal"] = deepcopy(
            stored["agent_proposal"]
        )
    if stored and decision in {"approve", "reject"}:
        proposal = deepcopy(stored["agent_proposal"])
        if decision == "approve":
            error = stale_proposal_error(proposal, current, context, frame["config"])
            if error:
                return {"status": "rejected", "state": current, "reason": error, "submitted": False}
        record_id = stored["record_id"]
    else:
        from phase_agent.tools.workflows.initial_tool_proposal import select_initial_proposal

        selected = select_initial_proposal(
            current=current,
            mode="interactive",
            config=frame["config"],
            context=context,
            decision_state=summary,
            registry=frame["registry"],
            agent_client=frame["agent_client"],
            invocation_id=key,
            propose=tools.propose_agent_tool_action,
            revise=tools.revise_tool_proposal,
            prepare_debug=tools._prepare_debug_relax_screen_action,
            is_verified_second=tools._is_verified_second_mc_action,
            mc_block=tools._mc_continuation_block,
            model_failed=tools._model_failure_action,
            record_id_factory=tools._record_id,
        )
        if not selected.get("_proposal_selected"):
            return selected  # Answer/config request/failed proposal preserves previous approval.
        current, summary = selected["current"], selected["decision_state"]
        proposal, record_id = selected["proposal"], selected["record_id"]
        if stored:
            stored = deepcopy(stored)
            stored["revision"] = int(stored.get("revision", 0)) + 1
            stored.setdefault("feedback_history", []).append(
                {
                    "comment": summary["user_message"],
                    "revision_status": "model_reproposed",
                    "prior_proposal": deepcopy(stored["agent_proposal"]),
                }
            )
        feedback = None  # A new or modified proposal is never already approved.
    if decision != "reject":
        validation = validate_tool_action(
            proposal["raw_action"], current, frame["session"], frame["registry"]
        )
        if not validation["valid"]:
            from phase_agent.runtime.turn_process import process_event

            process_event("执行前约束校验", validation)
            return {
                "status": "rejected",
                "state": current,
                "validation": validation,
                "reason": "方案未通过执行约束：" + "; ".join(validation["errors"]),
                "submitted": False,
            }
    return {
        **frame,
        "_graph_prepared": True,
        "current": current,
        "mode": "interactive",
        "pending_key": key,
        "stored": stored,
        "proposal": proposal,
        "record_id": record_id,
        "decision_state": summary,
        "human_feedback": feedback,
    }


def stale_proposal_error(proposal, current, context, config):
    action = proposal.get("raw_action") or {}
    from phase_agent.tools.state.approved_direction import direction_action_errors

    if direction_action_errors(action, current):
        return "科研方向已变化；请生成并审核新方案，旧批准不能用于新动作。"
    from phase_agent.tools.state.training_pending_validation import training_pending_validation

    if action.get("tool") == "update_mlip" and training_pending_validation(current):
        return "训练结果已回收；旧训练方案不可再次执行。"
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment

    assessment = post_dft_assessment(current, context.get("effective_config") or config)
    if action.get("_post_dft_scope_key") and (
        not assessment or action["_post_dft_scope_key"] != assessment.get("scope_key")
    ):
        return "DFT分析依据已变化；请重新生成并审核方案。"
    return None
