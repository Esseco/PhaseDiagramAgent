"""QBC metrics -> agent/fixed decision -> validation -> reservation -> submission -> retrain check."""

from copy import deepcopy

from scientific_layer.qbc.build_candidate_metrics import build_qbc_candidate_metrics
from decision_layer.qbc_selection.check_mlip_retrain_trigger import check_mlip_retrain_trigger
from decision_layer.qbc_selection.create_qbc_fixed_decisions import create_qbc_fixed_decisions
from decision_layer.qbc_selection.decide_dft_actions import decide_dft_actions
from execution_layer.budget.reserve_dft_actions import reserve_dft_actions
from execution_layer.workflows.submit_dft_actions import submit_dft_actions
from execution_layer.workflows.validate_dft_agent_decisions import validate_dft_agent_decisions
from execution_layer.workflows.update_dft_action_results import update_dft_action_results
from execution_layer.policy.execution_policy import apply_execution_policy, build_agent_proposal
from execution_layer.policy.validate_tool_action import validate_tool_action
from execution_layer.budget.estimate_stage_cost import estimate_dft_cost
from execution_layer.state.state_manager import agent_state_summary, update_state_snapshot


def run_qbc_dft_decision_flow(
    candidates: list[dict],
    state: dict,
    *,
    config: dict,
    config_version: str,
    dft_parameters: dict,
    context=None,
    agent_client=None,
    qbc_evaluator=None,
    dft_submitter=None,
    invocation_id: str,
    execution_mode: str = "autonomous",
    human_feedback=None,
    replay_record=None,
    execution_session=None,
    tool_registry=None,
):
    current = deepcopy(state)
    inputs = context or {}
    for source, target in (("phase_diagram_state", "phase_diagrams"),
                           ("decision_memory", "decision_memory"), ("rewards", "rewards"),
                           ("action_records", "action_records"), ("training_coverage", "coverage")):
        if inputs.get(source) is not None:
            current[target] = deepcopy(inputs[source])
    current["budget_remaining"] = inputs.get("remaining_dft_budget", current.get("budget_remaining"))
    current["qbc_candidates"] = deepcopy(candidates)
    current = update_state_snapshot(current, config_version=config_version)
    decision_snapshot = agent_state_summary(current)
    invocations = current.setdefault("qbc_dft_invocations", {})
    if invocation_id in invocations:
        return {**deepcopy(invocations[invocation_id]), "state": current, "idempotent_replay": True}
    metrics = build_qbc_candidate_metrics(candidates, qbc_evaluator=qbc_evaluator)
    remaining = float((context or {}).get("remaining_dft_budget", 0))
    pending = current.setdefault("pending_qbc_execution_policies", {})
    stored = pending.get(invocation_id)
    if stored:
        proposal = deepcopy(stored["proposal"])
        agent_proposal = deepcopy(stored["agent_proposal"])
    elif config.get("mode") == "qbc_fixed":
        proposal = create_qbc_fixed_decisions(candidates, config=config, remaining_dft_budget=remaining)
    elif config.get("mode") in {"agent", "agent_qbc"}:
        proposal = decide_dft_actions(metrics["metrics"], decision_snapshot, agent_client=agent_client)
    else:
        return {"status": "not_configured", "error": "unknown DFT decision mode", "state": current}
    if not stored and proposal.get("status") != "completed":
        response = {
            "status": proposal.get("status", "not_configured"),
            "mode": config.get("mode"),
            "metrics": metrics,
            "proposal": proposal,
            "validation": None,
            "submissions": [],
            "retrain": None,
            "error": proposal.get("error") or "agent scheduler did not produce actions",
        }
        invocations[invocation_id] = deepcopy(response)
        current = update_state_snapshot(current, config_version=config_version)
        return {**response, "state": current, "idempotent_replay": False}
    if not stored:
        agent_proposal = _build_execution_proposal(proposal, decision_snapshot, config, remaining, invocation_id, metrics["metrics"])
    policy = apply_execution_policy(
        agent_proposal,
        execution_mode=execution_mode,
        human_feedback=human_feedback,
        replay_record=replay_record,
    )
    if policy["status"] == "awaiting_approval":
        pending[invocation_id] = {
            "proposal": deepcopy(proposal),
            "agent_proposal": deepcopy(agent_proposal),
        }
        response = _policy_response("awaiting_approval", execution_mode, config, metrics, proposal, agent_proposal, policy)
        _record_action(current, invocation_id, execution_mode, agent_proposal, policy, None, None, response["status"])
        current = update_state_snapshot(current, config_version=config_version)
        return {**response, "state": current, "idempotent_replay": False}
    pending.pop(invocation_id, None)
    if policy["status"] == "rejected_by_user":
        response = _policy_response("rejected_by_user", execution_mode, config, metrics, proposal, agent_proposal, policy)
        _record_action(current, invocation_id, execution_mode, agent_proposal, policy, None, None, response["status"])
        invocations[invocation_id] = deepcopy(response)
        current = update_state_snapshot(current, config_version=config_version)
        return {**response, "state": current, "idempotent_replay": False}
    proposal = deepcopy(policy["final_action"]["parameters"]["proposal"])
    tool_validation = None
    if execution_session is not None or tool_registry is not None:
        if execution_session is None or tool_registry is None:
            raise ValueError("execution_session 和 tool_registry 必须同时提供")
        tool_validation = validate_tool_action(
            policy["final_action"], current, execution_session, tool_registry
        )
        if not tool_validation["valid"]:
            response = {"status": "rejected", "mode": config["mode"], "execution_mode": execution_mode, "metrics": metrics, "proposal": proposal, "agent_proposal": agent_proposal, "human_feedback": policy["human_feedback"], "final_action": policy["final_action"], "tool_validation": tool_validation, "validation": None, "submissions": [], "retrain": None, "execution_result": None}
            _record_action(current, invocation_id, execution_mode, agent_proposal, policy, policy["final_action"], None, response["status"])
            invocations[invocation_id] = deepcopy(response)
            current = update_state_snapshot(current, config_version=config_version)
            return {**response, "state": current, "idempotent_replay": False}
    validation = validate_dft_agent_decisions(proposal, metrics["metrics"], current, config=config, config_version=config_version, remaining_budget=remaining)
    if not validation["valid"]:
        response = {"status": "rejected", "mode": config["mode"], "execution_mode": execution_mode, "metrics": metrics, "proposal": proposal, "agent_proposal": agent_proposal, "human_feedback": policy["human_feedback"], "final_action": policy["final_action"], "tool_validation": tool_validation, "validation": validation, "submissions": [], "retrain": None, "execution_result": None}
    elif not policy["execute"]:
        response = {"status": "planned_only", "mode": config["mode"], "execution_mode": execution_mode, "metrics": metrics, "proposal": proposal, "agent_proposal": agent_proposal, "human_feedback": policy["human_feedback"], "final_action": policy["final_action"], "tool_validation": tool_validation, "validation": validation, "submissions": [], "retrain": None, "execution_result": None}
    else:
        current = reserve_dft_actions(current, validation["accepted"])
        submissions = submit_dft_actions(validation["accepted"], submitter=dft_submitter, dft_parameters=dft_parameters, context=context)
        current = update_dft_action_results(current, submissions)
        history = list(current.get("new_dft_records", [])) + [item for item in submissions if item.get("status") == "completed"]
        current["new_dft_records"] = history
        retrain = check_mlip_retrain_trigger(history, requested_action=validation["global_action"], config=config)
        response = {"status": "completed", "mode": config["mode"], "execution_mode": execution_mode, "metrics": metrics, "proposal": proposal, "agent_proposal": agent_proposal, "human_feedback": policy["human_feedback"], "final_action": policy["final_action"], "tool_validation": tool_validation, "validation": validation, "submissions": submissions, "retrain": retrain, "execution_result": submissions}
    _record_action(current, invocation_id, execution_mode, agent_proposal, policy, policy["final_action"], response.get("execution_result"), response["status"])
    invocations = current.setdefault("qbc_dft_invocations", {}); invocations[invocation_id] = deepcopy(response)
    current = update_state_snapshot(current, config_version=config_version)
    return {**response, "state": current, "idempotent_replay": False}


def _build_execution_proposal(proposal, state, config, remaining_budget, invocation_id, metrics):
    decisions = proposal.get("decisions") or []
    by_id = {item["candidate_id"]: item for item in metrics}
    action = {
        "tool": "select_dft_candidates",
        "parameters": {"proposal": deepcopy(proposal)},
        "analysis": {
            "candidate_count": len(decisions),
            "remaining_dft_budget": remaining_budget,
            "decision_mode": config.get("mode"),
            "state_status": state.get("status"),
        },
        "reason": proposal.get("reason") or "QBC uncertainty and phase-boundary acquisition decision",
        "budget": sum(estimate_dft_cost(item.get("action"), by_id.get(item.get("candidate_id"), {}), config) for item in decisions),
        "expected_purpose": "Select validated structures for DFT labeling and possible MLIP retraining",
        "decision_source": proposal.get("source", config.get("mode")),
        "task_key": invocation_id,
    }
    return build_agent_proposal(action, state)


def _policy_response(status, execution_mode, config, metrics, proposal, agent_proposal, policy):
    return {
        "status": status,
        "mode": config.get("mode"),
        "execution_mode": execution_mode,
        "metrics": metrics,
        "proposal": proposal,
        "agent_proposal": agent_proposal,
        "human_feedback": deepcopy(policy.get("human_feedback")),
        "final_action": deepcopy(policy.get("final_action")),
        "validation": None,
        "submissions": [],
        "retrain": None,
        "execution_result": None,
    }


def _record_action(state, record_id, mode, proposal, policy, final_action, result, status):
    record = {
        "record_id": record_id,
        "execution_mode": mode,
        "status": status,
        "agent_proposal": deepcopy(proposal),
        "human_feedback": deepcopy(policy.get("human_feedback")),
        "final_action": deepcopy(final_action),
        "execution_result": deepcopy(result),
    }
    records = state.setdefault("action_records", [])
    for index, existing in enumerate(records):
        if existing.get("record_id") == record_id:
            records[index] = record
            return
    records.append(record)
