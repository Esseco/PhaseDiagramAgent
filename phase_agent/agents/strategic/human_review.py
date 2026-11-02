"""Human direction is scoped guidance, separate from scientific execution approval."""

from copy import deepcopy
from phase_agent.policies.file_approval import proposal_hash


def authorization_settings(config):
    import math
    from phase_agent.tools.dispatch.scientific_operations import OPERATIONS

    workflow = config.get("workflow") or {}
    automatic = workflow.get("auto_approved_operations", [])
    if (
        not isinstance(automatic, list)
        or any(not isinstance(name, str) or name not in OPERATIONS for name in automatic)
        or len(automatic) != len(set(automatic))
    ):
        raise ValueError("workflow.auto_approved_operations must list unique registered operations")
    threshold = workflow.get("human_review_relative_cost")
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("workflow.human_review_relative_cost must be positive and finite")
    retry_limit = workflow.get("auto_retry_limit", 1)
    if type(retry_limit) is not int or retry_limit < 0:
        raise ValueError("workflow.auto_retry_limit must be a nonnegative integer")
    return {
        "auto_approved_operations": automatic,
        "human_review_relative_cost": threshold,
        "auto_retry_limit": retry_limit,
    }


def review_reasons(action, config, *, estimated_cost=None):
    from phase_agent.agents.strategic.research_plan import HumanReviewRequest
    from phase_agent.policies.workflow_policy import needs_human

    settings = authorization_settings(config)
    reasons = []
    request = HumanReviewRequest.model_validate(action.get("human_review_request") or {})
    if request.required or request.major_decision or request.expert_conflict == "major":
        reasons.append(request.reason)
    if needs_human(action, config):
        reasons.append("sensitive_action_requires_human_approval")
    threshold = settings["human_review_relative_cost"]
    from phase_agent.agents.strategic.research_plan import ResearchPlan

    plan = (
        ResearchPlan.model_validate(action["research_plan"])
        if action.get("research_plan")
        else None
    )
    active_estimate = plan.steps[0].estimated_cost if plan else None
    costs = [
        value
        for value in (action.get("budget"), estimated_cost, active_estimate)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if threshold is not None and costs and max(costs) > threshold:
        reasons.append("configured_high_cost_review_required")
    if (action.get("parameters") or {}).get("rebuild_inputs"):
        reasons.append("input_rebuild_requires_human_approval")
    if (action.get("tool") or action.get("action_type")) == "restart_failed_task":
        parameters = action.get("parameters") or {}
        maximum = parameters.get("max_retries", 1)
        if type(maximum) is not int or maximum > settings["auto_retry_limit"]:
            reasons.append("retry_limit_exceeds_confirmed_automatic_scope")
        if parameters.get("repair_profile"):
            reasons.append("retry_parameter_change_requires_human_review")
    return reasons


def policy_auto_approved(action, config, *, estimated_cost=None):
    return action.get("operation") in authorization_settings(config)[
        "auto_approved_operations"
    ] and not review_reasons(action, config, estimated_cost=estimated_cost)


def direction_review(action, *, feedback=None):
    return {
        "status": "accepted" if feedback is not None else "awaiting_human_input",
        "action_hash": proposal_hash(action),
        "feedback": deepcopy(feedback),
        "execution_approved": False,
        "instruction": "先审阅研究方向，可提出修改意见；接受方向后仍需单独批准具体执行方案。",
    }
