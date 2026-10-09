"""Framework-independent proposal checks shared by LangGraph nodes."""
from copy import deepcopy
from decision_layer.agent.action_contracts import action_contract_errors


def proposal_errors(action, payload):
    errors = action_contract_errors(action)
    if errors:
        return errors
    from decision_layer.agent.round_budget_review import round_budget_review_errors
    errors.extend(round_budget_review_errors(action, payload.get("decision_context") or {}))
    tool = action.get("tool") or action.get("action_type")
    if tool not in payload["allowed_tools"]:
        errors.append("action.tool: not allowed in this request")
    if (payload.get("decision_context") or {}).get("post_dft_assessment"):
        from decision_layer.agent.post_dft_review import post_dft_review_errors
        errors.extend(post_dft_review_errors(action))
    if tool == "select_dft_candidates":
        from decision_layer.agent.dft_contracts import dft_contract_errors
        errors.extend(dft_contract_errors(action.get("parameters", {}).get("decisions")))
    if tool == "allocate_mc_bohb":
        from decision_layer.agent.mc_contracts import mc_contract_errors
        errors.extend(mc_contract_errors(action.get("parameters", {}), fallback_budget=action.get("budget", 0)))
    if tool == "generate_branches" and action.get("parameters", {}).get("generation_plan"):
        from decision_layer.agent.generation_plan import validate_generation_plan
        try:
            validate_generation_plan(action["parameters"])
        except (ValueError, TypeError):
            errors.append("parameters.generation_plan: invalid allocation")
    return errors


def preserve_proposal_evidence(error, action):
    if not isinstance(action, dict):
        return
    if action.get("_llm_usage"):
        error.llm_usage = deepcopy(action["_llm_usage"])
    from decision_layer.agent.post_dft_review import post_dft_review_errors, REVIEW_TOOLS
    tool = action.get("tool") or action.get("action_type")
    if tool in REVIEW_TOOLS and not post_dft_review_errors(action, include_parameters=False):
        error.post_dft_review = deepcopy(action["post_dft_review"])
