"""Framework-independent proposal checks shared by LangGraph nodes."""

from copy import deepcopy
from phase_agent.decisions.agent.action_contracts import action_contract_errors


def proposal_errors(action, payload):
    from phase_agent.decisions.agent.dialogue_contract import is_dialogue, dialogue_errors

    if is_dialogue(action):
        return dialogue_errors(action, payload.get("unified_dialogue") is True)
    errors = action_contract_errors(action)
    if errors:
        return errors
    from phase_agent.decisions.agent.round_budget_review import round_budget_review_errors

    errors.extend(round_budget_review_errors(action, payload.get("decision_context") or {}))
    tool = action.get("tool") or action.get("action_type")
    mc = (payload.get("decision_context") or {}).get("mc_entry_conditions") or {}
    if (
        mc.get("required")
        and not mc.get("ready")
        and (
            tool == "allocate_mc_bohb"
            or (
                tool == "prepare_local_batch_files"
                and (action.get("parameters") or {}).get("mode") == "mc_inputs"
            )
        )
    ):
        errors.append(
            "action.prerequisite: MC requires recovered Relax results and current-model MLIP hull; first propose separate relax_inputs preparation, without MC allocation"
        )
    if tool not in payload["allowed_tools"]:
        errors.append("action.tool: not allowed in this request")
    if (payload.get("decision_context") or {}).get("post_dft_assessment"):
        from phase_agent.decisions.agent.post_dft_review import post_dft_review_errors

        errors.extend(post_dft_review_errors(action))
    if tool == "select_dft_candidates":
        from phase_agent.decisions.agent.dft_contracts import dft_contract_errors

        errors.extend(dft_contract_errors(action.get("parameters", {}).get("decisions")))
    if tool == "allocate_mc_bohb":
        from phase_agent.decisions.agent.mc_contracts import mc_contract_errors

        errors.extend(
            mc_contract_errors(
                action.get("parameters", {}), fallback_budget=action.get("budget", 0)
            )
        )
    if tool == "generate_branches":
        from phase_agent.decisions.agent.generation_plan import disabled_generation_allocations

        enabled = (payload.get("decision_context") or {}).get("enabled_generation_strategies")
        if enabled is not None:
            disabled = disabled_generation_allocations(action.get("parameters") or {}, enabled)
            if disabled:
                errors.append(
                    f"parameters.generation_plan: disabled strategies {disabled}; enabled={enabled}"
                )
    if tool == "generate_branches" and action.get("parameters", {}).get("generation_plan"):
        from phase_agent.decisions.agent.generation_plan import validate_generation_plan

        try:
            validate_generation_plan(action["parameters"])
        except (ValueError, TypeError) as error:
            errors.append(f"parameters.generation_plan: {error}")
    return errors


def preserve_proposal_evidence(error, action):
    if not isinstance(action, dict):
        return
    if action.get("_llm_usage"):
        error.llm_usage = deepcopy(action["_llm_usage"])
    from phase_agent.decisions.agent.post_dft_review import post_dft_review_errors, REVIEW_TOOLS

    tool = action.get("tool") or action.get("action_type")
    if tool in REVIEW_TOOLS and not post_dft_review_errors(action, include_parameters=False):
        error.post_dft_review = deepcopy(action["post_dft_review"])
