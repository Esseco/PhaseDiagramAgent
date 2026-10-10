"""Read-only chat approval classification; execution policy remains authoritative."""


def is_sensitive_proposal(proposal):
    action = proposal.get("raw_action") or {}
    tool = action.get("tool") or action.get("stage")
    from phase_agent.tools.policy.training_input_action import is_training_input_action

    if is_training_input_action(action):
        return False
    if tool in {"update_mlip", "activate_model", "change_hard_constraint", "dft_relax"}:
        return True
    decisions = (action.get("parameters") or {}).get("decisions") or []
    return any(row.get("action") == "DFT_RELAX" for row in decisions if isinstance(row, dict))
