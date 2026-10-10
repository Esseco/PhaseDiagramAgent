"""Validate an Agent-authored follow-up without choosing a scientific direction."""

from copy import deepcopy
from phase_agent.decisions.agent.action_contracts import action_contract_errors
from phase_agent.decisions.agent.post_dft_review import post_dft_review_errors
from phase_agent.tools.state.approved_direction import TOOLS


def followup_errors(choice, action, *, require_round_review=False):
    if choice == "activate":
        return []
    if not isinstance(action, dict):
        return ["parameters.followup_action必须包含完整的具体动作"]
    errors = action_contract_errors(action)
    if action.get("tool") not in TOOLS.get(choice, set()):
        errors.append("followup_action.tool与方向不一致")
    if not action.get("reason") or not action.get("expected_purpose"):
        errors.append("followup_action需要reason和expected_purpose")
    if require_round_review:
        errors.extend(post_dft_review_errors(action))
    return errors


def saved_followup(direction):
    action = direction.get("plan")
    return deepcopy(action) if isinstance(action, dict) else None
