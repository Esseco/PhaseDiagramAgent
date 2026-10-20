"""Reoffer a reviewed no-effect input plan, with fresh approval and no execution."""

from copy import deepcopy
import hashlib

from phase_agent.tools.policy.file_approval import proposal_hash

INPUT_MODES = {
    "relax_mc_inputs",
    "relax_inputs",
    "mc_inputs",
    "dft_inputs",
    "model_refresh_inputs",
    "ordering_search_inputs",
}


def reoffer_input_plan(state, original, *, identity, review_id):
    if not original or state.get("pending_execution_policies"):
        return None
    action = (original.get("agent_proposal") or {}).get("raw_action") or {}
    binding = original.get("project_identity") or {}
    if (
        action.get("tool") != "prepare_local_batch_files"
        or (action.get("parameters") or {}).get("mode") not in INPUT_MODES
        or not action.get("task_key")
        or identity.get("tool") != action["tool"]
        or identity.get("action_hash") != proposal_hash(action)
        or identity.get("config_version") != state.get("confirmed_config_version")
        or binding.get("config_version") != state.get("confirmed_config_version")
        or not binding.get("model_version")
        or binding.get("model_version") != state.get("active_model_version")
    ):
        return None
    suffix = hashlib.sha256((identity["invocation_id"] + review_id).encode()).hexdigest()[:24]
    plan_id = "recovery-input:" + suffix
    pending = deepcopy(original)
    pending.update(
        record_id=plan_id,
        revision=0,
        feedback_history=[],
        recovery_source_record_id=identity["invocation_id"],
        recovery_review_id=review_id,
    )
    pending.pop("human_feedback", None)
    state.setdefault("pending_execution_policies", {})[plan_id] = pending
    state.setdefault("action_records", []).append(
        {
            "record_id": plan_id,
            "status": "awaiting_approval",
            "agent_proposal": deepcopy(pending["agent_proposal"]),
            "human_feedback": None,
            "execution_result": None,
            "recovery_source_record_id": identity["invocation_id"],
        }
    )
    return plan_id
