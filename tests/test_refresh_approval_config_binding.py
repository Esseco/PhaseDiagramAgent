"""A persisted refresh approval cannot authorize a different config version."""

from copy import deepcopy
from tests.test_model_refresh_workflow import setup
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
from phase_agent.policies.file_approval import proposal_hash


def test_config_only_change_requires_fresh_durable_approval(tmp_path):
    state, session, context, registry = setup(tmp_path)
    state["confirmed_config_version"] = "c"
    context["state_path"] = str(tmp_path / "state.json")
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert first["status"] == "awaiting_approval"
    old = deepcopy(first["agent_proposal"])
    changed = first["state"]
    changed["confirmed_config_version"] = "c2"
    session["confirmed_snapshot"]["config_version"] = "c2"
    context["config_version"] = "c2"
    second = run_model_refresh_gate(
        changed, session, registry=registry, context=context, human_feedback="同意"
    )
    assert second["status"] == "awaiting_approval"
    assert not second["state"]["tasks"] and not (tmp_path / "submissions").exists()
    assert second["action"]["task_key"] == first["action"]["task_key"]
    assert second["action"]["parameters"] == first["action"]["parameters"]
    assert proposal_hash(second["agent_proposal"]) != proposal_hash(old)
    final = run_model_refresh_gate(
        second["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert final["status"] == "prepared", final
    assert final["state"]["tasks"]
    assert final["state"]["model_refresh"]["input_authorization"]["config_version"] == "c2"
