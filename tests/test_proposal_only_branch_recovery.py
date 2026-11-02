import json

import pytest

from phase_agent.tools.state.registered_branch_evidence import inspect_registered_branches


def setup(tmp_path, monkeypatch):
    import phase_agent.tools.workflows.region_review as view

    root = tmp_path / "project"
    state_path = root / "workflow_state/state.json"
    plan = root / "workflow_state/branch_plans/proposal.json"
    ledger = root / "workflow_state/ledgers/phase_data.json"
    plan.parent.mkdir(parents=True)
    ledger.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "status": "awaiting_approval",
                "task_key": "initial",
                "config_version": "c",
                "model_version": "m",
            }
        )
    )
    ledger.write_text(json.dumps({"branches": {}, "structures": {}}))
    monkeypatch.setattr(view, "path_for", lambda *a: plan)
    state = {"active_model_version": "m", "confirmed_config_version": "c"}
    action = {
        "tool": "generate_branches",
        "task_key": "initial",
        "parameters": {"sampling_mode": "region_adaptive"},
    }
    return root, state_path, plan, ledger, state, action


def test_bound_proposal_is_not_a_registered_output(tmp_path, monkeypatch):
    _, path, _, _, state, action = setup(tmp_path, monkeypatch)
    assert inspect_registered_branches(state, action, path) is None


@pytest.mark.parametrize(
    "fault", ["registered", "unregistered_file", "identity", "completed_empty"]
)
def test_possible_effects_or_invalid_identity_still_block_no_effect(tmp_path, monkeypatch, fault):
    root, path, plan, ledger, state, action = setup(tmp_path, monkeypatch)
    if fault == "registered":
        ledger.write_text(json.dumps({"branches": {"b": {}}, "structures": {}}))
    elif fault == "unregistered_file":
        source = root / "structures/candidate_structures/partial.vasp"
        source.parent.mkdir(parents=True)
        source.write_text("partial")
    else:
        record = json.loads(plan.read_text())
        record["config_version" if fault == "identity" else "status"] = (
            "wrong" if fault == "identity" else "completed"
        )
        plan.write_text(json.dumps(record))
    assert inspect_registered_branches(state, action, path)["status"] == "invalid"
