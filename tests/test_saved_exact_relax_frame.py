from copy import deepcopy
from unittest.mock import patch

import pytest

from phase_agent.tools.workflows.saved_exact_relax_frame import prepare_saved_exact_relax_frame


def build():
    hashes = {"c": "a" * 64}
    auth = {
        "status": "approved_for_preparation",
        "submit_authorized": False,
        "candidate_frame_sha256": hashes,
        "model_version": "m",
        "config_version": "v",
        "hull_version": "h",
    }
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_mlip_relax",
        "target_ids": ["c"],
        "parameters": {
            "mode": "relax_inputs",
            "selection_scope": "candidate_ids",
            "model_version": "m",
            "candidate_frame_sha256": hashes,
        },
    }
    cfg = {"x": 1}
    state = {
        "confirmed_config": cfg,
        "confirmed_config_version": "v",
        "active_model_version": "m",
        "active_model": {"version": "m"},
        "phase_diagrams": {"mlip": {"version": "h"}},
        "exact_candidate_relax_authorization": auth,
        "pending_execution_policies": {
            "p": {"config_version": "v", "agent_proposal": {"raw_action": action}}
        },
        "tasks": [],
    }
    return {
        "loaded_state": state,
        "snapshot": {"config_version": "v", "config": deepcopy(cfg)},
        "execution_mode": "interactive",
        "state_path": "state.json",
        "effective_config": {},
        "invocation_id": "p",
        "human_feedback": {"decision": "approve"},
    }


def run(frame, active=False, returned=False, backlog=False, unsettled=False):
    with (
        patch(
            "phase_agent.policies.workflow_policy.active_tasks", return_value=[1] if active else []
        ),
        patch(
            "phase_agent.tools.remote.transfer_bundle.has_registered_return", return_value=returned
        ),
        patch(
            "phase_agent.tools.workflows.recover_saved_results.recovery_candidate",
            return_value={"pending": backlog, "backlog_count": 0},
        ),
        patch(
            "phase_agent.tools.state.execution_receipts.recovery_report",
            return_value={"status": "clear", "unsettled": [1] if unsettled else []},
        ),
    ):
        return prepare_saved_exact_relax_frame(frame)


def test_frozen_exact_approval_reuses_state():
    f = build()
    assert run(f)["saved_input_only"] is True


@pytest.mark.parametrize(
    "change",
    [
        "config",
        "model",
        "hull",
        "hash",
        "active",
        "returned",
        "backlog",
        "unsettled",
        "other_pending",
        "recovered",
        "submit",
    ],
)
def test_fast_path_fails_closed(change):
    f = build()
    state = f["loaded_state"]
    kwargs = {}
    if change == "config":
        f["snapshot"]["config_version"] = "new"
    if change == "model":
        state["active_model_version"] = "new"
    if change == "hull":
        state["phase_diagrams"]["mlip"]["version"] = "new"
    if change == "hash":
        state["exact_candidate_relax_authorization"]["candidate_frame_sha256"] = {"c": "b" * 64}
    if change in {"active", "returned", "backlog", "unsettled"}:
        kwargs[change] = True
    if change == "other_pending":
        state["pending_execution_policies"]["other"] = {}
    if change == "recovered":
        f["recovered_results"] = [{"task_id": "t"}]
    if change == "submit":
        state["exact_candidate_relax_authorization"]["submit_authorized"] = True
    assert run(f, **kwargs) is f


@pytest.mark.parametrize("change", [None, "config_content", "model_content", "collector"])
def test_normal_frozen_record_uses_saved_project_identity(change):
    from phase_agent.evidence.project_authority import project_identity

    frame = build()
    state = frame["loaded_state"]
    pending = state["pending_execution_policies"]["p"]
    pending.pop("config_version")
    pending["project_identity"] = project_identity(state)
    if change == "config_content":
        state["confirmed_config"]["x"] = 2
        frame["snapshot"]["config"]["x"] = 2
    elif change == "model_content":
        state["active_model"]["sha256"] = "different"
    elif change == "collector":
        frame["result_collector"] = object()
    assert bool(run(frame).get("saved_input_only")) is (change is None)
