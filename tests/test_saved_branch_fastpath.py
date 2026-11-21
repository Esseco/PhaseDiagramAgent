from types import SimpleNamespace
import pytest
from phase_agent.tools.workflows.saved_dft_input_frame import prepare_saved_dft_input_frame


@pytest.mark.parametrize(
    "block", ["none", "active", "return", "feedback", "unknown", "model", "recovery"]
)
def test_branch_shortcut_keeps_recovery_and_identity_guards(tmp_path, monkeypatch, block):
    state = {
        "active_model_version": "m",
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m"}},
        "tasks": [],
        "processed_task_ids": [],
        "feedback_processed_task_ids": [],
        "pending_execution_policies": {},
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "runtime_adapters": {},
        "state_path": str(tmp_path / "state.json"),
        "effective_config": {},
        "agent_client": SimpleNamespace(_codex_structured=True, _frozen_generation_draft=True),
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return",
        lambda s, **kwargs: block == "return",
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report",
        lambda *a: {"unsettled": [1] if block == "recovery" else []},
    )
    if block == "active":
        state["tasks"] = [{"task_id": "busy", "status": "pending"}]
    if block == "feedback":
        state["tasks"] = [
            {"task_id": "scientific", "stage": "relax_and_feature", "status": "completed"}
        ]
        state["processed_task_ids"] = ["scientific"]
    if block == "unknown":
        state["processed_task_ids"] = ["unknown"]
    if block == "model":
        state["phase_diagrams"]["mlip"]["model_version"] = "other"
    assert bool(prepare_saved_dft_input_frame(frame).get("saved_input_only")) == (block == "none")


def test_frozen_approved_branch_and_ui_bookkeeping(tmp_path, monkeypatch):
    identifier = "webui-example:task"
    state = {
        "active_model_version": "m",
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m"}},
        "tasks": [{"task_id": identifier, "stage": None, "status": "completed"}],
        "processed_task_ids": [identifier],
        "pending_execution_policies": {
            "p": {"agent_proposal": {"raw_action": {"tool": "generate_branches"}}}
        },
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "invocation_id": "p",
        "human_feedback": {"decision": "approve"},
        "state_path": str(tmp_path / "state.json"),
        "effective_config": {},
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return", lambda s, **kwargs: False
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report", lambda *a: {"unsettled": []}
    )
    assert prepare_saved_dft_input_frame(frame)["saved_input_only"] is True
    frame["human_feedback"] = None
    assert not prepare_saved_dft_input_frame(frame).get("saved_input_only")


def test_branch_draft_with_precollected_results_or_pending_proposal_cannot_skip(
    tmp_path, monkeypatch
):
    state = {
        "active_model_version": "m",
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m"}},
        "tasks": [],
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "runtime_adapters": {},
        "state_path": str(tmp_path / "state.json"),
        "effective_config": {},
        "agent_client": SimpleNamespace(_codex_structured=True, _frozen_generation_draft=True),
    }
    frame["automatic_results"] = [{"task_id": "new"}]
    assert not prepare_saved_dft_input_frame(frame).get("saved_input_only")
    frame["automatic_results"] = []
    state["pending_execution_policies"] = {"other": {}}
    assert not prepare_saved_dft_input_frame(frame).get("saved_input_only")


@pytest.mark.parametrize(
    "block",
    [
        "none",
        "active",
        "return",
        "unprocessed",
        "recovery",
        "model",
        "hull",
        "config",
        "missing_preview",
        "new_results",
    ],
)
def test_frozen_mc_approval_shortcut_requires_current_idle_evidence(tmp_path, monkeypatch, block):
    action = {
        "tool": "allocate_mc_bohb",
        "target_ids": ["B"],
        "parameters": {
            "hull_reference_version": "h",
            "phase_diagram_version": "p",
            "budget_preview": {
                "model_version": "m",
                "hull_reference_version": "h",
                "phase_diagram_version": "p",
                "allocation_checksum": "checksum",
                "allocations": [{"branch_id": "B"}],
            },
        },
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "phase_diagrams_by_model": {
            "m": {"status": "completed", "model_version": "m", "version": "p"}
        },
        "branch_hull_batches": {"h": {"version": "h", "model_version": "m"}},
        "tasks": [],
        "pending_execution_policies": {
            "p": {"config_version": "c", "agent_proposal": {"raw_action": action}}
        },
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "invocation_id": "p",
        "human_feedback": {"decision": "approve"},
        "state_path": str(tmp_path / "state.json"),
        "snapshot": {"config_version": "c"},
        "effective_config": {},
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return",
        lambda s, **k: block == "return",
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report",
        lambda *a: {"unsettled": [1] if block == "recovery" else []},
    )
    if block == "active":
        state["tasks"] = [{"task_id": "busy", "status": "pending"}]
    if block == "unprocessed":
        state["processed_task_ids"] = ["unknown"]
    if block == "model":
        state["phase_diagrams_by_model"]["m"]["model_version"] = "old"
    if block == "hull":
        state["branch_hull_batches"]["h"]["version"] = "old"
    if block == "config":
        frame["snapshot"]["config_version"] = "other"
    if block == "missing_preview":
        action["parameters"]["budget_preview"] = {}
    if block == "new_results":
        frame["recovered_results"] = [{"task_id": "new"}]
    assert bool(prepare_saved_dft_input_frame(frame).get("saved_input_only")) == (block == "none")


def test_frozen_mc_draft_shortcut_requires_codex_action_and_no_other_proposal(
    tmp_path, monkeypatch
):
    action = {
        "tool": "allocate_mc_bohb",
        "target_ids": ["B"],
        "parameters": {
            "hull_reference_version": "h",
            "phase_diagram_version": "p",
            "budget_preview": {
                "model_version": "m",
                "hull_reference_version": "h",
                "phase_diagram_version": "p",
                "allocation_checksum": "checksum",
                "allocations": [{"branch_id": "B"}],
            },
        },
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "phase_diagrams_by_model": {
            "m": {"status": "completed", "model_version": "m", "version": "p"}
        },
        "branch_hull_batches": {"h": {"version": "h", "model_version": "m"}},
        "tasks": [],
        "pending_execution_policies": {},
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "snapshot": {"config_version": "c"},
        "state_path": str(tmp_path / "state.json"),
        "effective_config": {},
        "agent_client": SimpleNamespace(_codex_structured=True, _frozen_mc_action=action),
    }
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return", lambda s, **k: False
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report", lambda *a: {"unsettled": []}
    )
    assert prepare_saved_dft_input_frame(frame)["saved_input_only"] is True
    state["pending_execution_policies"] = {"other": {}}
    assert not prepare_saved_dft_input_frame(frame).get("saved_input_only")
    state["pending_execution_policies"] = {}
    frame["agent_client"]._codex_structured = False
    assert not prepare_saved_dft_input_frame(frame).get("saved_input_only")


@pytest.mark.parametrize("block", ["none", "model", "hull", "config", "checksum", "targets", "returns", "unsettled", "new_results"])
def test_approved_energy_repair_fast_path_checks_frozen_evidence(tmp_path, monkeypatch, block):
    from phase_agent.tools.remote.integrity import payload_checksum

    preview = {"config_version": "c", "model": {"version": "m", "sha256": "sha", "path": "/m"}, "hull_version": "h", "frames": [{"source_task_id": "t"}], "task_count": 1}
    preview["checksum"] = payload_checksum(preview)
    action = {"tool": "prepare_local_batch_files", "target_ids": ["t"], "parameters": {"mode": "mc_energy_repair_inputs", "energy_repair_preview": preview}}
    state = {"active_model_version": "m", "active_model": {"version": "m", "sha256": "sha", "model_path": "/m"}, "confirmed_config_version": "c", "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m", "version": "h"}}, "tasks": [], "pending_execution_policies": {"p": {"agent_proposal": {"raw_action": action}}}}
    frame = {"loaded_state": state, "execution_mode": "interactive", "invocation_id": "p", "human_feedback": {"decision": "approve"}, "snapshot": {"config_version": "c"}, "state_path": str(tmp_path / "state.json"), "effective_config": {}}
    monkeypatch.setattr("phase_agent.tools.remote.transfer_bundle.has_registered_return", lambda *args, **kwargs: block == "returns")
    monkeypatch.setattr("phase_agent.tools.state.execution_receipts.recovery_report", lambda *args: {"unsettled": [1] if block == "unsettled" else []})
    if block == "model": state["active_model"]["sha256"] = "changed"
    if block == "hull": state["phase_diagrams"]["mlip"]["version"] = "changed"
    if block == "config": state["confirmed_config_version"] = "changed"
    if block == "checksum": preview["checksum"] = "changed"
    if block == "targets": action["target_ids"] = ["other"]
    if block == "new_results": frame["automatic_results"] = [{"task_id": "new"}]
    assert bool(prepare_saved_dft_input_frame(frame).get("saved_input_only")) == (block == "none")
