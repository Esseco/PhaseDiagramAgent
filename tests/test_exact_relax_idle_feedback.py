from copy import deepcopy
from pathlib import Path
import hashlib
from types import SimpleNamespace

import pytest

from phase_agent.tools.workflows.saved_dft_input_frame import prepare_saved_dft_input_frame


def make_frame(tmp_path, monkeypatch, *, saved=False):
    path = tmp_path / "frame.vasp"
    path.write_bytes(b"frozen geometry")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_mlip_relax",
        "target_ids": ["candidate"],
        "parameters": {
            "mode": "relax_inputs",
            "selection_scope": "candidate_ids",
            "candidate_frame_sha256": {"candidate": digest},
            "model_version": "m",
        },
    }
    state = {
        "active_model_version": "m",
        "confirmed_config_version": "c",
        "confirmed_config": {},
        "phase_diagrams": {"mlip": {"status": "completed", "model_version": "m", "version": "h"}},
        "exact_candidate_relax_authorization": {
            "status": "approved_for_preparation",
            "model_version": "m",
            "config_version": "c",
            "hull_version": "h",
            "candidate_frame_sha256": {"candidate": digest},
            "user_instruction": "prepare",
            "submit_authorized": False,
        },
        "qbc_candidates": [
            {
                "candidate_id": "candidate",
                "structure_id": "s",
                "structure_path": str(path),
                "structure_sha256": digest,
            }
        ],
        "tasks": [],
        "pending_execution_policies": {},
    }
    frame = {
        "loaded_state": state,
        "execution_mode": "interactive",
        "runtime_adapters": {},
        "relax_recovery_updated": False,
        "state_path": tmp_path / "state.json",
        "effective_config": {},
        "snapshot": {"config_version": "c", "config": {}},
        "manager": SimpleNamespace(data={"structures": {"s": {}}}),
        "agent_client": SimpleNamespace(_codex_structured=True, _frozen_exact_relax_action=action),
    }
    if saved:
        frame.update(invocation_id="p", human_feedback={"decision": "approve"})
        state["pending_execution_policies"]["p"] = {
            "config_version": "c",
            "agent_proposal": {"raw_action": deepcopy(action)},
        }
    monkeypatch.setattr(
        "phase_agent.tools.remote.transfer_bundle.has_registered_return",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.recover_saved_results.recovery_candidate",
        lambda *args, **kwargs: {"pending": False, "backlog_count": 0},
    )
    monkeypatch.setattr(
        "phase_agent.tools.state.execution_receipts.recovery_report",
        lambda *args: {"status": "clear", "unsettled": []},
    )
    monkeypatch.setattr(
        "phase_agent.evidence.dft_failure_policy.unreviewed_failure_batches", lambda state: []
    )
    monkeypatch.setattr("phase_agent.persistence.shared_dft.settings", lambda *args: False)
    return frame, action, path


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(
    "block",
    [
        None,
        "new",
        "active",
        "backlog",
        "return",
        "receipt",
        "model",
        "effective_model",
        "config",
        "hull",
        "target",
        "file",
        "collector",
        "other_pending",
        "startup",
        "shared",
        "dft_review",
        "feedback_error",
        "replay",
    ],
)
def test_exact_relax_fastpath_fails_closed(tmp_path, monkeypatch, saved, block):
    frame, action, path = make_frame(tmp_path, monkeypatch, saved=saved)
    state = frame["loaded_state"]
    if block == "new":
        frame["recovered_results"] = [{"task_id": "new"}]
    if block == "active":
        state["tasks"] = [{"task_id": "active", "status": "running"}]
    if block == "backlog":
        monkeypatch.setattr(
            "phase_agent.tools.workflows.recover_saved_results.recovery_candidate",
            lambda *a, **k: {"pending": True, "backlog_count": 1},
        )
    if block == "return":
        monkeypatch.setattr(
            "phase_agent.tools.remote.transfer_bundle.has_registered_return", lambda *a, **k: True
        )
    if block == "receipt":
        monkeypatch.setattr(
            "phase_agent.tools.state.execution_receipts.recovery_report",
            lambda *a: {"status": "unsettled", "unsettled": ["x"]},
        )
    if block == "model":
        state["active_model_version"] = "changed"
    if block == "effective_model":
        frame["effective_config"]["mlip"] = {"version": "changed"}
    if block == "config":
        state["confirmed_config"] = {"changed": True}
    if block == "hull":
        state["phase_diagrams"]["mlip"]["version"] = "other"
    if block == "target":
        state["exact_candidate_relax_authorization"]["candidate_frame_sha256"] = {"other": "x"}
    if block == "file":
        path.write_bytes(b"changed")
    if block == "collector":
        frame["result_collector"] = object()
    if block == "other_pending":
        state["pending_execution_policies"]["other"] = {}
    if block == "startup":
        state["reference_dft_startup"] = {"status": "pending"}
    if block == "shared":
        monkeypatch.setattr("phase_agent.persistence.shared_dft.settings", lambda *a: True)
    if block == "dft_review":
        monkeypatch.setattr(
            "phase_agent.evidence.dft_failure_policy.unreviewed_failure_batches",
            lambda state: [{"failed": True}],
        )
    if block == "feedback_error":
        state["scientific_feedback_errors"] = {"t": {"code": "invalid"}}
    if block == "replay":
        frame["replay_record"] = {"status": "unknown"}
    from phase_agent.tools.workflows.saved_dft_input_frame import reuse_exact_after_collection

    assert not prepare_saved_dft_input_frame(frame).get("saved_input_only")
    frame["pre_reconciled"] = {"state": state, "reconciled": []}
    frame["combined_results"] = frame.get("recovered_results") or []
    assert reuse_exact_after_collection(frame) is (block is None)


def test_project_graph_collects_before_skipping_both_stages(tmp_path, monkeypatch):
    from phase_agent.graphs.project.runner import run_search_workflow_graph
    from phase_agent.tools.workflows.saved_dft_input_frame import reuse_exact_after_collection

    frame, _, _ = make_frame(tmp_path, monkeypatch)
    frame["_workflow_prepared"] = True
    calls = []

    def initialize():
        return prepare_saved_dft_input_frame(frame)

    def collect(current):
        calls.append("collect")
        current["combined_results"] = current.get("recovered_results") or []
        return current

    def analyze(current):
        current["pre_reconciled"] = {"state": current["loaded_state"], "reconciled": []}
        current["reuse"] = reuse_exact_after_collection(current)
        if not current["reuse"]:
            calls.append("feedback")
        current["feedback"] = {"state": current["loaded_state"]}
        return current

    def wait(current):
        return current

    def assess(current):
        if not current["reuse"]:
            calls.append("round")
        return current

    def act(current, **kwargs):
        calls.append("action")
        current["result"] = {"status": "awaiting_approval"}
        return current

    def finalize(current):
        return current["result"]

    stages = dict(
        initialize=initialize,
        collect=collect,
        analyze=analyze,
        wait=wait,
        assess=assess,
        act=act,
        finalize=finalize,
    )
    assert run_search_workflow_graph(**stages)["status"] == "awaiting_approval"
    assert calls == ["collect", "action"]
    calls.clear()
    frame["recovered_results"] = [{"task_id": "new"}]
    assert run_search_workflow_graph(**stages)["status"] == "awaiting_approval"
    assert calls == ["collect", "feedback", "round", "action"]


@pytest.mark.parametrize(
    "block",
    [
        None,
        "new",
        "backlog",
        "collector",
        "legacy",
        "legacy_valid",
        "duplicate",
        "duplicate_changed",
        "changed_frame",
        "changed_rules",
        "unrecorded",
        "training",
        "recovery_changed",
        "old_checkpoint",
        "collection_error",
        "metadata_error",
        "missing_state_path",
        "changed_energy",
        "outside",
        "no_match",
        "no_match_new_rules",
        "no_match_new_structure",
        "no_match_legacy",
        "missing_structure",
        "relax_failure",
    ],
)
def test_default_collector_reuse_only_with_verified_scientific_evidence(
    tmp_path, monkeypatch, block
):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.workflows.saved_dft_input_frame import reuse_exact_after_collection

    frame, _, path = make_frame(tmp_path, monkeypatch)
    state = frame["loaded_state"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state["tasks"] = [
        {
            "task_id": "t",
            "structure_id": "s",
            "model_version": "m",
            "status": "completed",
            "stage": "deep_search",
            "outputs": {"energy": -1},
        }
    ]
    state["phase_records"] = [
        {
            "source_task_id": "t",
            "structure_id": "s",
            "model_version": "m",
            "energy": -1,
            "status": "completed",
            "structure_path": str(path),
            "phase_identification": {
                "status": "identified",
                "phase": "O3",
                "method": "test",
                "portable_classification_scope": "rules",
                "structure_sha256": digest,
            },
        }
    ]
    from phase_agent.tools.workflows.apply_scientific_feedback import phase_records_checksum

    record = state["phase_records"][0]
    record.update(
        stage=state["tasks"][0]["stage"], energy_unit="eV", checks_passed=True, phase="O3"
    )
    state["tasks"][0]["outputs"].update(
        energy_unit="eV",
        actual_phase="O3",
        phase_identification=deepcopy(record["phase_identification"]),
    )
    state["phase_diagram_input_checksum"] = phase_records_checksum(state["phase_records"])
    frame.update(
        pre_reconciled={"state": state, "reconciled": []},
        combined_results=[],
        result_collector=RemoteBatchRunner(tmp_path, worker_command=[]),
        phase_references={},
    )
    monkeypatch.setattr(
        "phase_agent.analysis.phase.identify_result_phase._portable_classification_scope",
        lambda *a, **k: "rules",
    )
    loads = []

    def load_structure(outputs):
        loads.append(outputs["structure_path"])
        return None, hashlib.sha256(Path(outputs["structure_path"]).read_bytes()).hexdigest()

    monkeypatch.setattr(
        "phase_agent.science.structures.load_result_structure.load_result_structure", load_structure
    )
    if block == "new":
        frame["combined_results"] = [{"task_id": "new"}]
    if block == "backlog":
        frame["pre_reconciled"]["reconciled"] = [{"status": "settled"}]
    if block == "collector":
        frame["result_collector"] = object()
    if block in {"legacy", "legacy_valid"}:
        evidence = state["phase_records"][0]["phase_identification"]
        evidence.pop("portable_classification_scope")
        if block == "legacy_valid":
            from phase_agent.analysis.phase.identify_result_phase import _classification_scope

            evidence["classification_scope"] = _classification_scope(
                frame["manager"], {}, configured_phase=None
            )
            state["tasks"][0]["outputs"]["phase_identification"] = deepcopy(evidence)
            from phase_agent.tools.workflows.apply_scientific_feedback import phase_records_checksum

            state["phase_diagram_input_checksum"] = phase_records_checksum(state["phase_records"])
    if block in {"duplicate", "duplicate_changed"}:
        task = deepcopy(state["tasks"][0])
        task["task_id"] = "t2"
        state["tasks"].append(task)
        record = deepcopy(state["phase_records"][0])
        record["source_task_id"] = "t2"
        state["phase_records"].append(record)
        state["phase_diagram_input_checksum"] = phase_records_checksum(state["phase_records"])
        if block == "duplicate_changed":

            def change_file(*args):
                path.write_bytes(b"changed between records")
                return None

            monkeypatch.setattr(
                "phase_agent.analysis.phase.identify_result_phase._configured_phase", change_file
            )
    if block == "changed_frame":
        state["phase_records"][0]["phase_identification"]["structure_sha256"] = "old"
    if block == "changed_rules":
        monkeypatch.setattr(
            "phase_agent.analysis.phase.identify_result_phase._portable_classification_scope",
            lambda *a, **k: "other",
        )
    if block == "unrecorded":
        state["tasks"] = [
            {
                "task_id": "missing",
                "stage": "deep_search",
                "status": "completed",
                "outputs": {"energy": -1},
            }
        ]
    if block == "training":
        frame["training_returns"] = [{"status": "completed"}]
    if block == "recovery_changed":
        frame["relax_recovery_updated"] = True
    if block == "old_checkpoint":
        frame.pop("relax_recovery_updated")
    if block == "collection_error":
        frame["collection_report"] = {"invalid_count": 1}
    if block == "metadata_error":
        frame["collection_report"] = {"metadata_refresh_report": {"invalid_count": 1}}
    if block == "missing_state_path":
        frame["state_path"] = None
    if block == "changed_energy":
        state["tasks"][0]["outputs"]["energy"] = -2
    if block == "relax_failure":
        state["tasks"][0]["stage"] = "relax_and_feature"
        state["tasks"][0]["outputs"]["optimizer_converged"] = False
    if block in {
        "outside",
        "no_match",
        "no_match_new_rules",
        "no_match_new_structure",
        "no_match_legacy",
        "missing_structure",
    }:
        evidence = record["phase_identification"]
        evidence.pop("phase")
        record["phase"] = None
        if block == "outside":
            evidence.update(
                status="outside_boundary",
                observed_phase="O3",
                reason="phase_outside_search_boundary",
            )
            record.update(status="outside_search_boundary", observed_phase="O3")
        else:
            record["status"] = "pending_phase_identification"
            evidence.update(
                status="unknown",
                reason=(
                    "final_structure_missing"
                    if block == "missing_structure"
                    else "ValueError: unrecognized_phase:X"
                ),
            )
        if block == "no_match_legacy":
            evidence.pop("portable_classification_scope")
        if block == "no_match_new_rules":
            evidence["portable_classification_scope"] = "previous-rules"
        if block == "no_match_new_structure":
            evidence["structure_sha256"] = "previous-digest"
        state["tasks"][0]["outputs"].update(
            actual_phase=None, phase_identification=deepcopy(evidence)
        )
        state["phase_diagram_input_checksum"] = phase_records_checksum(state["phase_records"])
    expected = block in {None, "legacy_valid", "duplicate", "outside", "no_match"}
    assert reuse_exact_after_collection(frame) is expected
    if expected:
        assert frame["idle_exact_feedback_reason"] == "verified_unchanged"
    if block == "missing_structure":
        assert frame["idle_exact_feedback_reason"] == "phase_status_unverified"
    if block == "duplicate":
        assert len(loads) == 1
    if block == "duplicate_changed":
        assert len(loads) == 2


def test_round_reuse_keeps_feedback_cycle_and_incomplete_assessment_rechecks(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.lifecycle_recovery import _assess_workflow_round
    from phase_agent.evidence import post_dft_assessment as module

    calls = []
    monkeypatch.setattr(module, "post_dft_assessment", lambda *a: None)
    monkeypatch.setattr(
        "phase_agent.analysis.convergence.feedback_cycles.record_feedback_cycle",
        lambda state: calls.append("cycle"),
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.lifecycle_recovery._save_runner_state",
        lambda *a, **k: calls.append("save"),
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.export_dft_products.export_dft_products",
        lambda *a: calls.append("export"),
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.refresh_dft_comparisons.refresh_dft_comparisons",
        lambda state, **k: calls.append("compare") or state,
    )
    frame = {
        "feedback": {"state": {}},
        "effective_config": {},
        "manager": None,
        "runtime_adapters": {"final_frame_mlip_evaluator": None},
        "state_path": None,
        "idle_exact_feedback_reused": True,
    }
    _assess_workflow_round(frame)
    assert calls == ["cycle", "save"]
    calls.clear()
    monkeypatch.setattr(
        module, "post_dft_assessment", lambda *a: {"status": "evaluation_incomplete"}
    )
    _assess_workflow_round(frame)
    assert calls == ["compare", "export", "cycle", "save"]


def test_checkpoint_preserves_reuse_and_recovery_facts():
    from phase_agent.graphs.business_checkpoint import checkpoint_frame, restore_frame

    frame = {
        "loaded_state": {},
        "relax_recovery_updated": False,
        "idle_exact_feedback_reused": True,
    }
    saved = checkpoint_frame(frame)
    restored = restore_frame({"loaded_state": {}}, saved)
    assert restored["relax_recovery_updated"] is False
    assert restored["idle_exact_feedback_reused"] is True


def test_round_export_reuse_verifies_sources_and_files(tmp_path):
    from phase_agent.analysis.feedback.export_dft_products import export_dft_products
    from tests.test_dft_comparison_csv import add_result

    state = {}
    add_result(state)
    export_dft_products(state, tmp_path)
    before = deepcopy(state)
    assert export_dft_products(state, tmp_path, check_reuse_only=True) is True
    assert state == before
    state["dft_dataset_records"][0]["energy"] += 1
    assert export_dft_products(state, tmp_path, check_reuse_only=True) is False
    state = before
    Path(next(iter(state["dft_result_exports"].values()))["energy_csv_path"]).unlink()
    assert export_dft_products(state, tmp_path, check_reuse_only=True) is False


def test_reuse_check_does_not_modify_scientific_state(tmp_path, monkeypatch):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.workflows.saved_dft_input_frame import reuse_exact_after_collection

    frame, _, _ = make_frame(tmp_path, monkeypatch)
    state = frame["loaded_state"]
    frame.update(
        pre_reconciled={"state": state, "reconciled": []},
        combined_results=[],
        result_collector=RemoteBatchRunner(tmp_path, worker_command=[]),
        phase_references={},
    )
    before = deepcopy(state)
    assert reuse_exact_after_collection(frame) is True
    assert state == before


def test_dft_records_use_dft_version_not_active_model_version(tmp_path, monkeypatch):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.workflows.saved_dft_input_frame import reuse_exact_after_collection

    frame, _, path = make_frame(tmp_path, monkeypatch)
    state = frame["loaded_state"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    state["tasks"] = [
        {
            "task_id": "t",
            "structure_id": "s",
            "model_version": "m",
            "status": "completed",
            "stage": "dft_single_point",
            "outputs": {"energy": -1},
        }
    ]
    state["phase_records"] = [
        {
            "source_task_id": "t",
            "structure_id": "s",
            "model_version": None,
            "source_version": "atomate",
            "energy_method": "dft",
            "energy": -1,
            "status": "completed",
            "structure_path": str(path),
            "phase_identification": {
                "status": "identified",
                "phase": "O3",
                "method": "test",
                "portable_classification_scope": "rules",
                "structure_sha256": digest,
            },
        }
    ]
    from phase_agent.tools.workflows.apply_scientific_feedback import phase_records_checksum

    record = state["phase_records"][0]
    record.update(
        stage=state["tasks"][0]["stage"], energy_unit="eV", checks_passed=True, phase="O3"
    )
    state["tasks"][0]["outputs"].update(
        energy_unit="eV",
        actual_phase="O3",
        phase_identification=deepcopy(record["phase_identification"]),
    )
    state["phase_diagram_input_checksum"] = phase_records_checksum(state["phase_records"])
    frame.update(
        pre_reconciled={"state": state, "reconciled": []},
        combined_results=[],
        result_collector=RemoteBatchRunner(tmp_path, worker_command=[]),
        phase_references={},
    )
    monkeypatch.setattr(
        "phase_agent.analysis.phase.identify_result_phase._portable_classification_scope",
        lambda *a, **k: "rules",
    )
    monkeypatch.setattr(
        "phase_agent.science.structures.load_result_structure.load_result_structure",
        lambda *a: (None, digest),
    )
    assert reuse_exact_after_collection(frame) is True
    state["tasks"][0]["outputs"]["dft_version"] = "different"
    assert reuse_exact_after_collection(frame) is False


@pytest.mark.parametrize("changed", [False, True])
def test_normal_frozen_approval_identity_after_collection(tmp_path, monkeypatch, changed):
    from phase_agent.evidence.project_authority import project_identity
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    from phase_agent.tools.workflows.saved_dft_input_frame import reuse_exact_after_collection

    frame, _, _ = make_frame(tmp_path, monkeypatch, saved=True)
    state = frame["loaded_state"]
    pending = state["pending_execution_policies"]["p"]
    pending.pop("config_version")
    pending["project_identity"] = project_identity(state)
    if changed:
        state["active_model"] = {"version": "m", "sha256": "different"}
    frame.update(
        pre_reconciled={"state": state, "reconciled": []},
        combined_results=[],
        result_collector=RemoteBatchRunner(tmp_path, worker_command=[]),
    )
    assert reuse_exact_after_collection(frame) is (not changed)
