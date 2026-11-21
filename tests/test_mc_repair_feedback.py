import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

from phase_agent.persistence.ledger.mc_energy_repair import (
    missing_verified_repair_tasks,
    verified_mc_energy_repair,
)
from phase_agent.persistence.ledger.collect_calculation_results import collect_calculation_results
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager


def build_case(tmp_path):
    H = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    manager = PhaseDataManager({"P": ["O3"], "H": {"O3": [H]}, "TM_ratio": {"Co": 1}})
    branch = manager.add_branch(P="O3", H=H, x=1, T=["Co"], composition={"Na": 1, "Co": 1, "O": 2})
    sid = manager.add_structure(branch_id=branch, arrangement={"V": [1]})
    path = tmp_path / "same.vasp"
    write(
        path,
        Atoms(
            "NaCoO2",
            positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        format="vasp",
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    model = "a" * 64
    task = {
        "task_id": "fix",
        "task_key": "fix-key",
        "stage": "relax_and_feature",
        "status": "completed",
        "structure_id": sid,
        "branch_id": branch,
        "converged": None,
        "model_version": "m1",
        "model_sha256": model,
        "input_structure_sha256": sha,
        "energy_basis_repair_id": "repair",
        "energy_repair_source_task_id": "old",
        "energy_repair_source_record_id": "old-record",
        "parent_decision_id": "approved",
        "parameters": {"energy_basis_repair": True, "model_refresh_operation": "predict"},
        "outputs": {
            "energy": -20.0,
            "energy_unit": "eV",
            "energy_source": "main_model_selected_frame",
            "geometry": "mc_energy_repair_same_frame",
            "structure_path": str(path),
            "structure_checksum": sha,
            "verified_model_sha256": model,
            "single_point_completed": True,
            "composition": {"Na": 1, "Co": 1, "O": 2},
            "actual_phase": "O3",
            "phase_identification": {"status": "identified", "structure_sha256": sha},
        },
    }
    old = {
        "record_id": "old-record",
        "source_task_id": "old",
        "structure_id": sid,
        "structure_sha256": sha,
        "energy_method": "mlip",
        "status": "completed",
        "model_version": "m1",
    }
    frame = {
        "source_task_id": "old",
        "record_id": "old-record",
        "structure_id": sid,
        "structure_sha256": sha,
    }
    state = {
        "tasks": [task],
        "phase_records": [old],
        "feedback_processed_task_ids": ["fix"],
        "mc_energy_repairs": {
            "repair": {
                "approval_record_id": "approved",
                "task_ids": ["fix"],
                "preview": {
                    "checksum": "repair",
                    "model": {"version": "m1", "sha256": model},
                    "frames": [frame],
                },
            }
        },
    }
    return manager, task, state


def test_same_frame_repair_enters_phase_records_without_claiming_relaxation(tmp_path):
    manager, task, _ = build_case(tmp_path)
    assert verified_mc_energy_repair(task)
    first = collect_calculation_results(manager, task["structure_id"], task)
    assert first["phase_record"]["energy"] == -20 and first["phase_record"]["converged"] is None
    second = collect_calculation_results(manager, task["structure_id"], task, refresh_existing=True)
    assert second["result_id"] == first["result_id"]
    assert (
        len(manager.data["structures"][task["structure_id"]]["stage_history"]["relax_and_feature"])
        == 1
    )


def test_missing_feedback_only_recovers_bound_untampered_results(tmp_path):
    _, task, state = build_case(tmp_path)
    assert [r["task_id"] for r in missing_verified_repair_tasks(state)] == ["fix"]
    for key in ("verified_model_sha256", "structure_checksum", "energy_source"):
        bad = deepcopy(state)
        bad["tasks"][0]["outputs"][key] = "wrong"
        assert not missing_verified_repair_tasks(bad)
    bad = deepcopy(state)
    bad["mc_energy_repairs"]["repair"]["preview"]["frames"][0]["record_id"] = "wrong"
    assert not missing_verified_repair_tasks(bad)
    Path(task["outputs"]["structure_path"]).write_bytes(b"changed")
    assert not missing_verified_repair_tasks(state)


def test_feedback_replays_missed_repair_without_duplicate_cost_or_ledger(tmp_path, monkeypatch):
    from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback

    manager, task, state = build_case(tmp_path)
    original = collect_calculation_results(manager, task["structure_id"], task)
    state.update(
        processed_task_ids=["fix"],
        phase_diagrams={"mlip": {"version": "old"}},
        phase_diagram_input_checksum="old",
        active_model_version="m1",
    )
    monkeypatch.setattr("phase_agent.persistence.shared_dft.publish", lambda *a, **k: None)
    monkeypatch.setattr("phase_agent.persistence.shared_dft.import_records", lambda *a, **k: None)
    monkeypatch.setattr(
        "phase_agent.tools.workflows.apply_scientific_feedback.ensure_phase_identification",
        lambda state, *a, **k: (state, {}),
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.apply_scientific_feedback.refresh_identified_phases",
        lambda state: None,
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.dft_result_products.record_dft_products",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.dft_result_products.export_dft_products",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "phase_agent.analysis.feedback.export_dft_products.publish_output_catalog",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.apply_scientific_feedback.coverage", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.apply_scientific_feedback.summarize_agent_state",
        lambda *a, **k: {},
    )
    monkeypatch.setattr(
        "phase_agent.tools.workflows.apply_scientific_feedback.update_phase_diagram",
        lambda records, **k: {"diagrams": {"mlip": {"version": "new"}}, "mlip_by_version": {}},
    )
    result = apply_scientific_feedback(state, [task], manager=manager, active_model_version="m1")
    recovered = result["state"]
    records = [r for r in recovered["phase_records"] if r.get("source_task_id") == "fix"]
    assert len(records) == 1 and records[0]["record_id"] == original["result_id"]
    assert recovered["phase_diagrams"]["mlip"]["version"] == "new"
    assert recovered["feedback_processed_task_ids"].count("fix") == 1
    assert (
        len(manager.data["structures"][task["structure_id"]]["stage_history"]["relax_and_feature"])
        == 1
    )
    assert missing_verified_repair_tasks(recovered) == []


def test_model_hull_replaces_only_verified_same_frame_mean_energy(tmp_path, monkeypatch):
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    manager, task, state = build_case(tmp_path)
    new = collect_calculation_results(manager, task["structure_id"], task)["phase_record"]
    old = state["phase_records"][0]
    old.update(energy=-19, energy_unit="eV", composition={"Na": 1, "Co": 1, "O": 2}, phase="O3")
    records = [old, new]

    def snapshot(method, normalized, rejected, *args, **kwargs):
        return {
            "method": method,
            "status": "completed",
            "version": "v",
            "entries": normalized,
            "rejected": rejected,
        }

    monkeypatch.setattr("phase_agent.analysis.phase.update_phase_diagram._build_snapshot", snapshot)
    result = update_phase_diagram(records, output_state=state, active_model_version="m1")
    assert [r["record_id"] for r in result["mlip_by_version"]["m1"]["entries"]] == [
        new["record_id"]
    ]
    bad = deepcopy(state)
    bad["tasks"][0]["outputs"]["verified_model_sha256"] = "wrong"
    result = update_phase_diagram(records, output_state=bad, active_model_version="m1")
    assert {r["record_id"] for r in result["mlip_by_version"]["m1"]["entries"]} == {
        old["record_id"],
        new["record_id"],
    }


def test_model_only_rebuild_excludes_other_models_and_does_not_replace_current_csv(
    tmp_path, monkeypatch
):
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    manager, task, state = build_case(tmp_path)
    new = collect_calculation_results(manager, task["structure_id"], task)["phase_record"]
    old = state["phase_records"][0]
    old.update(energy=-19, energy_unit="eV", composition={"Na": 1, "Co": 1, "O": 2}, phase="O3")
    other = {**old, "record_id": "other", "model_version": "m0", "source_version": "m0"}
    state.update(
        mlip_energy_policy="latest_valid_per_structure_v1",
        upload_layout={"model_rounds": {"m0": 1, "m1": 2}},
    )
    built = []

    def snapshot(method, normalized, rejected, *args, **kwargs):
        built.append([r["record_id"] for r in normalized])
        return {
            "method": method,
            "status": "completed",
            "version": "v",
            "entries": normalized,
            "rejected": rejected,
        }

    def export(snapshot, path):
        Path(path).write_text("model-only", encoding="utf8")

    monkeypatch.setattr("phase_agent.analysis.phase.update_phase_diagram._build_snapshot", snapshot)
    monkeypatch.setattr(
        "phase_agent.analysis.phase.update_phase_diagram.phase_snapshot_directory",
        lambda root, method, version, state: Path(root) / version / "history",
    )
    monkeypatch.setattr(
        "phase_agent.analysis.phase.update_phase_diagram.export_phase_diagram_csv", export
    )
    monkeypatch.setattr(
        "phase_agent.analysis.phase.publish_current_csv.publish_current_csv",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("current CSV must not be replaced")),
    )
    result = update_phase_diagram(
        [old, new, other],
        output_state=state,
        active_model_version="m1",
        output_directory=tmp_path,
        model_versions_only=["m1"],
    )
    assert built == [[new["record_id"]]]
    assert set(result["mlip_by_version"]) == {"m1"} and result["diagrams"] == {}
    with pytest.raises(ValueError, match="scope_missing"):
        update_phase_diagram([old, new, other], output_state=state, model_versions_only=["missing"])


def test_cross_model_feedback_requires_verified_source_lineage(tmp_path):
    _, task, state = build_case(tmp_path)
    from phase_agent.tools.remote.integrity import payload_checksum
    old = {"task_id": "old", "model_version": "m0", "stage": "deep_search", "status": "completed",
           "structure_id": task["structure_id"],
           "outputs": {"structure_checksum": task["input_structure_sha256"]}}
    state["tasks"].append(old)
    state["phase_records"][0]["model_version"] = "m0"
    repair = state["mc_energy_repairs"].pop("repair")
    preview = repair["preview"]
    preview["schema"] = 2
    preview["frames"][0]["source_model_version"] = "m0"
    preview["checksum"] = payload_checksum({k:v for k,v in preview.items() if k != "checksum"})
    task["energy_basis_repair_id"] = preview["checksum"]
    task["energy_repair_source_model_version"] = "m0"
    state["mc_energy_repairs"][preview["checksum"]] = repair
    assert [r["task_id"] for r in missing_verified_repair_tasks(state)] == ["fix"]
    for target, key in [(task, "energy_repair_source_model_version"), (old, "model_version"),
                        (state["phase_records"][0], "model_version"),
                        (preview["frames"][0], "source_model_version")]:
        original = target[key]
        target[key] = "mismatched"
        assert not missing_verified_repair_tasks(state)
        target[key] = original


def test_cross_model_binding_requires_frozen_preview_and_task_identity(tmp_path):
    from phase_agent.persistence.ledger.mc_energy_repair import cross_model_repair_binding_valid
    from phase_agent.tools.remote.integrity import payload_checksum
    _, task, state = build_case(tmp_path)
    old = {"task_id": "old", "model_version": "m0", "stage": "deep_search", "status": "completed",
           "structure_id": task["structure_id"],
           "outputs": {"structure_checksum": task["input_structure_sha256"]}}
    state["tasks"].append(old)
    state["phase_records"][0]["model_version"] = "m0"
    repair = state["mc_energy_repairs"].pop("repair")
    preview = repair["preview"]
    preview["schema"] = 2
    preview["frames"][0]["source_model_version"] = "m0"
    preview["checksum"] = payload_checksum({k:v for k,v in preview.items() if k != "checksum"})
    task["energy_basis_repair_id"] = preview["checksum"]
    task["energy_repair_source_model_version"] = "m0"
    state["mc_energy_repairs"][preview["checksum"]] = repair
    assert cross_model_repair_binding_valid(state, task)
    for target, key in [(task, "model_sha256"), (old, "model_version"),
                        (state["phase_records"][0], "model_version"),
                        (preview["frames"][0], "source_model_version")]:
        original = target[key]
        target[key] = "tampered"
        assert not cross_model_repair_binding_valid(state, task)
        target[key] = original


def test_cross_model_result_cannot_override_frozen_identity(tmp_path):
    from phase_agent.persistence.ledger.result_validation import validate_result_binding, InvalidScientificResult
    _, task, _ = build_case(tmp_path)
    task["energy_repair_source_model_version"] = "m0"
    for key in ("energy_basis_repair_id", "energy_repair_source_task_id",
                "energy_repair_source_record_id", "energy_repair_source_model_version",
                "input_structure_sha256", "model_sha256", "parent_decision_id"):
        with pytest.raises(InvalidScientificResult, match="result_binding_mismatch"):
            validate_result_binding({"task_id": task["task_id"], key: None}, task)


def test_cross_model_repair_latest_hull_keeps_historical_model_snapshot(tmp_path, monkeypatch):
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
    manager, task, state = build_case(tmp_path)
    new = collect_calculation_results(manager, task["structure_id"], task)["phase_record"]
    old = state["phase_records"][0]
    old.update(energy=-19, energy_unit="eV", composition={"Na": 1, "Co": 1, "O": 2},
               phase="O3", model_version="m0", source_version="m0",
               structure_path=task["outputs"]["structure_path"])
    state.update(mlip_energy_policy="latest_valid_per_structure_v1",
                 upload_layout={"model_rounds": {"m0": 1, "m1": 2}})
    def snapshot(method, normalized, rejected, *args, **kwargs):
        return {"method": method, "status": "completed", "version": "v",
                "entries": normalized, "rejected": rejected}
    monkeypatch.setattr("phase_agent.analysis.phase.update_phase_diagram._build_snapshot", snapshot)
    result = update_phase_diagram([old, new], output_state=state, active_model_version="m1")
    assert [r["record_id"] for r in result["mlip_by_version"]["m0"]["entries"]] == ["old-record"]
    assert [r["record_id"] for r in result["diagrams"]["mlip"]["entries"]] == [new["record_id"]]
    assert old["model_version"] == "m0" and old["energy"] == -19
