import json, hashlib, zipfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from ase import Atoms
from ase.io import write
from phase_agent.tools.local.prepare_mc_energy_repair import preview, prepare, attach_preview
from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules


def setup_case(tmp_path):
    tasks = []
    records = []
    structures = {}
    branches = {}
    entries = []
    for i in range(2):
        atoms = Atoms(
            "NaCoO2",
            positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]],
            cell=[5 + i, 5, 5],
            pbc=True,
        )
        p = tmp_path / f"frame-{i}.vasp"
        write(p, atoms, format="vasp")
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        sid = f"S{i}"
        bid = f"B{i}"
        tid = f"MC{i}"
        rid = f"R{i}"
        structures[sid] = {
            "branch_id": bid,
            "source_path": str(p),
            "composition": {"Na": 1, "Co": 1, "O": 2},
        }
        branches[bid] = {"structure_ids": [sid], "P": "O3"}
        tasks.append(
            {
                "task_id": tid,
                "task_key": f"old:{i}",
                "stage": "deep_search",
                "status": "completed",
                "structure_id": sid,
                "branch_id": bid,
                "model_version": "m",
                "outputs": {
                    "energy": -20.0,
                    "energy_per_atom": -5.0,
                    "structure_path": str(p),
                    "structure_checksum": sha,
                    "atom_count": 4,
                    "qbc": {"member_count": 4},
                    "pool_summary": [{"energy_mean_per_atom": -5.0}],
                },
            }
        )
        records.append(
            {
                "record_id": rid,
                "source_task_id": tid,
                "structure_id": sid,
                "model_version": "m",
                "structure_sha256": sha,
                "energy": -20.0,
                "status": "completed",
            }
        )
        entries.append({"record_id": rid})
    manager = SimpleNamespace(
        data={"structures": structures, "branches": branches, "system_config": {}},
        boundary={"P": ["O3"]},
    )
    active = {
        "version": "m",
        "model_path": "/models/main.model",
        "remote_model_path": "/models/main.model",
        "sha256": "a" * 64,
        "members": [
            {"is_main_model": True, "model_path": "/models/main.model", "sha256": "a" * 64}
        ],
    }
    state = {
        "confirmed_config_version": "c",
        "active_model_version": "m",
        "active_model": active,
        "tasks": tasks,
        "pending_tasks": [],
        "phase_records": records,
        "phase_diagrams": {
            "mlip": {
                "version": "h",
                "status": "completed",
                "model_version": "m",
                "entries": entries,
            }
        },
        "dedup_gate": {"status": "ready"},
        "budget_reservations": {},
        "budget_usage": {"stages": {}},
        "reserved_relative_cost": 0.0,
    }
    config = {
        "budgets": default_budget_rules(),
        "upload_batches_directory": str(tmp_path / "submissions"),
        "mlip": {"version": "m", "framework": "mace", "device": "cuda", "mace_head": "omat_pbe"},
        "supercomputer": {
            "worker": {
                "command": [
                    "python3",
                    "--executor",
                    "phase_agent.science.mlip.slurm_executor:execute_mlip_task",
                ]
            },
            "batch_sizes": {"relax_and_feature": 20},
        },
    }
    context = {
        "manager": manager,
        "event_state": state,
        "effective_config": config,
        "config_version": "c",
        "phase_references": {},
        "approval_record_id": "approved",
    }
    action = {
        "tool": "prepare_local_batch_files",
        "target_ids": ["MC0", "MC1"],
        "parameters": {"mode": "mc_energy_repair_inputs"},
        "task_key": "energy-repair",
        "budget": 0,
        "reason": "correct main-model energies",
    }
    return state, config, context, action


def test_preview_checks_scope_hash_model_and_produces_no_tasks(tmp_path):
    state, config, context, action = setup_case(tmp_path)
    saved = deepcopy(state)
    plan = preview(action, state, context["manager"], config, "c")
    assert plan["task_count"] == 2 and plan["relaxation"] is False
    assert len(plan["affected_hull_records"]) == 2 and plan["model"]["sha256"] == "a" * 64
    assert state == saved and not (tmp_path / "submissions").exists()
    broken = deepcopy(action)
    broken["target_ids"] = ["MC0"]
    with pytest.raises(ValueError, match="scope_incomplete"):
        preview(broken, state, context["manager"], config, "c")
    Path(plan["frames"][0]["structure_path"]).write_text("tampered", encoding="utf8")
    with pytest.raises(ValueError, match="frame_binding_invalid"):
        preview(action, state, context["manager"], config, "c")


def test_prepare_requires_frozen_approval_and_packages_only_primary_predictions(tmp_path):
    state, config, context, action = setup_case(tmp_path)
    original = deepcopy(state)
    frozen = attach_preview(action, state, context, config)
    with pytest.raises(ValueError, match="approval_or_preview_missing"):
        prepare(action=frozen, context={**context, "approval_record_id": None})
    changed = deepcopy(frozen)
    changed["parameters"]["energy_repair_preview"]["model"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="preview_changed"):
        prepare(action=changed, context=context)
    result = prepare(action=frozen, context=context)
    assert (
        result["status"] == "prepared"
        and result["task_count"] == 2
        and result["submitted"] is False
    )
    assert state == original
    new = [x for x in result["state"]["tasks"] if x.get("energy_basis_repair_id")]
    assert len(new) == 2 and all(
        x["parameters"]["model_refresh_operation"] == "predict" for x in new
    )
    assert all(x["model_sha256"] == "a" * 64 for x in new)
    handoff = result["handoffs"][0]
    z = zipfile.ZipFile(handoff["upload_archive"])
    paths = [x for x in z.namelist() if x.endswith("/task.json")]
    assert len(paths) == 2
    for path in paths:
        task = json.loads(z.read(path))
        job = task["worker_job"]
        assert job["operation"] == "predict"
        assert job["model_path"] == "/models/main.model" and not job["model_paths"]
        assert job["input_structure_sha256"] == task["input_structure_sha256"]
        assert task["parameters"]["energy_basis_repair"] is True
        folder = path.rsplit("/", 1)[0]
        assert folder + "/_handoff_energy_predict.py" in z.namelist()
        assert "_handoff_energy_predict" in z.read(folder + "/_handoff_refresh_worker.py").decode()
    with pytest.raises(ValueError, match="already_registered"):
        prepare(action=frozen, context={**context, "event_state": result["state"]})


def test_same_frame_prediction_binds_geometry_and_keeps_primary_energy(tmp_path, monkeypatch):
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from phase_agent.science.mlip.predict_structure import predict_structure
    from phase_agent.science.mlip import model_interface

    atoms = Atoms(
        "NaCoO2", positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5], pbc=True
    )
    frame = tmp_path / "input.vasp"
    write(frame, atoms, format="vasp")
    sha = hashlib.sha256(frame.read_bytes()).hexdigest()
    out = tmp_path / "output"
    out.mkdir()

    class FakeCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -20.0, "forces": np.zeros((len(atoms), 3))}

    monkeypatch.setattr(model_interface, "create_calculator", lambda *a, **k: FakeCalculator())
    job = {
        "structure_path": str(frame),
        "input_structure_sha256": sha,
        "output_directory": str(out),
        "model_path": "/models/main.model",
        "model_version": "m",
        "parameters": {
            "energy_basis_repair": True,
            "model_refresh_operation": "predict",
            "main_model_index": 0,
        },
    }
    result = predict_structure(job)
    assert result["energy"] == -20.0 and result["atom_count"] == 4
    assert result["energy_source"] == "main_model_selected_frame"
    assert result["geometry"] == "mc_energy_repair_same_frame"
    assert result["structure_checksum"] == sha
    frame.write_bytes(frame.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="energy_repair_input_checksum_mismatch"):
        predict_structure(job)


def test_frozen_preview_cannot_be_replaced_after_frame_changes(tmp_path):
    state, config, context, action = setup_case(tmp_path)
    frozen = attach_preview(action, state, context, config)
    changed = deepcopy(frozen)
    changed["parameters"]["energy_repair_preview"]["model"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="preview_changed"):
        attach_preview(changed, state, context, config)


def test_repair_same_geometry_lineage_supersedes_legacy_energy_only_after_success():
    from phase_agent.analysis.phase.latest_mlip_lineage import latest_mlip_records
    from phase_agent.analysis.phase.energy_consistency import legacy_mc_mean_record_ids

    state, _, _, _ = setup_case_for_lineage()
    old = state["phase_records"][0]
    old.update(energy_method="mlip", checks_passed=True, composition={"Na": 1, "Co": 1, "O": 2})
    old_task = state["tasks"][0]
    old_task["runtime_observation"] = {"ended_at_epoch": 10.0}
    new = deepcopy(old)
    new.update(record_id="r2", source_task_id="fix", energy=-21.0)
    fix = {
        "task_id": "fix",
        "structure_id": old["structure_id"],
        "model_version": "m",
        "status": "completed",
        "checks_passed": True,
        "input_structure_sha256": old["structure_sha256"],
        "parameters": {"model_refresh_operation": "predict", "energy_basis_repair": True},
        "runtime_observation": {"ended_at_epoch": 20.0},
        "outputs": {"energy_source": "main_model_selected_frame"},
    }
    selected, issues = latest_mlip_records([old, new], [old_task, fix], {"m": 1})
    assert not issues and [r["record_id"] for r in selected] == ["r2"]
    state["phase_diagrams"]["mlip"]["entries"] = [{"record_id": old["record_id"]}]
    assert legacy_mc_mean_record_ids(state, state["phase_diagrams"]["mlip"]) == [old["record_id"]]
    state["phase_records"].append(new)
    state["tasks"].append(fix)
    state["phase_diagrams"]["mlip"]["entries"] = [{"record_id": "r2"}]
    assert legacy_mc_mean_record_ids(state, state["phase_diagrams"]["mlip"]) == []


def setup_case_for_lineage():
    sha = "b" * 64
    old = {
        "record_id": "r1",
        "source_task_id": "mc",
        "structure_id": "s",
        "structure_sha256": sha,
        "model_version": "m",
        "source_version": "m",
        "energy": -20.0,
        "status": "completed",
    }
    task = {
        "task_id": "mc",
        "structure_id": "s",
        "model_version": "m",
        "stage": "deep_search",
        "status": "completed",
        "outputs": {
            "energy": -20.0,
            "energy_per_atom": -5.0,
            "qbc": {"member_count": 4},
            "pool_summary": [{"energy_mean_per_atom": -5.0}],
        },
    }
    return (
        {"phase_records": [old], "tasks": [task], "phase_diagrams": {"mlip": {"entries": []}}},
        None,
        None,
        None,
    )


def test_cross_model_preview_binds_history_and_rejects_mismatches(tmp_path):
    state, config, context, action = setup_case(tmp_path)
    state["upload_layout"] = {"model_rounds": {"old": 1, "m": 2}}
    for task, record in zip(state["tasks"], state["phase_records"]):
        task["model_version"] = "old"
        record["model_version"] = "old"
    plan = preview(action, state, context["manager"], config, "c")
    assert plan["schema"] == 2
    assert {f["source_model_version"] for f in plan["frames"]} == {"old"}
    assert plan["model"]["version"] == "m"
    broken = deepcopy(state)
    broken["phase_records"][0]["model_version"] = "m"
    with pytest.raises(ValueError, match="source_model_binding"):
        preview(action, broken, context["manager"], config, "c")
    broken = deepcopy(state)
    broken["upload_layout"]["model_rounds"]["old"] = 3
    with pytest.raises(ValueError, match="source_model_order"):
        preview(action, broken, context["manager"], config, "c")


def test_cross_model_prepare_preserves_old_tasks_and_binds_source(tmp_path):
    state, config, context, action = setup_case(tmp_path)
    state["upload_layout"] = {"model_rounds": {"old": 1, "m": 2}}
    for task, record in zip(state["tasks"], state["phase_records"]):
        task["model_version"] = "old"
        record["model_version"] = "old"
    frozen = attach_preview(action, state, context, config)
    result = prepare(action=frozen, context=context)
    assert result["status"] == "prepared"
    assert all(t["model_version"] == "old" for t in result["state"]["tasks"][:2])
    repairs = [t for t in result["state"]["tasks"] if t.get("energy_basis_repair_id")]
    assert len(repairs) == 2
    assert all(t["energy_repair_source_model_version"] == "old" and t["model_version"] == "m" for t in repairs)
