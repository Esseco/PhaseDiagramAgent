"""Portable composition tests: fake scientific executors/Slurm only, no real jobs."""

from copy import deepcopy
from pathlib import Path
import json
import shutil
import zipfile
import pytest
from pymatgen.core import Structure, Lattice

from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.tools.relax_mc_batch.plan import attach_preview, diagnose
from phase_agent.tools.relax_mc_batch.prepare import prepare_batch
from phase_agent.tools.relax_mc_batch.run import run_batch
from phase_agent.tools.relax_mc_batch.results import import_results, inspect_archive
from phase_agent.tools.relax_mc_batch.files import read, write, rebase
from phase_agent.tools.remote.integrity import file_checksum


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Real deployment uses a short HPC root. Avoid pytest's long function names
    # pushing existing nested worker artifacts above Windows' 260-char limit.
    return tmp_path_factory.mktemp("r")


def fixture(root, count=12, single_x=False):
    root.mkdir(parents=True, exist_ok=True)
    manager = PhaseDataManager({"P": ["O3"], "H": {"max_det": 100}, "TM_ratio": {"Fe": 1}})
    for i in range(count):
        # Small physical files, deliberately fake model calculations below.
        n_na = 1 if single_x or i % 2 else 2
        species = ["Fe", "Fe", "O", "O", "O", "O"] + ["Na"] * n_na
        structure = Structure(
            Lattice.cubic(10), species, [[0.1 + i * 0.08, 0.2, 0.3] for i in range(len(species))]
        )
        path = root / f"s{i}.vasp"
        structure.to(filename=path, fmt="poscar")
        full = root / f"full{i}.vasp"
        structure.to(filename=full, fmt="poscar")
        bid = manager.add_branch(
            P="O3",
            H=[[i + 1, 0, 0], [0, 1, 0], [0, 0, 1]],
            x=n_na / 2,
            T=["Fe"] * 2,
            composition=structure.composition.as_dict(),
        )
        sid = manager.add_structure(
            branch_id=bid,
            arrangement={"i": i},
            composition=structure.composition.as_dict(),
            source_path=path,
            metadata={"initialization_method": "electrostatic_top10_random3_layer_occupied"},
        )
        manager.data["structures"][sid]["full_na_structure_path"] = str(full)
    model = root / "fake.model"
    model.write_text("fake weights; no MACE load")
    config = default_layered_search_config()
    config["mlip"].update(name="m1", version="m1", model_path=str(model))
    config["upload_batches_directory"] = str(root / "uploads")
    config["bohb"]["relax_structures_per_branch"] = 1
    config.pop("ledger_path", None)
    config["supercomputer"]["batch_sizes"].update(relax_and_feature=5, deep_search=5)
    state = {
        "active_model_version": "m1",
        "confirmed_config_version": "v1",
        "dedup_gate": {"status": "ready"},
        "tasks": [],
    }
    context = {
        "effective_config": config,
        "manager": manager,
        "event_state": state,
        "config_version": "v1",
        "phase_references": {},
    }
    action = {
        "tool": "prepare_local_batch_files",
        "operation": "prepare_relax_mc_batch",
        "task_key": "pack",
        "target_ids": list(manager.data["branches"]),
        "parameters": {"mode": "relax_mc_inputs"},
        "reason": "user requested optional batch",
        "budget": 0,
    }
    return manager, state, config, context, action


def package(root, **kwargs):
    manager, state, config, context, action = fixture(root, **kwargs)
    action = attach_preview(action, state, context, config)
    result = prepare_batch(action=action, context=context)
    assert result["status"] == "prepared" and result["submitted"] is False
    return manager, state, config, context, action, result


def fake_executor(monkeypatch, *, fail_relax=False, fail_mc=False):
    from phase_agent.analysis.phase import ensure_phase_identification as phase_module

    def identify(task, manager, **kwargs):
        task = deepcopy(task)
        task.update(actual_phase="O3", phase_identification_status="identified")
        task["outputs"].update(
            actual_phase="O3", phase_identification={"status": "identified", "phase": "O3"}
        )
        return task

    monkeypatch.setattr(phase_module, "identify_result_phase", identify)
    from phase_agent.analysis.phase import build_composition_hull_entries as hull_module

    monkeypatch.setattr(
        hull_module,
        "check_na_layer_uniformity",
        lambda *a, **kw: {"uniform": True, "status": "uniform", "rule": "fake-test"},
    )
    calls = []

    def execute(batch, state, manager):
        results = []
        for tid in batch["task_ids"]:
            task = next(t for t in state["tasks"] if t["task_id"] == tid)
            worker = read(task["input_path"])
            task = {
                **task,
                **{key: worker[key] for key in ("protocol_version", "input_file_version")},
            }
            calls.append((task["stage"], task.get("segment_index"), task.get("max_mc_steps")))
            output_dir = Path(task["result_path"]).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            source = Path(task["structure_path"])
            geometry = output_dir / "final.vasp"
            shutil.copy2(source, geometry)
            composition = Structure.from_file(geometry).composition.as_dict()
            atoms = sum(composition.values())
            steps = task.get("max_mc_steps")
            segment = task.get("segment_index", -1)
            energy_per_atom = -5 - (segment + 1) * 0.006
            payload = {
                **deepcopy(task),
                "status": "failed"
                if (fail_relax and task["stage"] == "relax_and_feature")
                or (fail_mc and task["stage"] == "deep_search")
                else "completed",
                "result_path": str(geometry),
                "converged": True,
                "actual_mc_steps": steps,
                "stop_reason": "max_steps",
                "outputs": {
                    "structure_path": str(geometry),
                    "energy": energy_per_atom * atoms,
                    "energy_per_atom": energy_per_atom,
                    "energy_unit": "eV",
                    "composition": composition,
                    "relax_stopped_normally": True,
                    "mlip_version": "m1",
                    "qbc": {"status": "completed", "energy_per_atom_std": 0.01},
                },
            }
            write(output_dir / "result.json", payload)
            marker = {
                k: payload.get(k)
                for k in (
                    "task_id",
                    "task_key",
                    "batch_id",
                    "config_version",
                    "model_version",
                    "task_checksum",
                    "protocol_version",
                    "input_file_version",
                    "status",
                )
            }
            marker["result_checksum"] = file_checksum(output_dir / "result.json")
            write(output_dir / "task.finished.json", marker)
            results.append(payload)
        return results

    return execute, calls


def test_preview_diagnosis_workload_and_complete_portable_inventory(tmp_path):
    manager, state, config, context, action, result = package(tmp_path)
    plan = action["parameters"]["relax_mc_preview"]
    assert plan["bootstrap_first_mc"] is True
    assert len(plan["diagnosis"]["Na_contents"]) == 2
    assert plan["workload"]["relax_structure_tasks"] == 12
    assert plan["workload"]["gpu_jobs_upper_by_wave"] == [3, 3, 3, 3]
    assert plan["workload"]["mc_step_upper"] == 6000
    assert plan["workload"]["max_concurrent_jobs"] == 1
    with zipfile.ZipFile(result["upload_archive"]) as archive:
        names = archive.namelist()
        assert "batch.json" in names and "submit_all.sh" in names
        assert any(p.startswith("runtime/phase_agent/science/mc/") for p in names)
        assert any(p.startswith("runtime/mc_backend/Process_AL_MC/") for p in names)
        assert sum(p.startswith("inputs/") for p in names) == 24
        assert not any(p.endswith(".model") for p in names)
    script = Path(result["script_path"]).read_text()
    assert script.index("#SBATCH") < script.index("set -e")
    assert "--gres=gpu:1" in script and "--partition=v100m3" in script
    assert state["tasks"] == [] and len(result["state"]["tasks"]) == 12


def test_missing_files_duplicate_tasks_and_stale_approval_are_blocked(tmp_path):
    manager, state, config, context, action = fixture(tmp_path)
    action = attach_preview(action, state, context, config)
    source = Path(next(iter(manager.data["structures"].values()))["source_path"])
    source.write_text(source.read_text() + "\n")
    assert prepare_batch(action=action, context=context)["status"] == "awaiting_approval"
    assert not (tmp_path / "uploads").exists()
    source.unlink()
    with pytest.raises(ValueError, match="missing_structure"):
        diagnose(action["target_ids"], state, manager, config, "v1")


def test_no_hull_bootstrap_then_three_waves_and_safe_resume(tmp_path, monkeypatch):
    manager, state, config, context, action, result = package(tmp_path)
    execute, calls = fake_executor(monkeypatch)
    outcome = run_batch(result["directory"], submit=True, batch_executor=execute)
    assert outcome["status"] == "completed", outcome
    assert len(calls) == 48
    assert {step for stage, index, step in calls if stage == "deep_search" and index == 0} == {100}
    assert {step for stage, index, step in calls if stage == "deep_search" and index in (1, 2)} == {
        200
    }
    report = inspect_archive(outcome["archive"])
    assert report["summary"]["completed_mc_waves"] == [0, 1, 2]
    assert report["summary"]["gpu_job_count"] == 12
    assert report["candidate_pool"]["branch_candidates"]
    remote = read(Path(result["directory"]) / "workflow_state.json")
    assert remote["phase_diagrams"]["mlip"]["status"] == "completed"
    assert all(
        r["relaxed_ehull"] is None
        for r in remote["tasks"]
        if r.get("stage") == "deep_search" and r["segment_index"] == 0
    )
    rerun = run_batch(result["directory"], submit=True, batch_executor=execute)
    assert rerun["status"] == "completed" and len(calls) == 48


def test_single_composition_bootstraps_once_then_pauses_with_results(tmp_path, monkeypatch):
    *_, result = package(tmp_path, single_x=True)
    execute, calls = fake_executor(monkeypatch)
    outcome = run_batch(result["directory"], submit=True, batch_executor=execute)
    assert outcome["status"] == "paused", outcome
    assert len(calls) == 24
    report = inspect_archive(outcome["archive"])
    assert report["summary"]["completed_mc_waves"] == [0]
    assert report["summary"]["reason"] == "insufficient_Na_endpoints"


def test_remote_dry_run_never_executes_or_submits(tmp_path, monkeypatch):
    *_, result = package(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run executed a task")

    import subprocess

    monkeypatch.setattr(subprocess, "run", forbidden)
    dry = run_batch(result["directory"], submit=True, dry_run=True, batch_executor=forbidden)
    assert dry["status"] == "planned_only" and dry["submitted"] is False
    assert not (Path(result["directory"]) / "workflow_state.json").exists()


def test_model_missing_and_input_corruption_submit_no_children(tmp_path, monkeypatch):
    *_, result = package(tmp_path)
    Path(tmp_path / "fake.model").unlink()
    execute, calls = fake_executor(monkeypatch)
    outcome = run_batch(result["directory"], submit=True, batch_executor=execute)
    assert outcome["status"] == "failed" and calls == []
    assert "remote_model_not_found" in outcome["reason"]
    initial = next((Path(result["directory"]) / "inputs").glob("*.vasp"))
    initial.write_text(initial.read_text() + "\n")
    with pytest.raises(ValueError, match="input_package_integrity_failed"):
        run_batch(result["directory"], submit=True, batch_executor=execute)
    assert calls == []


def test_return_import_rebases_results_preserves_ledger_and_is_idempotent(tmp_path, monkeypatch):
    manager, state, config, context, action, result = package(tmp_path)
    execute, calls = fake_executor(monkeypatch)
    outcome = run_batch(result["directory"], submit=True, batch_executor=execute)
    returned = Path(result["expected_results_archive"])
    returned.parent.mkdir()
    shutil.copy2(outcome["archive"], returned)
    imported = import_results(
        action={"parameters": {"bundle_id": result["bundle_id"]}},
        context={**context, "event_state": result["state"]},
    )
    assert imported["status"] == "completed"
    assert len(imported["state"]["tasks"]) == 48
    assert imported["state"]["phase_diagrams"]["mlip"]["status"] == "completed"
    assert imported["state"]["qbc_candidates"]
    assert all(Path(r["structure_path"]).is_file() for r in imported["state"]["qbc_candidates"])
    assert len(manager.data["branches"]) == 12 and len(manager.data["structures"]) == 12
    repeated = import_results(
        action={"parameters": {"bundle_id": result["bundle_id"]}},
        context={**context, "event_state": imported["state"]},
    )
    assert repeated["status"] == "already_imported"


def test_prior_model_hull_is_carried_and_model_mismatch_is_rejected(tmp_path):
    manager, state, config, context, action = fixture(tmp_path)
    sid = next(iter(manager.data["structures"]))
    record = {
        "structure_id": sid,
        "structure_path": manager.data["structures"][sid]["source_path"],
        "energy_method": "mlip",
        "model_version": "m1",
        "status": "completed",
        "phase_identification_status": "identified",
        "energy": -40,
        "energy_unit": "eV",
        "composition": manager.data["structures"][sid]["composition"],
        "phase": "O3",
    }
    state["phase_records"] = [record]
    plan = diagnose(action["target_ids"], state, manager, config, "v1")
    assert not plan["bootstrap_first_mc"] and plan["history_records"] == [record]
    assert any(row["kind"] == "history_geometry" for row in plan["inputs"])
    state["phase_diagrams"] = {"mlip": {"entries": [record], "model_version": "old"}}
    with pytest.raises(ValueError, match="existing_phase_diagram_model_mismatch"):
        diagnose(action["target_ids"], state, manager, config, "v1")


def test_failure_does_not_overwrite_submission_receipt_checkpoint(tmp_path, monkeypatch):
    *_, result = package(tmp_path)
    from phase_agent.tools.relax_mc_batch import run as module

    def uncertain(root, state, manager, *args, **kwargs):
        current = deepcopy(state)
        current["slurm_batches"] = [
            {"batch_id": "b", "submission_state": "submitting", "task_ids": []}
        ]
        module.checkpoint(root, current, manager)
        raise RuntimeError("submission_receipt_uncertain")

    monkeypatch.setattr(module, "wave", uncertain)
    outcome = run_batch(result["directory"], submit=True)
    assert outcome["status"] == "failed"
    saved = read(Path(result["directory"]) / "workflow_state.json")
    assert saved["slurm_batches"][0]["submission_state"] == "submitting"


@pytest.mark.parametrize("mode", ["Sequential", "Adaptive"])
def test_combined_operation_both_modes_dry_run_creates_no_tasks(mode, tmp_path, monkeypatch):
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.tools.workflows.run_event_loop import run_event_loop
    import subprocess

    manager, state, config, context, action = fixture(tmp_path)
    state["search_mode"] = mode
    before = deepcopy(state)
    calls = []

    def forbidden(*a, **kw):
        calls.append("execution")
        raise AssertionError("dry-run executed")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    plan = {
        "config_version": "v1",
        "model_version": "m1",
        "reason": "batch dry-run",
        "actions": [{"id": "batch", "kind": "Relax", "depends_on": [], "action": action}],
    }
    registry = create_tool_registry({name: forbidden for name in create_tool_registry()})
    result = run_event_loop(
        state,
        {"status": "confirmed", "confirmed_snapshot": {"config_version": "v1", "config": config}},
        registry=registry,
        context={"manager": manager},
        execution_mode="dry_run",
        max_steps=1,
        agent_client=lambda request: deepcopy(
            plan if request.get("mode") == "adaptive_plan" else action
        ),
        state_path=tmp_path / "dry-state.json",
    )
    assert result["status"] == "planned_only", result
    assert not calls and state == before and not result["state"].get("tasks")
    assert not result["state"].get("budget_reservations") and not result["state"].get(
        "slurm_batches"
    )
    assert not (tmp_path / "uploads").exists()
    write(
        tmp_path / f"{mode}-batch-dry-run.json",
        {"mode": mode, "status": result["status"], "execution_calls": 0, "new_tasks": 0},
    )


def test_submission_collection_loop_with_fake_slurm_only(tmp_path, monkeypatch):
    manager, state, config, context, action, result = package(tmp_path)
    root = Path(result["directory"])
    execute, calls = fake_executor(monkeypatch)
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("real submission"))

    class FakeSlurm:
        submissions = []

        def active(self, ids):
            return set()

        def submit(self, path):
            remote = rebase(read(root / "workflow_state.json"), root)
            batch = next(b for b in remote["slurm_batches"] if b["script_path"] == str(path))
            execute(batch, remote, manager)
            self.submissions.append(str(path))
            return str(100 + len(self.submissions))

    io = FakeSlurm()
    outcome = run_batch(root, submit=True, io=io, poll_seconds=0)
    assert outcome["status"] == "completed", outcome
    assert len(calls) == 48 and len(io.submissions) == 12
    assert len(set(io.submissions)) == 12
    assert run_batch(root, submit=True, io=io, poll_seconds=0)["submitted"] is False
    assert len(io.submissions) == 12


def test_corrupt_return_is_reported_without_advancing_pending_tasks(tmp_path):
    from phase_agent.tools.relax_mc_batch.results import recover_available

    archive = tmp_path / "results.zip"
    archive.write_bytes(b"incomplete zip")
    state = {
        "active_model_version": "m1",
        "tasks": [{"task_id": "t", "status": "pending"}],
        "relax_mc_batches": {
            "b": {
                "expected_results_archive": str(archive),
                "model_version": "m1",
                "config_version": "v1",
            }
        },
    }
    result = recover_available(
        state,
        {
            "effective_config": {
                "mlip": {"version": "m1"},
                "upload_batches_directory": str(tmp_path),
            },
            "config_version": "v1",
        },
    )
    assert result["tasks"] == state["tasks"] and "return_error" in result["relax_mc_batches"]["b"]
