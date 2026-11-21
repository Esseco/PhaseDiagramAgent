from copy import deepcopy
import json
from types import SimpleNamespace

from ase import Atoms
from ase.io import write

from phase_agent.configuration.defaults.default_budget_rules import default_budget_rules
from phase_agent.persistence.models.require_structure_refresh import require_structure_refresh
from phase_agent.tools.local.prepare_local_batch_files import prepare_local_batch_files
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate, advance_refresh
from phase_agent.tools.workflows.run_tool_step import run_tool_step


def setup(tmp_path):
    path = tmp_path / "original.vasp"
    write(
        path,
        Atoms(
            "NaFeO2",
            positions=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        format="vasp",
    )
    manager = SimpleNamespace(
        data={
            "structures": {
                "s": {
                    "branch_id": "b",
                    "source_path": str(path),
                    "composition": {"Na": 1, "Fe": 1, "O": 2},
                }
            },
            "branches": {"b": {"structure_ids": ["s"]}},
        },
        boundary={},
    )
    config = {
        "budgets": default_budget_rules(),
        "mlip": {"version": "old", "model_path": "/models/old.model"},
        "upload_batches_directory": str(tmp_path / "submissions"),
        "supercomputer": {
            "worker": {
                "command": [
                    "python",
                    "-m",
                    "phase_agent.tools.remote.run_slurm_array_task",
                    "--executor",
                    "phase_agent.science.mlip.slurm_executor:execute_mlip_task",
                ]
            }
        },
    }
    state = {
        "tasks": [],
        "active_model": {"version": "new", "model_path": "/models/new.model"},
        "active_model_version": "new",
        "dedup_gate": {"status": "ready", "valid_structure_ids": ["s"]},
        "phase_diagrams": {
            "mlip": {
                "status": "completed",
                "model_version": "old",
                "entries": [
                    {
                        "structure_id": "s",
                        "structure_path": str(path),
                        "ehull": 0.005,
                        "ehull_unit": "eV/atom",
                        "phase": "O3",
                    }
                ],
            }
        },
    }
    require_structure_refresh(state, "old", "new")
    # Historical single-structure checkpoint: no phase-record hull was saved.
    # These tests exercise legacy approval and recovery, not the new HD policy.
    state.pop("mlip_energy_policy", None)
    session = {
        "status": "confirmed",
        "confirmed_snapshot": {
            "config_version": "c",
            "config": {**config, "agent": {"allowed_tools": ["prepare_local_batch_files"]}},
        },
    }
    context = {"manager": manager, "effective_config": config, "config_version": "c"}
    registry = {"prepare_local_batch_files": {"handler": prepare_local_batch_files}}
    return state, session, context, registry


def test_reopened_refresh_packages_only_missing_frames(tmp_path):
    import hashlib
    from pathlib import Path

    state, session, context, registry = setup(tmp_path)
    state["active_model"]["sha256"] = "model-digest"
    refresh = state["model_refresh"]
    original = refresh["old_diagram"]["entries"][0]
    original["ehull"] = 0
    source = Path(original["structure_path"])
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    alternate = tmp_path / "second.vasp"
    alternate.write_bytes(source.read_bytes() + b"\n")
    manager = context["manager"]
    manager.data["structures"]["s2"] = {**manager.data["structures"]["s"], "source_path": str(alternate)}
    refresh["old_diagram"]["entries"].append(
        {**original, "structure_id": "s2", "structure_path": str(alternate), "ehull": 0.02}
    )
    state["dedup_gate"]["valid_structure_ids"].append("s2")
    refresh["reopened_from_plan"] = "old-plan"
    state["completed_refresh_replans"] = [{
        "prior_refresh": {
            "new_model_version": "new", "task_ids": ["reused"],
            "plan": {"checksum": "old-plan", "model": {"sha256": "model-digest"}},
        },
    }]
    state["tasks"].append({
        "task_id": "reused", "task_key": "old-attempt", "status": "completed",
        "structure_id": "s", "input_structure_sha256": sha, "model_version": "new",
        "converged": True, "parameters": {"model_refresh_operation": "relax"},
        "outputs": {
            "mlip_version": "new", "structure_checksum": "relaxed-sha",
            "optimizer_converged": True, "extrapolation_monitor": {"extrapolation_failure": False},
        },
    })
    state["phase_records"] = [{
        "source_task_id": "reused", "source_version": "new", "structure_sha256": "relaxed-sha",
    }]
    proposal = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert proposal["status"] == "awaiting_approval"
    assert proposal["action"]["target_ids"] == ["s2"]
    outcome = run_model_refresh_gate(
        proposal["state"], session, registry=registry, context=context,
        human_feedback={"decision": "approve", "comment": "同意"},
    )
    assert outcome["status"] == "prepared", outcome["status"]
    saved = outcome["state"]
    assert saved["model_refresh"]["reused_task_ids"] == ["reused"]
    assert len(saved["model_refresh"]["task_ids"]) == 1
    assert saved["tasks"][0]["task_key"] == "old-attempt"
    assert saved["tasks"][1]["structure_id"] == "s2"


def test_refresh_requires_own_approval_and_generates_portable_single_point(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(
        state,
        session,
        registry=registry,
        context=context,
        human_feedback={"decision": "approve", "comment": "同意"},
    )
    assert first["status"] == "awaiting_approval"
    assert not (tmp_path / "submissions").exists()
    second = run_model_refresh_gate(
        first["state"],
        session,
        registry=registry,
        context=context,
        human_feedback={"decision": "approve", "comment": "同意"},
    )
    assert second["status"] == "prepared", second
    task = second["state"]["tasks"][0]
    portable = json.loads(
        __import__("pathlib").Path(task["input_path"]).read_text(encoding="utf-8")
    )
    assert portable["worker_job"]["operation"] == "predict"
    assert portable["worker_job"]["parameters"]["refresh_force_threshold"] == 0.1
    task_dir = __import__("pathlib").Path(task["input_path"]).parent
    assert (task_dir / "_handoff_refresh_executor.py").is_file()
    assert (task_dir / "_handoff_refresh_worker.py").is_file()
    assert (task_dir / "_handoff_refresh_inline.py").is_file()
    import subprocess
    import sys
    checked = subprocess.run([sys.executable, "-c", "import _handoff_refresh_executor; import _handoff_refresh_worker; import _handoff_refresh_inline"], cwd=task_dir, capture_output=True, text=True, timeout=20)
    assert checked.returncode == 0, checked.stderr
    assert "_handoff_refresh_executor:execute_mlip_task" in (task_dir.parent / "run_mlip_batch.py").read_text()
    from zipfile import ZipFile
    batches = second["state"]["slurm_batches"]
    archive = next(b["handoff"]["upload_archive"] for b in batches if b.get("handoff") and task["task_id"] in b.get("task_ids", []))
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        assert any(name.endswith("/_handoff_refresh_executor.py") for name in names)
        assert any(name.endswith("/_handoff_refresh_worker.py") for name in names)
        assert any(name.endswith("/_handoff_refresh_inline.py") for name in names)
    assert portable["worker_job"]["model_path"] == "/models/new.model"
    assert portable["worker_job"]["structure_path"] == "initial.vasp"
    assert "structure_path" not in portable
    assert "Model-refresh-0001" in task["input_path"] and "Single-point" in task["input_path"]
    assert "epoch1" in __import__("pathlib").Path(task["input_path"]).parts
    assert second["state"]["model_refresh"]["wave"] == 0


def test_far_selected_task_has_inline_point_two_threshold(tmp_path):
    state, session, context, registry = setup(tmp_path)
    state["model_refresh"]["old_diagram"]["entries"][0]["ehull"] = 0.02
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    second = run_model_refresh_gate(first["state"], session, registry=registry, context=context, human_feedback="同意")
    assert second["status"] == "prepared", second
    task = second["state"]["tasks"][0]
    assert task["parameters"]["refresh_force_threshold"] == 0.2
    assert task["parameters"]["model_refresh_operation"] == "predict"
    portable = json.loads(__import__("pathlib").Path(task["input_path"]).read_text(encoding="utf-8"))
    assert portable["worker_job"]["parameters"]["refresh_force_threshold"] == 0.2


def recovered_state(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    prepared = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )["state"]
    task = prepared["tasks"][0]
    task.update(
        status="completed",
        converged=False,
        outputs={
            "single_point_completed": True,
            "structure_path": context["manager"].data["structures"]["s"]["source_path"],
            "forces": [[0.11, 0, 0]] * 4,
            "forces_unit": "eV/angstrom",
        },
    )
    prepared["phase_records"] = [
        {"record_id": "r", "source_task_id": task["task_id"], "source_version": "new"}
    ]
    prepared["phase_diagrams"]["mlip"] = {
        "status": "completed",
        "model_version": "new",
        "entries": [{"record_id": "r", "ehull": 0.002, "ehull_unit": "eV/atom"}],
    }
    return prepared, session, context, registry


def test_inline_result_finishes_refresh_without_a_second_batch(tmp_path):
    state, session, context, registry = recovered_state(tmp_path)
    task = state["tasks"][0]
    source_sha = task["input_structure_sha256"]
    task.update(converged=True)
    task["outputs"].update(optimizer_converged=True, single_point_completed=False, refresh_force_followup={"input_structure_sha256": source_sha, "force_max_ev_per_angstrom": .11, "threshold_ev_per_angstrom": .1, "relaxed": True, "initial_prediction": {"single_point_completed": True, "forces_unit": "eV/angstrom", "forces": [[.11, 0, 0]] * 4, "structure_checksum": source_sha, "mlip_version": "new"}})
    result = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert result["status"] == "completed", result
    assert len(result["state"]["tasks"]) == 1
    assert result["state"]["model_refresh"]["status"] == "completed"


def test_legacy_supplement_checkpoint_migrates_to_completed(tmp_path):
    state, session, context, registry = recovered_state(tmp_path)
    state["model_refresh"]["plan"]["policy"].pop("force_followup_version")
    state["model_refresh"].update(
        status="supplement_ready",
        supplemental_candidates=[{"structure_id": "obsolete"}],
    )
    state["pending_execution_policies"] = {
        "legacy": {
            "agent_proposal": {
                "raw_action": {"parameters": {"mode": "model_refresh_inputs"}}
            }
        }
    }
    result = run_model_refresh_gate(state, session, registry=registry, context=context)
    assert result["status"] == "completed", result
    assert result["state"]["model_refresh"]["supplemental_candidates"] == []
    assert "legacy" not in result["state"]["pending_execution_policies"]


def test_old_model_hull_cannot_supply_supplement_ehull(tmp_path):
    state, _, _, _ = recovered_state(tmp_path)
    state["phase_diagrams"]["mlip"]["model_version"] = "old"
    final, waiting = advance_refresh(state)
    assert waiting and "不会沿用旧模型能量" in waiting
    assert final["model_refresh"]["wave"] == 0


def test_public_tool_step_intercepts_search_before_llm(tmp_path):
    state, session, context, registry = setup(tmp_path)
    response = run_tool_step(
        state,
        session,
        registry=registry,
        context=context,
        execution_mode="interactive",
        invocation_id="refresh-test",
    )
    assert response["status"] == "awaiting_approval"
    assert "refresh-test" in response["state"]["pending_execution_policies"]
    assert response["agent_proposal"]["estimated_cost"]["estimated_total_cost"] > 0


def test_new_direct_recovery_requires_parent_first_perturbation(tmp_path):
    from phase_agent.models.capabilities import copy_model_capabilities

    state, session, context, registry = setup(tmp_path)
    context["manager"].data["branches"]["b"].update(P="O3", H=[[1,0,0],[0,1,0],[0,0,1]], x=1, T=["Fe"])
    state["model_refresh"].update(status="waiting_results", task_ids=["failed"])
    state["tasks"] = [{"task_id": "failed", "stage": "relax_and_feature", "status": "failed", "model_version": "new"}]
    action = {"tool": "prepare_local_batch_files", "operation": "sample_direct_extrapolation",
              "parameters": {"mode": "direct_extrapolation", "scope_id": "epoch_refresh", "target": 100},
              "target_ids": [], "task_key": "direct-test", "budget": 0.0,
              "reason": "User requested DIRECT", "expected_purpose": "Prepare review plan"}

    class Client:
        _codex_structured = True
        _validates_original_facts = True
        _explicit_direct_scope = "epoch_refresh"
        _decision_source = "codex"

        def __call__(self, payload):
            return action

    base = Client()
    def wrapper(payload):
        return base(payload)
    copy_model_capabilities(base, wrapper)
    response = run_tool_step(state, session, registry=registry, context=context,
                             agent_client=wrapper, execution_mode="interactive", invocation_id="direct-test")
    assert response["status"] == "not_configured", response
    assert "new_recovery_requires_parent_first_perturbation" in response["reason"]
    assert not (tmp_path / "submissions").exists()


def test_changed_input_requires_fresh_approval(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    path = __import__("pathlib").Path(context["manager"].data["structures"]["s"]["source_path"])
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    second = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert second["status"] == "awaiting_approval"
    assert not (tmp_path / "submissions").exists()


def test_legacy_cost_cap_does_not_block_refresh(tmp_path):
    state, session, context, registry = setup(tmp_path)
    context["effective_config"]["budgets"]["enforcement"] = "hard"
    session["confirmed_snapshot"]["config"]["budgets"]["enforcement"] = "hard"
    context["effective_config"]["budgets"]["total_relative_cost"] = 0.0001
    session["confirmed_snapshot"]["config"]["budgets"]["total_relative_cost"] = 0.0001
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    second = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert second["status"] == "prepared"
    assert len(second["state"]["tasks"]) == 1


def test_refresh_task_limit_is_not_silently_truncated(tmp_path):
    state, session, context, registry = setup(tmp_path)
    context["effective_config"]["budgets"]["stage_limits"]["relax_and_feature"]["max_tasks"] = 0
    session["confirmed_snapshot"]["config"]["budgets"]["stage_limits"]["relax_and_feature"]["max_tasks"] = 0
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    second = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert second["status"] == "budget_exhausted"
    assert not second["state"]["tasks"]
    assert not (tmp_path / "submissions").exists()


def test_failure_waiver_does_not_change_task_status(tmp_path):
    state, _, _, _ = recovered_state(tmp_path)
    state["tasks"][0]["status"] = "failed"
    updated, waiting = advance_refresh(state)
    assert waiting and updated["model_refresh"]["status"] == "waiting_results"
    updated, _ = advance_refresh(state, "不再等待刷新")
    assert updated["tasks"][0]["status"] == "failed"
    assert updated["tasks"][0]["refresh_wait_waived"] is True


def test_refresh_single_point_enters_versioned_hull_without_faking_convergence(tmp_path):
    from phase_agent.persistence.ledger.collect_calculation_results import _phase_record
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    manager = SimpleNamespace(
        data={"structures": {"s": {"branch_id": "b"}}, "branches": {"b": {"P": "O3"}}}
    )
    records = []
    for index, composition in enumerate(({"Fe": 1, "O": 2}, {"Na": 1, "Fe": 1, "O": 2})):
        result = {
            "status": "completed",
            "task_id": str(index),
            "converged": False,
            "model_version": "new",
            "model_refresh_id": "approved",
            "parameters": {"model_refresh_operation": "predict"},
            "outputs": {
                "energy": -10 - index,
                "energy_unit": "eV",
                "mlip_version": "new",
                "composition": composition,
                "single_point_completed": True,
                "actual_phase": "O3",
                "phase_identification": {"status": "identified"},
            },
        }
        record = _phase_record(manager, "s", f"r{index}", "relax_and_feature", result)
        assert record is not None
        records.append(record)
        ordinary = deepcopy(result)
        ordinary.pop("model_refresh_id")
        assert _phase_record(manager, "s", "not-converged", "relax_and_feature", ordinary) is None
    records.append(
        {
            **records[0],
            "record_id": "old-record",
            "source_version": "old",
            "model_version": "old",
            "energy": -1000,
        }
    )
    diagram = update_phase_diagram(records, active_model_version="new")["diagrams"]["mlip"]
    assert diagram["status"] == "completed"
    assert {r["source_version"] for r in diagram["entries"]} == {"new"}
    assert len(diagram["entries"]) == 2


def test_empty_refresh_task_list_cannot_complete(tmp_path):
    state, _, _, _ = recovered_state(tmp_path)
    state["model_refresh"]["task_ids"] = []
    updated, waiting = advance_refresh(state)
    assert "清单为空" in waiting
    assert updated["model_refresh"]["status"] == "waiting_results"
    assert state == updated


def test_duplicate_refresh_evidence_cannot_complete(tmp_path):
    for location in ("tasks", "phase_records", "entries", "task_ids"):
        case_path = tmp_path / location
        case_path.mkdir()
        state, _, _, _ = recovered_state(case_path)
        records = (
            state["phase_diagrams"]["mlip"]["entries"]
            if location == "entries"
            else state["model_refresh"]["task_ids"]
            if location == "task_ids"
            else state[location]
        )
        records.append(deepcopy(records[0]))
        updated, waiting = advance_refresh(state)
        assert waiting and ("重复" in waiting or "多个" in waiting)
        assert updated["model_refresh"]["status"] == "waiting_results"
        assert updated == state


def test_refresh_working_state_isolates_mutable_records_without_copying_history():
    from phase_agent.tools.local.prepare_model_refresh import _refresh_working_state
    class History:
        def __deepcopy__(self, memo):
            raise AssertionError("scientific history must not be deep copied")
    original = {"phase_records": History(), "tasks": [{"task_id": "old"}],
                "pending_tasks": [], "budget_reservations": {"old": {"status": "reserved"}},
                "slurm_batches": [], "upload_layout": {}, "model_refresh": {"status": "approval_required"}}
    copied = _refresh_working_state(original)
    copied["tasks"][0]["task_id"] = "changed"
    copied["budget_reservations"]["old"]["status"] = "changed"
    copied["model_refresh"]["status"] = "changed"
    assert copied["phase_records"] is original["phase_records"]
    assert original["tasks"][0]["task_id"] == "old"
    assert original["budget_reservations"]["old"]["status"] == "reserved"
    assert original["model_refresh"]["status"] == "approval_required"


def test_preserved_unregistered_batch_directory_is_not_overwritten(tmp_path, monkeypatch):
    from phase_agent.tools.remote import build_upload_batch_directory as module
    original_builder = module.build_upload_batch_directory
    occupied = []
    def builder(*args, **kwargs):
        path, metadata = original_builder(*args, **kwargs)
        if kwargs["batch_id"] == "remote-000001":
            path.mkdir(parents=True, exist_ok=True)
            (path / "preserved.txt").write_text("old", encoding="utf8")
            occupied.append(path)
        return path, metadata
    monkeypatch.setattr(module, "build_upload_batch_directory", builder)
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    second = run_model_refresh_gate(first["state"], session, registry=registry, context=context,
                                    human_feedback={"decision": "approve", "comment": "同意"})
    assert second["status"] == "prepared"
    assert occupied and (occupied[0] / "preserved.txt").read_text() == "old"
    assert second["state"]["slurm_batches"][-1]["batch_id"] == "remote-000002"
    assert occupied[0] not in __import__("pathlib").Path(second["state"]["tasks"][0]["input_path"]).parents


def test_handoff_memory_failure_retains_registered_tasks_and_execution(tmp_path, monkeypatch):
    from phase_agent.tools.remote import submission_handoff
    def fail(_state):
        raise MemoryError("synthetic")
    monkeypatch.setattr(submission_handoff, "ensure_waiting_handoffs", fail)
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    second = run_model_refresh_gate(first["state"], session, registry=registry, context=context,
                                    human_feedback={"decision": "approve", "comment": "同意"})
    assert second["status"] == "handoff_preparation_failed"
    assert len(second["state"]["tasks"]) == 1
    assert second["state"]["model_refresh"]["handoff_status"] == "incomplete"
    assert second["state"]["invocations"][second["state"]["model_refresh"]["approved_by"]]["execution"]["status"] == "completed"
    assert not second["state"]["pending_execution_policies"]


def test_batch_id_remains_unique_after_skipping_preserved_directory(tmp_path, monkeypatch):
    from phase_agent.tools.remote import build_upload_batch_directory as module
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner
    original_builder = module.build_upload_batch_directory
    def builder(*args, **kwargs):
        path, metadata = original_builder(*args, **kwargs)
        if kwargs["batch_id"] == "remote-000001":
            path.mkdir(parents=True, exist_ok=True)
        return path, metadata
    monkeypatch.setattr(module, "build_upload_batch_directory", builder)
    tasks = [{"task_id": f"t{i}", "task_key": f"k{i}", "stage": "offline_check_dedup", "status": "pending"} for i in range(2)]
    state = {"tasks": tasks, "budget_reservations": {f"k{i}": {"status": "reserved"} for i in range(2)}}
    runner = RemoteBatchRunner(tmp_path / "root", worker_command=[], stage_batch_sizes={"offline_check_dedup": 1})
    first = runner.prepare(state)
    second = runner.prepare(first["state"])
    assert first["batch"]["batch_id"] == "remote-000002"
    assert second["batch"]["batch_id"] == "remote-000003"
