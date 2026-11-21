from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from types import SimpleNamespace
import zipfile
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from phase_agent.tools.state.prepared_unsubmitted import (
    FIELD,
    create_record,
    verified_ids,
)
from phase_agent.tools.workflows.manual_training_coexistence import allowed
from phase_agent.policies.workflow_policy import active_tasks, action_errors
from phase_agent.tools.state.workflow_control import commit_control, read_control, restore_control


@pytest.fixture
def sample(tmp_path):
    folder = tmp_path / "DFT_input"
    folder.mkdir()
    archive = folder / "inputs.zip"
    structure = b"fixed structure"
    structure_hash = sha256(structure).hexdigest()
    task = {
        "task_id": "t1",
        "batch_id": "b1",
        "stage": "dft_single_point",
        "status": "pending",
        "task_checksum": "c1",
        "input_structure_sha256": structure_hash,
        "model_version": "m1",
        "config_version": "c1",
        "input_path": str(folder / "task.json"),
    }
    (folder / "task.json").write_text(json.dumps(task), encoding="utf8")
    (folder / "initial_structure.vasp").write_bytes(structure)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("tasks/t1/task.json", json.dumps(task))
        z.writestr("tasks/t1/initial_structure.vasp", structure)
    batch = {
        "batch_id": "b1",
        "status": "prepared",
        "job_id": None,
        "model_version": "m1",
        "config_version": "c1",
        "task_ids": ["t1"],
        "handoff": {"folder": str(folder), "upload_archive": str(archive)},
    }
    state = {
        "tasks": [task],
        "slurm_batches": [batch],
        "active_model_version": "m1",
        "confirmed_config_version": "c1",
        "confirmed_config": {"storage": {"workspace_root": str(tmp_path)}},
    }
    now = datetime.now(timezone.utc)
    observation = {
        "task_ids": ["t1"],
        "observed_at": now.isoformat(),
        "server": "hpc",
        "remote_input_directory": "/remote/DFT_input",
        "scheduler_history_start": "2026-11-01",
        "directory_absent": True,
        "jobs_absent": True,
        "source": "electerm_ssh",
        "user_message": "approved",
        "evidence_note": "SSH checked the exact workdir and scheduler history",
    }
    return state, observation, now, archive


def test_evidence_is_scoped_expiring_and_does_not_mutate_task(sample):
    state, observation, now, archive = sample
    original = deepcopy(state)
    record = create_record(state, observation, now=now)
    assert state == original
    state[FIELD] = [record]
    assert verified_ids(state, now=now) == {"t1"}
    assert len(active_tasks(state)) == 1  # Normal result collection remains unchanged.
    assert not verified_ids(state, now=now + timedelta(minutes=31))
    archive.write_bytes(b"changed")
    assert not verified_ids(state, now=now)


@pytest.mark.parametrize(
    "change",
    [
        "running",
        "unknown",
        "submitted",
        "job_id",
        "task_checksum",
        "model",
        "config",
        "receipt",
        "local_input",
    ],
)
def test_changed_or_ambiguous_evidence_blocks(sample, change):
    state, observation, now, archive = sample
    state[FIELD] = [create_record(state, observation, now=now)]
    if change in {"running", "unknown", "submitted"}:
        state["tasks"][0]["status"] = change
    elif change == "job_id":
        state["slurm_batches"][0]["job_id"] = "42"
    elif change == "task_checksum":
        state["tasks"][0]["task_checksum"] = "changed"
    elif change == "model":
        state["active_model_version"] = "new"
    elif change == "config":
        state["confirmed_config_version"] = "new"
    elif change == "local_input":
        (archive.parent / "initial_structure.vasp").write_bytes(b"changed")
    else:
        (archive.parent / ".submitted_jobs.json").write_text("{}")
    assert not verified_ids(state, now=now)


def test_invalid_observations_fail_closed(sample):
    state, observation, now, _ = sample
    for change in [
        {"jobs_absent": False},
        {"directory_absent": False},
        {"source": "model"},
        {"remote_input_directory": "/other"},
        {"observed_at": (now - timedelta(hours=1)).isoformat()},
        {"user_message": ""},
        {"task_ids": ["t1", "t1"]},
    ]:
        with pytest.raises(ValueError):
            create_record(state, {**observation, **change}, now=now)
    assert create_record(state, {**observation, "source": "ssh"}, now=now)


def test_wait_gate_only_unblocks_frozen_training_input_action(sample):
    state, observation, now, _ = sample
    state[FIELD] = [create_record(state, observation, now=now)]
    action = {
        "tool": "update_mlip",
        "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
        "_decision_team": {"backend": "codex_native"},
        "_simulation_plan": {"steps": ["verified"]},
    }
    wait = {
        "task_ids": ["t1"],
        "waiting_task_count": 1,
        "waiting_by_stage": {"dft_single_point": 1},
    }
    frame = {
        "feedback": {"state": state},
        "manual_wait": wait,
        "agent_client": SimpleNamespace(_frozen_training_action=action),
    }
    assert allowed(frame)
    for mutation in [
        lambda a: a.update(tool="select_dft_candidates"),
        lambda a: a["parameters"].update(prepare_inputs_only=False),
        lambda a: a.pop("_simulation_plan"),
    ]:
        altered = deepcopy(action)
        mutation(altered)
        assert not allowed(
            {**frame, "agent_client": SimpleNamespace(_frozen_training_action=altered)}
        )
    assert not allowed({**frame, "manual_wait": {**wait, "task_ids": ["t1", "other"]}})
    state["pending_execution_policies"] = {"i1": {"agent_proposal": {"raw_action": action}}}
    approved = {
        **frame,
        "agent_client": None,
        "human_feedback": {"decision": "approve"},
        "invocation_id": "i1",
    }
    assert allowed(approved)
    state["slurm_batches"][0]["status"] = "submitted"
    assert not allowed(approved)


def test_legacy_state_without_evidence_still_blocks(sample):
    state, _, _, _ = sample
    assert not verified_ids(state)
    assert len(active_tasks(state)) == 1


def test_training_policy_excludes_only_verified_prepared_inputs(sample):
    state, observation, now, _ = sample
    state["new_dft_records"] = [
        {
            "data_id": "d1",
            "status": "completed",
            "converged": True,
            "checks_passed": True,
            "energy": -1.0,
            "model_version": "m1",
        }
    ]
    action = {
        "tool": "update_mlip",
        "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
        "_decision_team": {"backend": "codex_native"},
        "_simulation_plan": {"steps": ["verified"]},
    }
    config = {"workflow": {"finetune_min_new_dft": 1}}
    assert "fine_tune_waits_for_inflight_tasks" in action_errors(action, state, config)
    state[FIELD] = [create_record(state, observation, now=now)]
    assert "fine_tune_waits_for_inflight_tasks" not in action_errors(action, state, config)
    direct = deepcopy(action)
    direct["parameters"]["prepare_inputs_only"] = False
    assert "fine_tune_waits_for_inflight_tasks" in action_errors(direct, state, config)
    non_team = deepcopy(action)
    non_team.pop("_decision_team")
    assert "fine_tune_waits_for_inflight_tasks" in action_errors(non_team, state, config)
    no_plan = deepcopy(action)
    no_plan.pop("_simulation_plan")
    assert "fine_tune_waits_for_inflight_tasks" in action_errors(no_plan, state, config)
    state["tasks"].append({"task_id": "unknown", "status": "unknown"})
    assert "fine_tune_waits_for_inflight_tasks" in action_errors(action, state, config)


def test_control_revision_restores_evidence_and_preserves_old_tasks(sample, tmp_path):
    state, observation, now, _ = sample
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state), encoding="utf8")
    before = deepcopy(state)
    record = create_record(state, observation, now=now)
    digest = sha256(p.read_bytes()).hexdigest()
    result = commit_control(
        p,
        state,
        {"append": {FIELD: [record]}},
        evidence={"record": record},
        expected_sha256=digest,
    )
    assert result["tasks"] == before["tasks"]
    assert result["slurm_batches"] == before["slurm_batches"]
    restored = restore_control(p, before)
    assert restored[FIELD] == [record]
    assert restored["tasks"] == before["tasks"]
    assert verified_ids(restored, now=now) == {"t1"}
    with pytest.raises(ValueError, match="workflow_state_changed"):
        commit_control(
            p,
            state,
            {"append": {FIELD: [record]}},
            evidence={"record": record},
            expected_sha256=digest,
        )


def test_archive_contents_and_history_scope_are_checked(sample):
    state, observation, now, archive = sample
    with pytest.raises(ValueError, match="history_too_short"):
        create_record(state, {**observation, "scheduler_history_start": "9999-01-01"}, now=now)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(
            "tasks/t1/task.json",
            json.dumps({**state["tasks"][0], "task_checksum": "different"}),
        )
        z.writestr("tasks/t1/initial_structure.vasp", b"fixed structure")
    with pytest.raises(ValueError, match="archive_invalid"):
        create_record(state, observation, now=now)


def test_cross_project_state_fails_closed(sample, tmp_path):
    state, observation, now, _ = sample
    record = create_record(state, observation, now=now)
    state[FIELD] = [record]
    state["confirmed_config"]["storage"]["workspace_root"] = str(tmp_path / "other_project")
    assert not verified_ids(state, now=now)


def test_lifecycle_wait_gate_uses_verified_evidence_only_for_training(sample):
    from phase_agent.tools.workflows.lifecycle_recovery import _workflow_wait_gate

    state, observation, now, _ = sample
    state[FIELD] = [create_record(state, observation, now=now)]
    action = {
        "tool": "update_mlip",
        "parameters": {"prepare_inputs_only": True},
        "_decision_team": {"backend": "codex_native"},
        "_simulation_plan": {"steps": ["verified"]},
    }
    frame = {
        "recovery_question": None,
        "execution_mode": "interactive",
        "feedback": {"state": state},
        "state_path": None,
        "effective_config": {},
        "recovered_count": 0,
        "collection_report": None,
        "snapshot": {"config_version": "c1"},
        "pre_reconciled": {"reconciled": []},
        "rebuilding": False,
        "manual_wait": {
            "task_ids": ["t1"],
            "waiting_task_count": 1,
            "waiting_by_stage": {"dft_single_point": 1},
        },
        "agent_client": SimpleNamespace(_frozen_training_action=action),
    }
    with (
        patch(
            "phase_agent.tools.workflows.reference_dft_startup.startup_gate",
            return_value=None,
        ),
        patch("phase_agent.tools.workflows.lifecycle_recovery._save_runner_state"),
    ):
        assert _workflow_wait_gate(frame) is frame
        state[FIELD] = []
        assert _workflow_wait_gate(frame)["status"] == "awaiting_manual_submission"


def test_competing_control_writes_cannot_overwrite(sample, tmp_path):
    state, observation, now, _ = sample
    p = tmp_path / "state.json"
    p.write_text(json.dumps(state), encoding="utf8")
    record = create_record(state, observation, now=now)
    digest = sha256(p.read_bytes()).hexdigest()

    def commit():
        try:
            commit_control(
                p,
                state,
                {"append": {FIELD: [record]}},
                evidence={"record": record},
                expected_sha256=digest,
            )
            return "saved"
        except ValueError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: commit(), range(2))) == ["saved", "stale"]


def test_formal_registration_does_not_weaken_existing_recovery_guard(sample, tmp_path):
    from threading import RLock
    from phase_agent.runtime.local_agent_control import LocalAgentControl

    state, observation, _, _ = sample
    directory = tmp_path / "workflow_state"
    directory.mkdir()
    p = directory / "state.json"
    p.write_text(json.dumps(state), encoding="utf8")
    digest = sha256(p.read_bytes()).hexdigest()
    guarded = commit_control(
        p,
        state,
        {"append": {}},
        evidence={"kind": "prior"},
        expected_sha256=digest,
        desired_state=state,
        restore_allowed=False,
    )
    assert read_control(p)["restore_allowed"] is False
    original_tasks = deepcopy(guarded["tasks"])
    control = SimpleNamespace(state_path=p, chat_handler=SimpleNamespace(lock=RLock()))
    result = LocalAgentControl.register_prepared_unsubmitted_evidence(control, observation)
    assert result["status"] == "recorded" and result["tasks_modified"] is False
    assert read_control(p)["restore_allowed"] is False
    saved = json.loads(p.read_text(encoding="utf8"))
    assert saved["tasks"] == original_tasks
    assert verified_ids(saved) == {"t1"}
    with pytest.raises(ValueError, match="stale_business_state"):
        restore_control(p, state)
