from copy import deepcopy
from pathlib import Path

from config_layer.defaults.default_layered_search_config import default_layered_search_config
from config_layer.session.apply_config_revision import apply_config_revision
from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
from config_layer.session.create_config_draft import create_config_draft
from execution_layer.remote.summarize_manual_upload_wait import summarize_manual_upload_wait
from execution_layer.dispatch.create_tool_registry import create_tool_registry
from run.main import run_workflow
from run.open_webui_api import format_workflow_reply


def _confirmed_session():
    draft = create_config_draft(default_layered_search_config())
    draft = apply_config_revision(draft, {
        "calculation.mlip_version": "m1",
        "dft.parameters": {"encut": 520},
        "convergence.hull_tolerance": 0.01,
        "convergence.minimum_dft_validations": 1,
        "convergence.coverage_threshold": 0.9,
    })
    return confirm_config_snapshot(draft, user_confirmed=True)


def _prepared_state(task_path, config_version):
    task = {
        "task_id": "RELAX-1", "task_key": "relax-key-1",
        "stage": "relax_and_feature", "status": "pending",
        "batch_id": "remote-000001", "input_path": str(task_path),
        "result_path": str(Path(task_path).parent / "result.json"),
    }
    return {
        "confirmed_config_version": config_version,
        "tasks": [task], "pending_tasks": [deepcopy(task)],
        "budget_usage": {"total_relative_cost": 0.0, "stages": {}},
        "reserved_relative_cost": 0.0,
    }


def test_manual_wait_summary_lists_task_directories():
    task_path = Path("upload/remote-000001/00000-RELAX-1/task.json")
    summary = summarize_manual_upload_wait(_prepared_state(task_path, "c1"))
    assert summary["waiting_task_count"] == 1
    assert summary["task_directories"] == [str(task_path.parent)]
    assert summary["upload_root"] == "upload"


def test_interactive_workflow_waits_instead_of_proposing_duplicate_prep():
    session = _confirmed_session()
    task_path = Path("upload/remote-000001/00000-RELAX-1/task.json")
    state = _prepared_state(task_path, session["confirmed_snapshot"]["config_version"])
    agent_calls = []

    class EmptyCollector:
        def collect_results(self, _state):
            return []

    result = run_workflow(
        None, {}, {}, session,
        state=state, result_collector=EmptyCollector(),
        agent_client=lambda payload: agent_calls.append(payload),
        execution_mode="interactive", max_steps=1,
    )
    assert result["status"] == "awaiting_manual_submission"
    assert result["submitted"] is False
    assert result["steps_executed"] == 0
    assert result["manual_wait"]["waiting_task_count"] == 1
    assert agent_calls == []


def test_newly_prepared_relax_action_returns_handoff_not_processing_status():
    session = _confirmed_session()
    task_path = Path("upload/remote-000001/00000-RELAX-1/task.json")

    def prepare(*, action, context):
        state = deepcopy(context["event_state"])
        task = {
            "task_id": "RELAX-1", "task_key": "relax-key-1",
            "stage": "relax_and_feature", "status": "pending",
            "batch_id": "remote-000001", "input_path": str(task_path),
        }
        state["tasks"] = [task]
        state["pending_tasks"] = [deepcopy(task)]
        return {"status": "prepared", "state": state, "task_count": 1,
                "batch_count": 1, "batches": [{"task_ids": ["RELAX-1"]}]}

    action = {
        "tool": "prepare_local_batch_files", "task_key": "prepare-relax-1",
        "target_ids": ["B1"], "parameters": {"mode": "relax_inputs"},
        "budget": 0, "reason": "prepare existing Relax inputs",
        "expected_purpose": "manual Relax input preparation",
    }
    result = run_workflow(
        None, {}, {}, session, state={"dedup_gate": {"status": "ready"}},
        registry=create_tool_registry({"prepare_local_batch_files": prepare}),
        agent_client=lambda _payload: action,
        execution_mode="interactive", human_feedback={"decision": "approve", "comment": "approve"},
    )
    assert result["status"] == "awaiting_manual_submission"
    assert result["submitted"] is False
    assert result["manual_wait"]["waiting_task_count"] == 1


def test_returned_result_is_reconciled_before_next_agent_action():
    session = _confirmed_session()
    version = session["confirmed_snapshot"]["config_version"]
    task_path = Path("upload/remote-000001/00000-RELAX-1/task.json")
    task = _prepared_state(task_path, version)["tasks"][0]
    state = {
        "confirmed_config_version": version,
        "tasks": [task], "pending_tasks": [deepcopy(task)],
        "budget_reservations": {"relax-key-1": {
            "task_key": "relax-key-1", "stage": "relax_and_feature",
            "reserved_cost": 2.0, "status": "reserved",
        }},
        "reserved_relative_cost": 2.0,
        "budget_usage": {"total_relative_cost": 0.0, "stages": {}},
    }
    returned = {"task_id": "RELAX-1", "task_key": "relax-key-1",
                "stage": "relax_and_feature", "status": "completed", "actual_cost": 1.0}

    class ResultCollector:
        def collect_results(self, _state):
            return [returned]

    result = run_workflow(
        None, {}, {}, session, state=state, result_collector=ResultCollector(),
        agent_client=lambda _payload: {"tool": "pause_search", "parameters": {},
            "budget": 0, "reason": "review recovered Relax results"},
        execution_mode="interactive", human_feedback={"decision": "approve", "comment": "approve"},
    )
    assert result["recovered_count"] == 1
    assert result["state"]["tasks"][0]["status"] == "completed"
    assert result["state"]["budget_usage"]["total_relative_cost"] == 1.0
    assert result["state"]["reserved_relative_cost"] == 0.0


def test_waiting_reply_tells_user_to_submit_and_return_results():
    task_dir = Path("upload/remote-000001/00000-RELAX-1")
    reply = format_workflow_reply({
        "status": "awaiting_manual_submission",
        "manual_wait": {
            "task_count": 1, "waiting_task_count": 1,
            "task_directories": [str(task_dir)], "recovered_count": 0,
            "upload_root": str(task_dir.parent.parent),
        },
    }, Path("state.json"))
    assert "尚未本机提交" in reply
    assert "GPU.sh" in reply
    assert "results" in reply
    assert "未回传任务保持等待" in reply
