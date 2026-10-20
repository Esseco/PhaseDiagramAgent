from copy import deepcopy
from phase_agent.runtime.result_recovery_status import result_recovery_facts
from phase_agent.tools.cost.recovery_job_accounting import add_recovered_job_records
from phase_agent.tools.cost.gpu_usage_ledger import summarize


def test_history_diagnostic_and_current_recovery_are_separate():
    state = {
        "active_model_version": "m",
        "tasks": [
            {"task_id": "t", "status": "pending", "stage": "deep_search", "relax_mc_bundle_id": "b"}
        ],
        "relax_mc_batches": {
            "b": {
                "model_version": "m",
                "import_checksum": "verified",
                "result_summary": {"status": "failed", "reason": "old squeue error"},
            }
        },
    }
    before = deepcopy(state)
    row = result_recovery_facts(state)[0]
    assert row["kind"] == "awaiting_verification" and row["current_error"] is None
    assert row["historical_controller_diagnostic"] == "old squeue error" and state == before
    state["tasks"][0]["status"] = "completed"
    assert result_recovery_facts(state)[0]["kind"] == "completed"
    state["relax_mc_batches"]["b"]["return_error"] = "checksum mismatch"
    assert result_recovery_facts(state)[0]["kind"] == "blocked"
    state["relax_mc_batches"]["b"]["model_version"] = "old"
    assert result_recovery_facts(state) == []


def test_whole_job_gpu_clock_deduplicates_task_slices_and_repeat_recovery():
    records = {
        "task:t:run": {
            "kind": "task_slice",
            "stage": "deep_search",
            "gpu_hours": 0.1,
            "job_id": "",
            "cluster": "",
        }
    }
    state = {
        "result_recoveries": {
            "b": {
                "status": "completed",
                "recovered_task_ids": ["t"],
                "job_evidence": {
                    "job_id": "123",
                    "source": "sacct",
                    "state": "COMPLETED",
                    "elapsed_seconds": 3600,
                    "allocated_gpus": 1,
                },
            }
        }
    }
    add_recovered_job_records(records, state)
    first = summarize(records)
    assert first["total_gpu_hours"] == 1 and not records["task:t:run"]["included_in_total"]
    add_recovered_job_records(records, state)
    assert summarize(records) == first
    assert records["task:t:run"]["gpu_hours"] == 0.1


def test_unknown_gpu_clock_does_not_become_zero_cost():
    records = {}
    state = {
        "result_recoveries": {
            "b": {
                "status": "completed",
                "job_evidence": {
                    "job_id": "1",
                    "source": "sacct",
                    "elapsed_seconds": None,
                    "allocated_gpus": 1,
                },
            }
        }
    }
    add_recovered_job_records(records, state)
    assert records == {}


def test_update_gpu_ledger_accounts_recovery_job_once_without_zero_filling(tmp_path):
    from phase_agent.tools.cost.gpu_usage_ledger import update_gpu_usage

    state = {
        "tasks": [{"task_id": "t", "stage": "deep_search", "status": "completed"}],
        "result_recoveries": {
            "b": {
                "status": "completed",
                "recovered_task_ids": ["t"],
                "job_evidence": {
                    "job_id": "123",
                    "source": "sacct",
                    "state": "COMPLETED",
                    "elapsed_seconds": 3600,
                    "allocated_gpus": 1,
                },
            }
        },
    }
    config = {
        "state_path": str(tmp_path / "state.json"),
        "phase_diagram_directory": str(tmp_path / "analysis"),
    }
    first = update_gpu_usage(state, config)
    second = update_gpu_usage(state, config)
    assert first == second and first["total_gpu_hours"] == 1
    import json

    ledger = json.loads((tmp_path / "ledgers/gpu_usage.json").read_text())
    assert ledger["records"]["task:t:missing"]["gpu_hours"] is None
