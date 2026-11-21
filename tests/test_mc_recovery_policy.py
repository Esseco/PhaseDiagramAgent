from copy import deepcopy

from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer
from phase_agent.tools.state.reconcile_task_results import mark_recovered_batches


def test_main_mc_uses_relax_threshold_three(tmp_path, monkeypatch):
    path = tmp_path / "full.vasp"
    path.write_text("test")

    class Manager:
        boundary = {}
        data = {
            "structures": {
                "s": {
                    "branch_id": "b",
                    "source_path": str(path),
                    "full_na_structure_path": str(path),
                }
            },
            "branches": {"b": {}},
        }

    monkeypatch.setattr(
        "phase_agent.science.mc.project_occupation_history.history_seed",
        lambda *args, **kwargs: "seed",
    )
    prep = create_mlip_task_preparer(Manager(), {}, {"mlip": {"model_path": "/m", "version": "v"}})
    task = {"stage": "deep_search", "structure_id": "s", "parameters": {}, "budget": 20}
    assert prep(task)["worker_job"]["parameters"]["feature_difference_threshold"] == 3


def state():
    return {
        "tasks": [
            {"task_id": "a", "batch_id": "b", "status": "completed"},
            {"task_id": "c", "batch_id": "b", "status": "failed"},
        ],
        "processed_task_ids": ["a", "c"],
        "slurm_batches": [{"batch_id": "b", "task_ids": ["a", "c"], "status": "prepared"}],
    }


def test_full_verified_batch_cannot_be_resubmitted():
    s = state()
    assert mark_recovered_batches(s) == 1
    assert s["slurm_batches"][0]["status"] == "results_synced"
    assert mark_recovered_batches(s) == 0


def test_partial_unverified_cancelled_and_wrong_binding_stay_prepared():
    for modify in (
        lambda s: s["processed_task_ids"].pop(),
        lambda s: s["tasks"][1].update(status="pending"),
        lambda s: s["tasks"][1].update(batch_id="other"),
        lambda s: s["tasks"][1].update(recovery_cancelled=True),
        lambda s: s["slurm_batches"][0]["task_ids"].append("a"),
    ):
        s = state()
        modify(s)
        before = deepcopy(s)
        assert mark_recovered_batches(s) == 0
        assert s == before


def test_recovery_fastpath_updates_saved_batch_without_mutating_input(monkeypatch):
    from phase_agent.tools.workflows import recover_saved_results as recovery

    original = state()
    original.update(confirmed_config={"valid": True}, confirmed_config_version="v")
    saved = []
    monkeypatch.setattr(recovery, "recovery_candidate", lambda *a, **kw: {"pending": False})
    monkeypatch.setattr(recovery, "_save_runner_state", lambda value, path: saved.append(value))
    out = recovery.recover_saved_results(
        original, task_runner=object(), manager=object(), state_path="p"
    )
    assert out["slurm_batches"][0]["status"] == "results_synced"
    assert original["slurm_batches"][0]["status"] == "prepared"
    assert saved == [out]


def test_cancelled_task_not_counted_as_missing(tmp_path):
    from phase_agent.tools.remote.batch_runner import RemoteBatchRunner

    task = {
        "task_id": "a",
        "stage": "relax_and_feature",
        "status": "cancelled",
        "recovery_cancelled": True,
    }
    report = RemoteBatchRunner(tmp_path, worker_command=[]).collect_results_with_report(
        {"tasks": [task]}
    )["report"]
    assert report["considered_count"] == 0
    assert report["missing_result_count"] == 0


def test_stale_submission_index_refreshed_only_when_needed(tmp_path, monkeypatch):
    import json
    from phase_agent.tools.workflows.recover_saved_results import (
        _refresh_recovered_submission_index,
    )

    root = tmp_path / "submissions"
    root.mkdir()
    directory = root / "epoch1"
    folder = str(directory / "delivery" / "transfers" / "MC_input")
    index = root / "submission_index.json"
    index.write_text(
        json.dumps({"handoffs": [{"submission_path": {"path": folder}, "status": ["prepared"]}]})
    )
    s = {
        "slurm_batches": [
            {
                "status": "results_synced",
                "handoff": {"folder": folder, "import_directory": str(directory)},
            }
        ]
    }
    calls = []
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_layout.write_submission_index",
        lambda path, state: calls.append((path, state)),
    )
    _refresh_recovered_submission_index(s)
    assert calls == [(root, s)]
    index.write_text(
        json.dumps(
            {"handoffs": [{"submission_path": {"path": folder}, "status": ["results_synced"]}]}
        )
    )
    _refresh_recovered_submission_index(s)
    assert len(calls) == 1
