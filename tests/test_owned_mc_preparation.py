from copy import deepcopy
from types import SimpleNamespace
import pytest

from phase_agent.analysis.phase.identify_result_phase import identify_result_phase
from phase_agent.tools.local.prepare_mc_upload_batches import prepare_mc_upload_batches


class NoCopy(list):
    def __deepcopy__(self, memo):
        raise AssertionError("owned task payload was copied again")


def test_owned_phase_identification_skips_copy_without_weakening_default():
    manager = SimpleNamespace(data={})
    owned = {"stage": "not_a_scientific_task", "payload": NoCopy([1])}
    assert identify_result_phase(owned, manager, _owned=True) is owned
    public = {"stage": "not_a_scientific_task", "payload": [1]}
    result = identify_result_phase(public, manager)
    result["payload"].append(2)
    assert public["payload"] == [1]


@pytest.mark.parametrize("fail", [False, True])
def test_mc_batches_reuse_private_state_preserve_caller(tmp_path, monkeypatch, fail):
    import phase_agent.tools.local.prepare_mc_upload_batches as module
    import phase_agent.science.mlip.slurm_executor as executor
    import phase_agent.tools.remote.submission_handoff as handoff

    src = tmp_path / "relaxed.vasp"
    src.write_text("fake", encoding="utf8")
    task = {
        "task_id": "MC-1",
        "task_key": "k",
        "stage": "deep_search",
        "status": "pending",
        "branch_id": "B",
        "structure_id": "S",
        "structure_path": str(src),
    }
    state = {
        "dedup_gate": {"status": "ready"},
        "tasks": [task],
        "large_history": [{"details": list(range(5))}],
    }
    original = deepcopy(state)
    calls = []

    class Runner:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, current, *, _owned=False, _defer_index=False, _defer_archive=False):
            calls.append((_owned, _defer_index, _defer_archive))
            current["tasks"][0]["private_mutation"] = len(calls)
            if fail:
                raise RuntimeError("synthetic failure")
            if len(calls) == 1:
                return {
                    "status": "prepared",
                    "state": current,
                    "batch": {
                        "batch_id": "1",
                        "upload_directory": str(tmp_path),
                        "results_directory": str(tmp_path),
                        "task_ids": ["MC-1"],
                    },
                }
            return {"status": "no_tasks", "state": current}

    monkeypatch.setattr(module, "ManualUploadBatchRunner", Runner)
    monkeypatch.setattr(module, "create_mlip_task_preparer", lambda *a, **k: None)
    monkeypatch.setattr(executor, "_resolve_structure_id", lambda *a: "S")
    monkeypatch.setattr(handoff, "finalize_prepared_handoffs", lambda state: {})
    context = {
        "event_state": state,
        "effective_config": {
            "upload_batches_directory": str(tmp_path),
            "supercomputer": {"worker": {"command": ["worker"]}},
        },
        "manager": SimpleNamespace(data={"structures": {"S": {"branch_id": "B"}}}),
    }
    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            prepare_mc_upload_batches(action={}, context=context)
    else:
        result = prepare_mc_upload_batches(action={}, context=context)
        assert result["status"] == "prepared" and result["task_count"] == 1
    assert calls == ([(True, True, True)] if fail else [(True, True, True)] * 2)
    assert state == original
