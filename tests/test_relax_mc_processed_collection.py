from copy import deepcopy
from pathlib import Path

import pytest

from phase_agent.tools.relax_mc_batch.dependencies import advance, collect_phase
from phase_agent.tools.relax_mc_batch.files import read, rebase, write
from phase_agent.tools.remote.integrity import file_checksum
from tests.test_relax_mc_resume_and_gate import fixture, Scheduler


@pytest.mark.parametrize("damage", [None, "checksum", "structure", "validation_error"])
def test_processed_results_reverified_without_duplicate_feedback(tmp_path, monkeypatch, damage):
    state = fixture(tmp_path)
    state["processed_task_ids"] = [t["task_id"] for t in state["tasks"]]
    for t in state["tasks"]:
        t["status"] = "completed"
        d = Path(t["result_path"]).parent
        (d / "final.vasp").write_text("synthetic structure")
        write(
            d / "result.json",
            {
                **t,
                "outputs": {
                    "structure_path": "final.vasp",
                    "structure_checksum": file_checksum(d / "final.vasp"),
                },
            },
        )
        write(d / "task.finished.json", {**t, "result_checksum": file_checksum(d / "result.json")})
        t["result_path"] = str(d / "final.vasp")
    first = state["tasks"][0]
    d = Path(first["result_path"]).parent
    if damage == "checksum":
        with (d / "result.json").open("a") as f:
            f.write(" ")
    elif damage == "structure":
        (d / "final.vasp").write_text("changed")
    elif damage == "validation_error":
        state["result_validation_errors"] = {first["task_id"]: "invalid science"}
    snapshot = deepcopy(state["tasks"])
    from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner

    monkeypatch.setattr(ManualUploadBatchRunner, "collect_results", lambda self, state: [])
    import phase_agent.tools.relax_mc_batch.dependencies as deps

    monkeypatch.setattr(deps, "feedback", lambda *a, **k: pytest.fail("duplicate feedback"))
    captured = []
    import phase_agent.tools.relax_mc_batch.failure_diagnostics as diag

    monkeypatch.setattr(
        diag, "record_remote_diagnostics", lambda r, s, res, missing, *a: captured.append(missing)
    )
    collect_phase(
        tmp_path,
        state,
        None,
        {"config": {"supercomputer": {"worker": {"command": []}}}},
        Scheduler(),
    )
    assert captured == [set() if damage is None else {first["task_id"]}]
    assert state["tasks"] == snapshot


def test_recollect_final_reuses_original_packaging_without_any_submission(
    tmp_path_factory, monkeypatch
):
    from tests.test_relax_mc_phase_dependencies import setup, run_batch
    from phase_agent.tools.relax_mc_batch.completion_gate import require_complete_archive

    root, io, calls = setup(tmp_path_factory.mktemp("rc"), monkeypatch)
    run_batch(root, submit=True, io=io)
    for phase in range(4):
        result = run_batch(root, submit=True, io=io, phase=phase)
    assert result["status"] == "completed"
    before = rebase(read(root / "workflow_state.json"), root)
    before["processed_task_ids"] = [t["task_id"] for t in before["tasks"]]
    before["relax_mc_remote"].update(status="paused", phase=3, completed_mc_waves=[0, 1])
    write(root / "workflow_state.json", rebase(before, root, portable=True))
    receipts, computations = len(io.rows), len(calls)
    manifest = rebase(read(root / "batch.json"), root)
    result = advance(root, manifest, io, phase=3, recollect_final=True)
    assert result["status"] == "completed"
    assert len(io.rows) == receipts and len(calls) == computations
    after = rebase(read(root / "workflow_state.json"), root)
    assert after["tasks"] == before["tasks"]
    assert after["relax_mc_remote"]["completed_mc_waves"] == [0, 1, 2]
    require_complete_archive(result["archive"])
    assert advance(root, manifest, io, phase=3, recollect_final=True)["submitted"] is False


@pytest.mark.parametrize("problem", ["phase", "pending", "active"])
def test_recollect_final_rejects_unsafe_scope(tmp_path, problem):
    state = fixture(tmp_path)
    for t in state["tasks"]:
        t["status"] = "completed"
    phase, io = 3, Scheduler()
    if problem == "phase":
        phase = 2
    elif problem == "pending":
        state["tasks"][0]["status"] = "pending"
    else:
        io.active_ids.add("12")
    write(tmp_path / "workflow_state.json", rebase(state, tmp_path, portable=True))
    original = (tmp_path / "workflow_state.json").read_bytes()
    with pytest.raises(ValueError, match="recollect_final"):
        advance(tmp_path, read(tmp_path / "batch.json"), io, phase=phase, recollect_final=True)
    assert not io.calls
    assert (tmp_path / "workflow_state.json").read_bytes() == original
