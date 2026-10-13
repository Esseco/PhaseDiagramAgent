from pathlib import Path
import zipfile
import json
from phase_agent.science.mlip.slurm_executor import remote_model_references
from phase_agent.tools.remote.submission_handoff import prepare_handoff
from phase_agent.decisions.agent.dialogue_contract import dialogue_errors


def test_hpc_path_wins_without_local_inspection(monkeypatch):
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda *args: (_ for _ in ()).throw(AssertionError("local weight inspection")),
    )
    path = "/hpc/models/com_1.model"
    main, paths, evidence = remote_model_references(
        {"model_path": "C:/local/model.model", "remote_model_path": path, "sha256": "a" * 64}
    )
    assert main == path and paths == [] and evidence[0]["sha256"] == "a" * 64


def test_existing_local_weight_is_not_copied_or_rewritten(tmp_path):
    weight = tmp_path / "model.model"
    weight.write_bytes(b"weight")
    root = tmp_path / "operation"
    batch = root / "inputs/batch"
    batch.mkdir(parents=True)
    script = batch / "GPU.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    task = {"worker_job": {"model_path": str(weight)}, "stage": "relax_and_feature"}
    original = json.dumps(task).encode()
    (batch / "task.json").write_bytes(original)
    handoff = prepare_handoff(root, [script])
    with zipfile.ZipFile(handoff["upload_archive"]) as archive:
        assert archive.read("inputs/batch/task.json") == original
        assert not any(name.startswith("models/") for name in archive.namelist())


def test_transport_annotations_do_not_break_continue_or_approve():
    transport = {
        "type": "dialogue",
        "_proposal_harness": "legacy",
        "_analysis_route": {"harness": "legacy"},
    }
    assert dialogue_errors({"kind": "advance", **transport}, True) == []
    assert (
        dialogue_errors(
            {"kind": "review", "plan_id": "frozen", "decision": "approve", **transport}, True
        )
        == []
    )
    assert dialogue_errors({"kind": "advance", "parameters": {}, **transport}, True)
    assert dialogue_errors({"kind": "advance", "type": "execute"}, True)
    assert dialogue_errors({"kind": "advance", "type": {}}, True)
