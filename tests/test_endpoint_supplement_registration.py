import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from ase import Atoms
from ase.io import write

from phase_agent.evidence.endpoint_supplement import register_endpoint_supplement, supplement_facts
from phase_agent.analysis.phase.phase_dft_candidates import extrapolation_dft_candidates


def sha(data):
    return hashlib.sha256(data).hexdigest()


def case(tmp_path):
    taskdir = tmp_path / "inputs" / "batch" / "00000-t1"
    taskdir.mkdir(parents=True)
    resultdir = tmp_path / "results" / "00000-t1"
    resultdir.mkdir(parents=True)
    atoms = Atoms(
        "NaCoO2", positions=[[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0]], cell=[5, 5, 5], pbc=True
    )
    write(taskdir / "initial.vasp", atoms, format="vasp")
    initial = sha((taskdir / "initial.vasp").read_bytes())
    task = {
        "task_id": "t1",
        "task_key": "key",
        "structure_id": "s1",
        "branch_id": "b1",
        "model_version": "m",
        "status": "failed",
        "input_structure_sha256": initial,
        "input_path": str(taskdir / "task.json"),
        "result_path": str(resultdir / "result.json"),
    }
    (taskdir / "task.json").write_text(json.dumps(task))
    result = {
        "task_id": "t1",
        "task_key": "key",
        "status": "failed",
        "error": "ValueError: refresh_relaxation_not_converged",
    }
    data = json.dumps(result).encode()
    (resultdir / "result.json").write_bytes(data)
    (resultdir / "task.finished.json").write_text(
        json.dumps(
            {"task_id": "t1", "task_key": "key", "status": "failed", "result_checksum": sha(data)}
        )
    )
    atoms.positions[0, 0] += 0.1
    buf = io.StringIO()
    write(buf, atoms, format="vasp")
    frame = buf.getvalue().encode()
    path = "batch/00000-t1/initial_relaxed.vasp"
    archive = tmp_path / "frames.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(path, frame)
        z.writestr(
            "transfer_manifest.json",
            json.dumps(
                {
                    "kind": "results",
                    "files": [{"path": path, "size": len(frame), "sha256": sha(frame)}],
                }
            ),
        )
    csvpath = tmp_path / "evidence.csv"
    fields = [
        "task_id",
        "structure_id",
        "archive_path",
        "initial_sha256",
        "final_sha256",
        "score",
        "classification",
        "error",
    ]
    with csvpath.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(
            dict(
                task_id="t1",
                structure_id="s1",
                archive_path=path,
                initial_sha256=initial,
                final_sha256=sha(frame),
                score="2",
                classification="extrapolation_failure",
                error="",
            )
        )
    state = {
        "active_model_version": "m",
        "analysis_output_directory": str(tmp_path / "analysis"),
        "model_refresh": {"new_model_version": "m", "task_ids": ["t1"]},
        "tasks": [task],
    }
    kwargs = dict(csv_sha256=sha(csvpath.read_bytes()), archive_sha256=sha(archive.read_bytes()))
    return state, csvpath, archive, kwargs


def test_register_exact_frame_and_idempotence(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    state, receipt = register_endpoint_supplement(state, csvpath, archive, **kwargs)
    assert receipt["failure_count"] == 1 and state["tasks"][0]["status"] == "failed"
    assert supplement_facts(state)["failure_count"] == 1
    candidates = extrapolation_dft_candidates(state, [])
    assert (
        len(candidates) == 1
        and candidates[0]["structure_sha256"]
        == state["tasks"][0]["outputs"]["extrapolation_failures"][0]["structure_checksum"]
    )
    assert (
        register_endpoint_supplement(state, csvpath, archive, **kwargs)[1]["status"]
        == "already_registered"
    )
    Path(candidates[0]["structure_path"]).write_text("tampered")
    assert supplement_facts(state) == {"invalid": True}
    assert extrapolation_dft_candidates(state, []) == []
    with pytest.raises(ValueError, match="existing_registration_invalid"):
        register_endpoint_supplement(state, csvpath, archive, **kwargs)


def test_relocate_registered_evidence_only_after_full_validation(tmp_path):
    import shutil

    state, csvpath, archive, kwargs = case(tmp_path)
    state, receipt = register_endpoint_supplement(state, csvpath, archive, **kwargs)
    old = Path(receipt["directory"])
    moved = tmp_path / "analysis" / "latest" / "Historical_Reports" / old.name
    moved.parent.mkdir(parents=True)
    shutil.move(str(old), str(moved))
    assert supplement_facts(state) == {"invalid": True}
    restored, result = register_endpoint_supplement(
        state, moved / "evidence.csv", moved / "frames.zip", **kwargs
    )
    assert result["status"] == "relocated"
    assert supplement_facts(restored)["failure_count"] == 1
    assert state["endpoint_supplement"]["directory"] == str(old)
    assert Path(restored["tasks"][0]["outputs"]["extrapolation_failures"][0]["structure_path"]).is_file()
    (moved / "frames" / "t1.vasp").write_text("corrupt")
    with pytest.raises(ValueError, match="existing_registration_invalid"):
        register_endpoint_supplement(state, moved / "evidence.csv", moved / "frames.zip", **kwargs)


def test_rejects_wrong_checksum_and_input_before_writing(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        register_endpoint_supplement(state, csvpath, archive, **{**kwargs, "csv_sha256": "bad"})
    (Path(state["tasks"][0]["input_path"]).parent / "initial.vasp").write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        register_endpoint_supplement(state, csvpath, archive, **kwargs)
    assert not (tmp_path / "analysis").exists()


def test_rejects_wrong_model_and_scope(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    state["active_model_version"] = "other"
    with pytest.raises(ValueError, match="model"):
        register_endpoint_supplement(state, csvpath, archive, **kwargs)
    state["active_model_version"] = "m"
    state["model_refresh"]["task_ids"] = []
    with pytest.raises(ValueError, match="task_or_score"):
        register_endpoint_supplement(state, csvpath, archive, **kwargs)


def test_orphaned_verified_directory_can_be_reused(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    import copy

    initial = copy.deepcopy(state)
    _, receipt = register_endpoint_supplement(state, csvpath, archive, **kwargs)
    _, again = register_endpoint_supplement(initial, csvpath, archive, **kwargs)
    assert again["status"] == "registered" and again["directory"] == receipt["directory"]


def test_tampered_source_blocks_candidate_even_if_frame_is_intact(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    state, receipt = register_endpoint_supplement(state, csvpath, archive, **kwargs)
    assert len(extrapolation_dft_candidates(state, [])) == 1
    (Path(receipt["directory"]) / "evidence.csv").write_text("tampered")
    assert supplement_facts(state) == {"invalid": True}
    assert extrapolation_dft_candidates(state, []) == []


def test_mcp_dispatches_endpoint_registration_and_read_only_rejects():
    from types import SimpleNamespace
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    class Control:
        def register_endpoint_supplement(self, **kwargs):
            return {"status": "registered", **kwargs}

    server = CodexMcpServer(SimpleNamespace(control=Control()))
    args = {"csv_path": "a", "archive_path": "b", "csv_sha256": "c", "archive_sha256": "d"}
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "phase_register_endpoint_supplement", "arguments": args},
        }
    )
    assert json.loads(response["result"]["content"][0]["text"])["status"] == "registered"
    server.read_only = True
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "phase_register_endpoint_supplement", "arguments": args},
        }
    )
    assert "read_only" in response["error"]["message"]


def test_new_supplement_uses_recorded_epoch_and_preserves_legacy_reads(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    state["upload_layout"] = {"model_rounds": {"m": 2}}
    state, receipt = register_endpoint_supplement(state, csvpath, archive, **kwargs)
    assert Path(receipt["directory"]).is_relative_to(tmp_path / "analysis" / "epoch1" / "endpoint_supplements")
    assert supplement_facts(state)["failure_count"] == 1
    assert not (tmp_path / "analysis" / "endpoint_supplements").exists()


def test_unknown_epoch_supplement_stays_under_latest_unassigned(tmp_path):
    state, csvpath, archive, kwargs = case(tmp_path)
    state, receipt = register_endpoint_supplement(state, csvpath, archive, **kwargs)
    from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory
    assert Path(receipt["directory"]).is_relative_to(analysis_metadata_directory(tmp_path / "analysis") / "Unassigned")
    assert supplement_facts(state)["failure_count"] == 1


def test_legacy_unregistered_supplement_is_reused_without_new_copy(tmp_path):
    from copy import deepcopy
    import shutil

    original, csvpath, archive, kwargs = case(tmp_path)
    old_state = deepcopy(original)
    _, receipt = register_endpoint_supplement(original, csvpath, archive, **kwargs)
    legacy = tmp_path / "analysis" / "endpoint_supplements" / Path(receipt["directory"]).name
    legacy.parent.mkdir(parents=True)
    shutil.move(receipt["directory"], legacy)
    restored, second = register_endpoint_supplement(old_state, csvpath, archive, **kwargs)
    assert Path(second["directory"]) == legacy
    assert supplement_facts(restored)["failure_count"] == 1
