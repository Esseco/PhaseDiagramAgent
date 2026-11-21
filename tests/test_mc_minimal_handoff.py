import json
from phase_agent.tools.remote.submission_handoff import _handoff_input_files


def test_mc_package_omits_only_unused_refresh_helpers(tmp_path):
    mc = tmp_path / "mc"
    predict = tmp_path / "predict"
    mc.mkdir()
    predict.mkdir()
    for d, operation in ((mc, "mc"), (predict, "predict")):
        (d / "task.json").write_text(
            json.dumps(
                {"stage": "deep_search", "worker_job": {"operation": operation, "parameters": {}}}
            )
        )
        (d / "_handoff_refresh_inline.py").write_text("# refresh only")
        (d / "_handoff_refresh_worker.py").write_text("# worker")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            [
                {"stage": "deep_search", "input_path": "mc/task.json"},
                {"stage": "deep_search", "input_path": "predict/task.json"},
            ]
        )
    )
    files = _handoff_input_files(tmp_path)
    assert mc / "_handoff_refresh_inline.py" not in files
    assert (
        mc / "_handoff_refresh_worker.py" in files
        and predict / "_handoff_refresh_inline.py" in files
    )
    (tmp_path / "manifest.json").write_text("bad json")
    assert mc / "_handoff_refresh_inline.py" in _handoff_input_files(tmp_path)


def test_refresh_request_and_non_mc_are_not_pruned(tmp_path):
    (tmp_path / "task.json").write_text(
        json.dumps(
            {
                "stage": "deep_search",
                "worker_job": {"operation": "mc", "parameters": {"refresh_force_threshold": 0.1}},
            }
        )
    )
    helper = tmp_path / "_handoff_refresh_inline.py"
    helper.write_text("# refresh")
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"stage": "deep_search", "input_path": "task.json"}])
    )
    assert helper in _handoff_input_files(tmp_path)


def _scope_fixture(tmp_path):
    identity = {"invocation_id": "d"}
    task = {
        "task_id": "t",
        "task_key": "k",
        "stage": "deep_search",
        "status": "pending",
        "model_version": "m",
        "config_version": "c",
        "parent_decision_id": "d",
        "branch_id": "b",
        "input_path": "p",
    }
    ref = {k: v for k, v in task.items() if k != "branch_id"}
    action = {
        "tool": "allocate_mc_bohb",
        "target_ids": ["b"],
        "parameters": {"budget_preview": {"allocation_checksum": "hash"}},
    }
    events = [
        {"event": "claimed", "payload": {"identity": identity, "action": action}},
        {
            "event": "tool_returned",
            "payload": {
                "identity": identity,
                "outcome": "completed",
                "evidence": {"result_status": "completed", "returned_task_refs": [ref]},
            },
        },
    ]
    batch = {
        "batch_id": "batch",
        "task_ids": ["t"],
        "status": "prepared",
        "handoff": {"folder": str(tmp_path / "old"), "package_fingerprint": "fp"},
    }
    return {"tasks": [task], "slurm_batches": [batch]}, events


def test_repack_scope_is_exact_and_rejects_changed_or_submitted(tmp_path):
    from copy import deepcopy
    import pytest
    from phase_agent.tools.remote.repack_prepared_mc import validate_prepared_mc_scope

    state, events = _scope_fixture(tmp_path)
    batches, tasks, _ = validate_prepared_mc_scope(
        state, events, decision_id="d", allocation_checksum="hash", package_fingerprint="fp"
    )
    assert len(tasks) == len(batches) == 1
    for key, value in (("job_id", "123"), ("status", "running"), ("task_key", "changed")):
        altered = deepcopy(state)
        altered["tasks"][0][key] = value
        with pytest.raises(ValueError):
            validate_prepared_mc_scope(
                altered,
                events,
                decision_id="d",
                allocation_checksum="hash",
                package_fingerprint="fp",
            )
    altered = deepcopy(state)
    altered["slurm_batches"][0]["job_id"] = "123"
    with pytest.raises(ValueError):
        validate_prepared_mc_scope(
            altered, events, decision_id="d", allocation_checksum="hash", package_fingerprint="fp"
        )


def test_repacked_archive_must_only_drop_refresh_helper(tmp_path):
    import zipfile
    import pytest
    from phase_agent.tools.remote.repack_prepared_mc import _verify_repacked_archive

    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for d in (old, new):
        (d / "submit_all.sh").write_text("submit")
        (d / "dispatch_tasks.py").write_text("dispatch")

    def write(path, items):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for k, v in items.items():
                z.writestr(k, v)

    write(
        old / "inputs.zip",
        {
            "d/task.json": "same",
            "d/_handoff_refresh_inline.py": "unused",
            "transfer_manifest.json": "old",
        },
    )
    write(new / "inputs.zip", {"d/task.json": "same", "transfer_manifest.json": "new"})
    a = {"folder": str(old), "upload_archive": str(old / "inputs.zip")}
    b = {"folder": str(new), "upload_archive": str(new / "inputs.zip")}
    _verify_repacked_archive(a, b, 1)
    write(new / "inputs.zip", {"d/task.json": "changed", "transfer_manifest.json": "new"})
    with pytest.raises(ValueError, match="input_changed"):
        _verify_repacked_archive(a, b, 1)


def test_rebuild_registers_new_handoff_and_preserves_original(tmp_path, monkeypatch):
    import zipfile
    from phase_agent.tools.remote.repack_prepared_mc import rebuild_prepared_mc_handoff

    state, events = _scope_fixture(tmp_path)
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    for folder in (old, new):
        (folder / "submit_all.sh").write_text("submit")
        (folder / "dispatch_tasks.py").write_text("dispatch")
    with zipfile.ZipFile(old / "inputs.zip", "w") as z:
        z.writestr("x/task.json", "same")
        z.writestr("x/_handoff_refresh_inline.py", "helper")
        z.writestr("transfer_manifest.json", "old")
    with zipfile.ZipFile(new / "inputs.zip", "w") as z:
        z.writestr("x/task.json", "same")
        z.writestr("transfer_manifest.json", "new")
    original = {
        "folder": str(old),
        "upload_archive": str(old / "inputs.zip"),
        "package_fingerprint": "fp",
    }
    replacement = {
        "folder": str(new),
        "upload_archive": str(new / "inputs.zip"),
        "package_fingerprint": "new",
    }
    batch = state["slurm_batches"][0]
    batch.update(
        handoff=original,
        upload_directory=str(tmp_path / "epoch1" / "inputs" / "batch"),
        script_path=str(tmp_path / "script.sh"),
    )
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_handoff.handoff_root", lambda path: tmp_path
    )
    monkeypatch.setattr(
        "phase_agent.tools.remote.submission_handoff.prepare_handoff",
        lambda *a, **k: replacement.copy(),
    )
    state, out = rebuild_prepared_mc_handoff(
        state, events, decision_id="d", allocation_checksum="hash", package_fingerprint="fp"
    )
    assert old.joinpath("inputs.zip").is_file() and out["folder"] == str(new)
    assert (
        state["slurm_batches"][0]["handoff"] == out
        and state["handoff_repack_history"][0]["old_handoff"] == original
    )
