import json
from pathlib import Path
from threading import RLock
from types import SimpleNamespace

import pytest

from phase_agent.runtime.relax_mc_delivery import deliver
from phase_agent.tools.remote.transfer_bundle import checksum, create_bundle


def fixture(tmp_path):
    project = tmp_path / "project"
    work = project / "workflow_state"
    work.mkdir(parents=True)
    batch = project / "batch"
    out = project / "batch_output"
    out.mkdir()
    record = {"directory": str(batch), "expected_results_archive": str(out / "results.zip"),
              "plan_checksum": "p", "model_version": "m", "config_version": "c", "status": "prepared"}
    state = {"active_model_version": "m", "relax_mc_batches": {"b": record}}
    (work / "state.json").write_text(json.dumps(state))
    source_root = tmp_path / "remote"
    source_root.mkdir()
    (source_root / "result_summary.json").write_text(json.dumps({"bundle_id":"b", "plan_checksum":"p", "model_version":"m", "config_version":"c", "status":"completed", "task_count":1}))
    (source_root / "workflow_state.json").write_text(json.dumps({"tasks":[{"task_id":"t", "status":"completed"}]}))
    archive = tmp_path / "return.zip"
    create_bundle(source_root, list(source_root.iterdir()), archive, kind="results")
    control = SimpleNamespace(state_path=work / "state.json", chat_handler=SimpleNamespace(lock=RLock()))
    return control, archive, out / "results.zip"


def test_verified_delivery_and_idempotence(tmp_path):
    c, archive, target = fixture(tmp_path)
    kwargs = {"bundle_id":"b", "archive_path":str(archive), "archive_sha256":checksum(archive)}
    assert deliver(c, **kwargs)["status"] == "delivered"
    assert checksum(target) == checksum(archive)
    assert deliver(c, **kwargs)["status"] == "already_present"
    assert not list(target.parent.glob("*.part"))


def test_invalid_binding_and_conflict_do_not_overwrite(tmp_path):
    c, archive, target = fixture(tmp_path)
    kwargs = {"bundle_id":"wrong", "archive_path":str(archive), "archive_sha256":checksum(archive)}
    with pytest.raises(ValueError, match="unknown_or_cancelled"):
        deliver(c, **kwargs)
    assert not target.exists()
    target.write_bytes(b"existing")
    kwargs["bundle_id"] = "b"
    with pytest.raises(ValueError, match="conflicting_target"):
        deliver(c, **kwargs)
    assert target.read_bytes() == b"existing"


def test_source_digest_is_required(tmp_path):
    c, archive, target = fixture(tmp_path)
    with pytest.raises(ValueError, match="checksum_mismatch"):
        deliver(c, bundle_id="b", archive_path=str(archive), archive_sha256="bad")
    assert not target.exists()
