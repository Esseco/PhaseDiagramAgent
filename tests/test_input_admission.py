import ast
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from phase_agent.tools.remote.transfer_bundle import create_bundle, input_files


def bundle(tmp_path, *, code="print('ok')\n", structure=None):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "job.sh").write_text("#!/bin/bash\npython3 worker.py\n", newline="\n")
    (source / "worker.py").write_text(code)
    (source / "submission_plan.json").write_text(json.dumps({"scripts": ["job.sh"]}))
    if structure is not None:
        (source / "POSCAR.vasp").write_text(structure)
    folder = tmp_path / "upload"
    folder.mkdir(exist_ok=True)
    create_bundle(source, input_files(source), folder / "inputs.zip", kind="inputs")
    (folder / "submit_all.sh").write_text("#!/bin/bash\npython3 dispatch_tasks.py .\n", newline="\n")
    (folder / "dispatch_tasks.py").write_text("print('dispatch')\n")
    return folder


def test_final_archive_checks_code_without_executing_it(tmp_path):
    from phase_agent.tools.remote.input_validation import validate_handoff

    marker = tmp_path / "must-not-exist"
    folder = bundle(tmp_path, code=f"open({str(marker)!r}, 'w').write('unsafe')\n")
    report = validate_handoff(folder)
    assert report["status"] == "passed" and not marker.exists()
    assert report["remote_environment"] == "not_checked"
    assert report["model_calls"] == 0 and not report["submitted"]
    assert not list(folder.rglob("__pycache__"))


@pytest.mark.parametrize("code,expected", [("def broken(:", "python_syntax"), ("return 1", "python_syntax")])
def test_broken_generated_python_never_becomes_ready(tmp_path, code, expected):
    from phase_agent.tools.remote.input_validation import validate_handoff, InputValidationError

    with pytest.raises(InputValidationError) as caught:
        validate_handoff(bundle(tmp_path, code=code))
    assert caught.value.report["issues"][0]["code"] == expected
    assert "worker.py" in caught.value.report["issues"][0]["file"]


def test_content_cache_is_bounded_and_rechecks_changed_bytes(tmp_path, monkeypatch):
    from phase_agent.tools.remote import input_validation as module

    module.clear_validation_cache()
    folder = bundle(tmp_path)
    first = module.validate_handoff(folder)
    monkeypatch.setattr(ast, "parse", lambda *a, **k: pytest.fail("unchanged code parsed again"))
    second = module.validate_handoff(folder)
    assert second["cache_hit"] and second["fingerprint"] == first["fingerprint"]
    # Even restoring the old mtime cannot make altered final bytes pass a cache.
    outer = folder / "submit_all.sh"
    before = outer.stat()
    outer.write_text("#!/bin/bash\r\nfalse\r\n")
    import os

    os.utime(outer, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(module.InputValidationError):
        module.validate_handoff(folder)
    assert module.validation_cache_info()["packages"] <= module.PACKAGE_CACHE_SIZE


def test_checksum_and_job_references_checked_after_packaging(tmp_path):
    from phase_agent.tools.remote.input_validation import validate_handoff, InputValidationError

    folder = bundle(tmp_path)
    archive = folder / "inputs.zip"
    with zipfile.ZipFile(archive) as zipped:
        contents = {name: zipped.read(name) for name in zipped.namelist()}
    contents["worker.py"] = b"print('tampered')\n"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in contents.items():
            zipped.writestr(name, data)
    with pytest.raises(InputValidationError, match="checksum"):
        validate_handoff(folder)


def test_invalid_structure_is_reported_before_delivery(tmp_path):
    from phase_agent.tools.remote.input_validation import validate_handoff, InputValidationError

    with pytest.raises(InputValidationError) as caught:
        validate_handoff(bundle(tmp_path, structure="not a structure"))
    assert caught.value.report["issues"][0]["code"] == "structure_format"


def test_final_payload_is_streamed_and_large_nontext_not_retained(tmp_path, monkeypatch):
    from phase_agent.tools.remote.input_validation import validate_handoff

    folder = bundle(tmp_path)
    original = Path.read_bytes

    def small_only(path):
        assert path.suffix != ".zip", "archive loaded into memory"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", small_only)
    report = validate_handoff(folder)
    assert report["status"] == "passed"
    assert report["max_text_bytes"] <= 16 * 1024**2


def test_handoff_returns_structured_block_not_prepared(tmp_path):
    from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs

    folder = bundle(tmp_path, code="def broken(:")
    state = {"tasks": [{"task_id": "t", "status": "pending", "submission_state": "awaiting_manual_submit"}],
             "slurm_batches": [{"batch_id": "b", "status": "prepared", "task_ids": ["t"],
                                "handoff": {"folder": str(folder)}}]}
    result = finalize_prepared_handoffs(state)
    assert result["status"] == "not_configured" and not result["submitted"]
    assert result["verification"]["status"] == "failed"
    assert state["tasks"][0]["status"] == "pending"


def test_transfer_fingerprint_helper_is_streaming(tmp_path):
    from phase_agent.tools.remote.transfer_bundle import checksum

    path = tmp_path / "bytes"
    path.write_bytes(b"abc")
    assert checksum(path) == hashlib.sha256(b"abc").hexdigest()


def test_missing_literal_worker_is_blocked(tmp_path):
    from phase_agent.tools.remote.input_validation import validate_handoff, InputValidationError
    folder = bundle(tmp_path)
    (tmp_path / "source/worker.py").unlink()
    create_bundle(tmp_path / "source", input_files(tmp_path / "source"), folder / "inputs.zip", kind="inputs")
    with pytest.raises(InputValidationError) as caught:
        validate_handoff(folder)
    assert caught.value.report["issues"][0]["code"] == "missing_script_reference"


def test_windows_sharing_violation_retries_only_the_read(tmp_path, monkeypatch):
    from phase_agent.tools.remote import input_validation as module
    module.clear_validation_cache()
    folder = bundle(tmp_path)
    original = module.checksum
    calls = []
    def busy_once(path):
        calls.append(1)
        if len(calls) == 1:
            error = PermissionError("shared file")
            error.winerror = 32
            raise error
        return original(path)
    monkeypatch.setattr(module, "checksum", busy_once)
    assert module.validate_handoff(folder)["status"] == "passed"
    assert len(calls) > 1


def test_success_fact_is_small_and_does_not_carry_payloads(tmp_path):
    from phase_agent.tools.remote.input_validation import validate_handoff, verification_summary
    fact = verification_summary(validate_handoff(bundle(tmp_path)))
    assert set(fact) == {"version", "status", "fingerprint", "remote_environment"}
    assert len(json.dumps(fact)) < 220
