"""Small synthetic regression for shared-path archive publication."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import zipfile

import pytest

from phase_agent.tools.remote import transfer_bundle as transfer
from tests.test_consolidated_transfer import completed


def test_interleaved_writers_cannot_consume_each_others_temporary_file(tmp_path, monkeypatch):
    source = tmp_path / "result.json"
    source.write_text('{"status":"completed"}', encoding="utf-8")
    target = tmp_path / "all.results.zip"
    replace = os.replace
    temporary_paths = []

    def publish(temporary, destination):
        temporary_paths.append(Path(temporary))
        if len(temporary_paths) == 1:
            # Another node finishes after this writer closes its ZIP, before rename.
            transfer.create_bundle(tmp_path, [source], target, kind="results")
        return replace(temporary, destination)

    monkeypatch.setattr(transfer.os, "replace", publish)
    transfer.create_bundle(tmp_path, [source], target, kind="results")
    assert len(set(temporary_paths)) == 2
    assert not any(path.exists() for path in temporary_paths)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("result.json"))["status"] == "completed"


def test_failed_writer_preserves_last_good_archive_and_foreign_temp(tmp_path, monkeypatch):
    source = tmp_path / "result.json"
    source.write_text('{"status":"completed"}', encoding="utf-8")
    target = tmp_path / "all.results.zip"
    transfer.create_bundle(tmp_path, [source], target, kind="results")
    before = target.read_bytes()
    foreign = target.with_name(target.name + ".part")
    foreign.write_bytes(b"another writer owns this file")
    original = transfer.validate_capacity

    def fail_after_streaming(files, manifest_bytes, total_bytes):
        if manifest_bytes:
            raise ValueError("injected publication failure")
        return original(files, manifest_bytes, total_bytes)

    monkeypatch.setattr(transfer, "validate_capacity", fail_after_streaming)
    with pytest.raises(ValueError, match="injected"):
        transfer.create_bundle(tmp_path, [source], target, kind="results")
    assert target.read_bytes() == before
    assert foreign.read_bytes() == b"another writer owns this file"
    assert set(tmp_path.iterdir()) == {source, target, foreign}


def test_posix_range_lock_also_serializes_threads(tmp_path, monkeypatch):
    calls = []
    fake_fcntl = SimpleNamespace(LOCK_EX=2, LOCK_UN=8, lockf=lambda fd, mode: calls.append(mode))
    monkeypatch.setitem(sys.modules, "fcntl", fake_fcntl)
    # Select the POSIX branch without changing Python/pathlib's host platform.
    monkeypatch.setattr(transfer, "os", SimpleNamespace(name="posix"))
    start = threading.Barrier(4)
    active = 0
    peak = 0
    counter = threading.Lock()

    def writer():
        nonlocal active, peak
        start.wait(timeout=10)
        with transfer._archive_lock(tmp_path):
            with counter:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with counter:
                active -= 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(writer) for _ in range(4)]
        for future in futures:
            future.result(timeout=15)
    assert peak == 1
    assert calls == [2, 8] * 4


def test_consolidation_is_safe_across_worker_processes(tmp_path):
    root = tmp_path / "results"
    tasks = [completed(root, f"T{i}") for i in range(4)]
    program = "from pathlib import Path; import sys; from phase_agent.tools.remote.transfer_bundle import result_bundle; root=Path(sys.argv[1]); result_bundle(root,[root/sys.argv[2]],sys.argv[2])"
    processes = []
    try:
        for task in tasks:
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", program, str(root), task.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, (stdout, stderr)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=10)
    with zipfile.ZipFile(root / "transfer/all.results.zip") as archive:
        assert archive.testzip() is None
        assert {f"{task.name}/result.json" for task in tasks} <= set(archive.namelist())
    assert not list(root.rglob("*.part"))


@pytest.mark.parametrize("producer", ["gpu", "training"])
def test_new_portable_handoffs_include_interleaved_writer_fix(tmp_path, monkeypatch, producer):
    import importlib.util
    from tests.test_portable_transfer import emitted_helper

    helper = emitted_helper(tmp_path, producer)
    spec = importlib.util.spec_from_file_location("portable_archive_race", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "diagnostic.json"
    source.write_text('{"status":"completed"}', encoding="utf-8")
    target = tmp_path / "shared.results.zip"
    replace = os.replace
    paths = []

    def publish(temporary, destination):
        paths.append(Path(temporary))
        if len(paths) == 1:
            module.create_bundle(tmp_path, [source], target, kind="results")
        return replace(temporary, destination)

    monkeypatch.setattr(module.os, "replace", publish)
    module.create_bundle(tmp_path, [source], target, kind="results")
    assert len(set(paths)) == 2
    assert not any(path.exists() for path in paths)
    assert "fcntl.lockf" in helper.read_text(encoding="utf-8-sig")
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("diagnostic.json"))["status"] == "completed"
