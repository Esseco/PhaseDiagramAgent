import json
import os
import subprocess
import sys

import pytest

from phase_agent.runtime.local_project_launcher import create_project


@pytest.mark.parametrize("mode", ["debug", "automatic"])
def test_generated_entry_runs_read_only_in_unicode_spaced_folder(tmp_path, mode):
    root = tmp_path / "新项目 with spaces"
    create_project(root, registry=tmp_path / "registry.json", execution_mode=mode)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    local_appdata = tmp_path / "localappdata"
    index = local_appdata / "PhaseSearchAgent/projects.json"
    index.parent.mkdir(parents=True)
    index.write_bytes(b"do not repair optional registry from check")
    result = subprocess.run(
        [sys.executable, str(root / "run_project.py"), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env={
            **os.environ,
            "PYTHONUTF8": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "LOCALAPPDATA": str(local_appdata),
        },
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "configuration_required" and report["read_only"]
    assert not report["submitted"] and not report["automatic_actions"]
    assert index.read_bytes() == b"do not repair optional registry from check"
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
