"""Release-gate cleanup uses a fake subprocess and never runs science."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import recovery_release_gate as gate


@pytest.mark.parametrize("explicit, outcome", [(False, 0), (True, 1), (True, "error")])
def test_gate_cleans_its_own_run_and_keeps_reports(tmp_path, monkeypatch, explicit, outcome):
    parent = tmp_path / "scratch"
    parent.mkdir()
    sentinel = parent / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    report = tmp_path / "results.xml"
    argv = ["gate", "--junitxml", str(report)]
    if explicit:
        argv += ["--basetemp", str(parent)]
    monkeypatch.setattr(gate.sys, "argv", argv)
    monkeypatch.setattr(gate.tempfile, "gettempdir", lambda: str(parent))
    observed = []

    def run(command, *, cwd, env):
        scratch = Path(env["TEMP"])
        assert scratch == Path(env["TMP"])
        assert scratch.is_relative_to(parent)
        assert scratch != parent
        assert scratch.is_dir()
        observed.append(scratch)
        (scratch / "copy.bin").write_bytes(b"fixture")
        assert not Path(
            next(a.split("=", 1)[1] for a in command if a.startswith("--basetemp="))
        ).exists()
        report.write_text("<testsuite/>", encoding="utf-8")
        if outcome == "error":
            raise OSError("intentional subprocess failure")
        return SimpleNamespace(returncode=outcome)

    monkeypatch.setattr(gate.subprocess, "run", run)
    if outcome == "error":
        with pytest.raises(OSError, match="intentional"):
            gate.main()
    else:
        assert gate.main() == outcome
    assert observed and not observed[0].exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert report.is_file()
