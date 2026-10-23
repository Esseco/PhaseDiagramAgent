import json
from pathlib import Path

import pytest

from phase_agent.runtime.local_project_launcher import create_project


@pytest.mark.parametrize("mode", ["debug", "automatic"])
def test_new_project_has_one_entry_and_one_action_per_turn(tmp_path, mode):
    config = create_project(tmp_path / mode, registry=tmp_path / "registry.json", execution_mode=mode)
    settings = json.loads(config.read_text(encoding="utf-8"))
    assert settings["run_steps_per_click"] == 1
    assert (config.parent / "run_project.py").is_file()
    assert (config.parent / "structures/reference_structures").is_dir()
    assert not (config.parent / "phase_agent").exists()
    assert not (config.parent / "workflow_state/state.json").exists()
    for category in ("submissions", "analysis_outputs", "interaction_records"):
        assert (config.parent / category / "README.md").is_file()
    assert not (config.parent / "analysis_outputs/output_index.md").exists()
    assert not (config.parent / "交互记录").exists()
    assert "run_project.py --check" in (config.parent / "README.md").read_text(encoding="utf-8")
    assert "](analysis_outputs/output_index.md)" not in (config.parent / "README.md").read_text(encoding="utf-8")


def test_entry_never_starts_services_when_checking(tmp_path, monkeypatch):
    from phase_agent.runtime import local_project_launcher as launcher
    config = create_project(tmp_path / "project", registry=tmp_path / "registry.json")
    monkeypatch.setattr(launcher, "start_agent", lambda *a, **k: pytest.fail("service started"))
    monkeypatch.setattr(launcher, "register_project", lambda *a, **k: pytest.fail("read-only modified registry"))
    assert launcher.main(["--project", str(config), "--check"]) == 0


def test_existing_user_entry_is_not_overwritten(tmp_path):
    from phase_agent.configuration.runtime.project_template import write_project_entry
    root = tmp_path / "project"
    root.mkdir()
    path = root / "run_project.py"
    path.write_text("my own entry")
    with pytest.raises(FileExistsError):
        write_project_entry(root)
    assert path.read_text() == "my own entry"


def test_partial_publication_cleans_only_owned_files(tmp_path, monkeypatch):
    from phase_agent.configuration.session import project_scaffold as scaffold
    root = tmp_path / "project"
    mother = root / "structures/reference_structures/O3.vasp"
    mother.parent.mkdir(parents=True)
    mother.write_text("preexisting scientific input")
    original = scaffold.shutil.copyfileobj
    calls = []
    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("synthetic disk failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(scaffold.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="synthetic disk failure"):
        create_project(root, registry=tmp_path / "registry.json")
    assert mother.read_text() == "preexisting scientific input"
    assert not (root / "agent_runtime.json").exists()
    assert not list(root.glob(".project-*"))
    assert not (root / "parameters").exists()
