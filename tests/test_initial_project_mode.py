import json
import pytest
from phase_agent.runtime.local_project_launcher import create_project
from phase_agent.runtime.project_dashboard import select_folder


@pytest.mark.parametrize("mode,steps", [("debug", 1), ("automatic", 10)])
def test_initial_mode_saved_and_existing_selection_preserved(tmp_path, mode, steps):
    registry = tmp_path / "registry.json"
    path = create_project(tmp_path / "project", registry=registry, execution_mode=mode)
    settings = json.loads(path.read_text(encoding="utf-8"))
    assert settings["execution_mode"] == mode
    assert settings["run_steps_per_click"] == steps
    select_folder(path.parent, registry)
    assert json.loads(path.read_text(encoding="utf-8")) == settings


def test_invalid_mode_creates_nothing(tmp_path):
    with pytest.raises(ValueError):
        create_project(tmp_path / "project", execution_mode="bad")
    assert not (tmp_path / "project").exists()


@pytest.mark.parametrize("name", ["workflow_state/state.json", "agent_memory/chat.json", "submissions/job.json", "analysis_outputs/result.csv", "state.json", "phase_data.json"])
def test_new_project_rejects_previous_runtime_facts_without_modifying_them(tmp_path, name):
    root = tmp_path / "project"
    prior = root / name
    prior.parent.mkdir(parents=True)
    prior.write_text("existing scientific record", encoding="utf-8")
    with pytest.raises(FileExistsError, match="添加已有项目"):
        create_project(root, registry=tmp_path / "registry.json")
    assert prior.read_text(encoding="utf-8") == "existing scientific record"
    assert not (root / "agent_runtime.json").exists()
    assert not (root / "parameters").exists()


def test_new_project_accepts_reference_structures_and_stays_unconfirmed(tmp_path):
    root = tmp_path / "project"
    reference = root / "structures/reference_structures/O3.vasp"
    reference.parent.mkdir(parents=True)
    reference.write_text("user supplied reference", encoding="utf-8")
    create_project(root, registry=tmp_path / "registry.json")
    session = json.loads((root / "parameters/config_session.json").read_text(encoding="utf-8"))
    assert session["status"] == "draft"
    assert not session.get("confirmed_snapshot")
    assert session["config"]["system"]["constraints"]["TM_ratio"] == {}
    assert session["config"]["mlip"]["model_path"] is None
    assert not (root / "workflow_state/state.json").exists()
    assert reference.read_text(encoding="utf-8") == "user supplied reference"
