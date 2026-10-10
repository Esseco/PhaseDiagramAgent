import json
from pathlib import Path
import pytest

from phase_agent.tools.local.rename_workspace_directories import rename_workspace_directories
from phase_agent.tools.local.remove_migrated_duplicates import remove_migrated_duplicates


def workspace(root):
    (root / "runtime/ledgers").mkdir(parents=True)
    (root / "config").mkdir()
    (root / "upload_batches/epoch0_m1/results").mkdir(parents=True)
    (root / "InitFile").mkdir()
    (root / "InitFile/POSCAR").write_bytes(b"reference atoms")
    result = {"energy": -123.456, "forces": [[0.1, 0.2, 0.3]],
              "structure_path": str(root / "InitFile/POSCAR")}
    (root / "upload_batches/epoch0_m1/results/result.json").write_text(json.dumps(result))
    frozen = {"path": str(root / "InitFile/POSCAR")}
    state = {"confirmed_config": frozen, "confirmed_config_version": "v1",
             "tasks": [{"outputs": result}]}
    (root / "runtime/state.json").write_text(json.dumps(state))
    (root / "config/config_session.json").write_text(json.dumps({"status": "confirmed", "config": frozen}))
    (root / "config/search_config.project.json").write_text(json.dumps({"config": {"reference": frozen["path"]}}))
    (root / "agent_runtime.json").write_text(json.dumps({"state_path": "runtime/state.json",
        "config_session_path": "config/config_session.json", "phase_references_path": "InitFile/POSCAR"}))
    return state


def test_names_and_scientific_values_preserved(tmp_path):
    old = workspace(tmp_path)
    assert rename_workspace_directories(tmp_path)["status"] == "planned"
    report = rename_workspace_directories(tmp_path, apply=True, agent_port=None)
    state = json.loads((tmp_path / "workflow_state/state.json").read_text())
    assert state["confirmed_config"] == old["confirmed_config"]
    assert state["tasks"][0]["outputs"]["energy"] == -123.456
    assert (tmp_path / "structures/reference_structures/POSCAR").read_bytes() == b"reference atoms"
    raw = json.loads((tmp_path / "submissions/epoch0_m1/results/result.json").read_text())
    assert Path(raw["structure_path"]).exists()
    assert raw["forces"] == [[0.1, 0.2, 0.3]]
    assert not (tmp_path / "runtime").exists()
    assert report["verified_files"] > 0
    assert rename_workspace_directories(tmp_path)["status"] == "unchanged"


def test_conflict_does_not_move(tmp_path):
    workspace(tmp_path)
    (tmp_path / "parameters").mkdir()
    with pytest.raises(ValueError):
        rename_workspace_directories(tmp_path, apply=True, agent_port=None)
    assert (tmp_path / "runtime/state.json").exists()


def test_delete_identical_old_only(tmp_path):
    archive = tmp_path / "history_backups/old"
    archive.mkdir(parents=True)
    live = tmp_path / "submissions"
    live.mkdir()
    (live / "input.xyz").write_bytes(b"atom data")
    (archive / "input.xyz").write_bytes(b"atom data")
    (archive / "unique.json").write_text('{"old":1}')
    result = remove_migrated_duplicates(tmp_path, apply=True, agent_port=None)
    assert result["count"] == 1
    assert not (archive / "input.xyz").exists()
    assert (archive / "unique.json").exists()
    assert (live / "input.xyz").exists()


def test_relocations_do_not_change_snapshot_or_external_paths(tmp_path):
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths
    original = {"system": {"phase_references": {"O3": str(tmp_path / "InitFile/O3.vasp")}},
                "model_path": "/remote/models/m1.model"}
    changed = relocate_workspace_paths(original, [{"from": str(tmp_path / "InitFile"),
        "to": str(tmp_path / "structures/reference_structures")}], tmp_path)
    assert "InitFile" in original["system"]["phase_references"]["O3"]
    assert changed["model_path"] == original["model_path"]
    assert "reference_structures" in changed["system"]["phase_references"]["O3"]
    with pytest.raises(ValueError):
        relocate_workspace_paths(original, [{"from": str(tmp_path), "to": str(tmp_path.parent)}], tmp_path)
