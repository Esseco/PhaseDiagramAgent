import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from phase_agent.runtime.local_project_launcher import (
    create_project,
    read_project_registry,
    register_project,
)


def test_bom_registry_remains_usable_without_dropping_projects(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(["old-project"]), encoding="utf-8-sig")
    assert read_project_registry(path) == ["old-project"]


def test_invalid_registry_is_preserved_and_new_project_can_run(tmp_path):
    registry = tmp_path / "registry.json"
    broken = b'{"broken":'
    registry.write_bytes(broken)
    target = create_project(tmp_path / "new project", registry=registry)
    assert read_project_registry(registry) == [str(target)]
    backups = list(tmp_path.glob("registry.invalid-*.json"))
    assert len(backups) == 1 and backups[0].read_bytes() == broken
    assert (target.parent / "run_project.py").is_file()
    assert not (target.parent / "workflow_state/state.json").exists()


def test_malformed_entries_preserve_good_paths_and_do_not_expand_backups(tmp_path):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(["valid", 3, "valid", None, ""]), encoding="utf-8")
    assert read_project_registry(registry) == ["valid"]
    assert read_project_registry(registry) == ["valid"]
    assert len(list(tmp_path.glob("registry.invalid-*.json"))) == 1


def test_broken_other_project_does_not_poison_new_project(tmp_path):
    registry = tmp_path / "registry.json"
    old = create_project(tmp_path / "old", registry=registry)
    old.write_text('{"state_path":', encoding="utf-8")
    target = create_project(tmp_path / "new", registry=registry)
    assert set(read_project_registry(registry)) == {str(old), str(target)}
    assert old.read_text(encoding="utf-8") == '{"state_path":'


@pytest.mark.parametrize("bad_path", [42, [], {}, " "])
def test_bad_path_type_in_other_project_is_isolated(tmp_path, bad_path):
    registry = tmp_path / "registry.json"
    old = create_project(tmp_path / "old", registry=registry)
    old.write_text(
        json.dumps({"state_path": bad_path, "ledger_path": "ledger.json"}), encoding="utf-8"
    )
    original = old.read_bytes()
    new = create_project(tmp_path / "new", registry=registry)
    assert set(read_project_registry(registry)) == {str(old), str(new)}
    assert old.read_bytes() == original


def test_same_folder_bad_config_still_cannot_register_duplicate(tmp_path):
    registry = tmp_path / "registry.json"
    old = create_project(tmp_path / "old", registry=registry)
    other = old.with_name("other.json")
    other.write_text(
        json.dumps({"state_path": "different.json", "ledger_path": "different-ledger.json"}),
        encoding="utf-8",
    )
    old.write_text("invalid", encoding="utf-8")
    with pytest.raises(ValueError, match="每个文件夹"):
        register_project(other, registry=registry)


def test_concurrent_registrations_keep_every_project(tmp_path):
    registry = tmp_path / "registry.json"

    def add(i):
        return str(create_project(tmp_path / f"project-{i}", registry=registry))

    with ThreadPoolExecutor(max_workers=4) as pool:
        targets = list(pool.map(add, range(8)))
    assert set(read_project_registry(registry)) == set(targets)
    assert not list(tmp_path.glob("*.tmp"))


def test_backup_failure_never_replaces_the_original_registry(tmp_path, monkeypatch):
    registry = tmp_path / "registry.json"
    original = b"broken index bytes"
    registry.write_bytes(original)
    real_open = Path.open

    def blocked(path, *args, **kwargs):
        if ".invalid-" in path.name:
            raise PermissionError("backup directory is read-only")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", blocked)
    with pytest.raises(PermissionError):
        read_project_registry(registry)
    assert registry.read_bytes() == original


def test_failed_registry_write_can_reopen_completed_scaffold_without_regeneration(
    tmp_path, monkeypatch
):
    from phase_agent.runtime import project_registry, project_dashboard

    registry = tmp_path / "registry.json"
    real_write = project_registry._write_json

    def fail(*args, **kwargs):
        raise PermissionError("temporarily locked")

    monkeypatch.setattr(project_registry, "_write_json", fail)
    root = tmp_path / "new"
    with pytest.raises(PermissionError):
        create_project(root, registry=registry)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(project_registry, "_write_json", real_write)
    target = project_dashboard.select_folder(root, registry)
    assert target == root / "agent_runtime.json"
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert read_project_registry(registry) == [str(target)]
