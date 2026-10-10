import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest
from phase_agent.runtime.project_service import available_port, project_lock, prepare_studio_config, service_directory


def test_two_folders_have_independent_locks_and_studio_storage(tmp_path):
    a, b = tmp_path / "a/runtime.json", tmp_path / "b/runtime.json"
    with project_lock(a), project_lock(b):
        with pytest.raises(RuntimeError, match="已经运行"):
            with project_lock(a):
                pass
        for path in (a, b):
            graph = prepare_studio_config(path)
            assert graph.parent == service_directory(path)
            assert json.loads(graph.read_text())["http"]["app"] == "phase_agent.runtime.studio_runtime:app"
    with project_lock(a):
        pass
    assert service_directory(a) != service_directory(b)


def test_lock_excludes_another_process(tmp_path):
    path = tmp_path / "runtime.json"
    script = "from phase_agent.runtime.project_service import project_lock; import sys\nwith project_lock(sys.argv[1]): pass"
    with project_lock(path):
        result = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True)
        assert result.returncode != 0
        assert b"RuntimeError" in result.stderr


def test_allocates_other_port_when_first_is_occupied():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        occupied = listener.getsockname()[1]
        assert available_port(occupied) != occupied


def test_single_cr_scope_removes_stale_species():
    from phase_agent.configuration.session.synchronize_system_scope import synchronize_system_scope
    source = {"system": {"boundary": {"P": ["O3"], "TM_ratio": {"Cr": 1}},
                         "species": {"substitutional": ["Fe", "Mn"]}}}
    result = synchronize_system_scope(source)
    assert result["system"]["species"]["substitutional"] == ["Cr"]
    assert result["system"]["constraints"] == {"TM_ratio": {"Cr": 1}, "phases": ["O3"]}
    assert source["system"]["species"]["substitutional"] == ["Fe", "Mn"]


def test_folder_picker_reopens_same_project_without_recreating(tmp_path):
    from phase_agent.runtime.project_dashboard import select_folder
    registry = tmp_path / "projects.json"
    folder = tmp_path / "NaCrO2_O3"
    first = select_folder(folder, registry)
    original = first.read_bytes()
    assert select_folder(folder, registry) == first
    assert first.read_bytes() == original
    assert first.parent.name == "NaCrO2_O3"


def test_rejects_cross_folder_writable_path(tmp_path):
    from phase_agent.runtime.local_project_launcher import validate_project_paths
    target = tmp_path / "a/agent_runtime.json"
    target.parent.mkdir()
    target.write_text(json.dumps({"state_path": "../b/state.json", "ledger_path": "ledger.json"}))
    with pytest.raises(ValueError, match="超出项目文件夹"):
        validate_project_paths(target)


def test_start_button_does_not_compute_before_configuration(tmp_path):
    from phase_agent.runtime.project_dashboard import select_folder, send_start
    path = select_folder(tmp_path / "NaCrO2", tmp_path / "projects.json")
    assert "首次配置尚未确认" in send_start(path, {}, "unused")


def test_single_cr_single_phase_disables_inapplicable_generators():
    from phase_agent.configuration.session.synchronize_system_scope import synchronize_system_scope
    config = {"system": {"boundary": {"P": ["O3"], "TM_ratio": {"Cr": 1}}},
              "generation_actions": {"enabled": ["coverage", "tm_ordering", "competing_phase", "periodic_extension"]}}
    assert synchronize_system_scope(config)["generation_actions"]["enabled"] == ["coverage", "periodic_extension"]


def test_explicit_moved_root_preserves_specific_directory_mapping(tmp_path):
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths
    old = tmp_path / "old"
    new = tmp_path / "new"
    result = relocate_workspace_paths({"storage": {"workspace_root": str(old)},
        "file": str(old / "runtime/state.json")}, [
        {"from": str(old / "runtime"), "to": str(new / "workflow_state")},
        {"from": str(old), "to": str(new)}], new)
    assert result["storage"]["workspace_root"] == str(new)
    assert result["file"] == str(new / "workflow_state/state.json")


def test_moved_root_does_not_allow_unrelated_destination(tmp_path):
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths
    with pytest.raises(ValueError, match="跨工作区"):
        relocate_workspace_paths({}, [{"from": str(tmp_path / "old"),
            "to": str(tmp_path / "unrelated")}], tmp_path / "new")
