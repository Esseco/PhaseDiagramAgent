import json
from pathlib import Path
import pytest
from phase_agent.configuration.session.project_config_json import (
    create_project_config_json,
    write_project_config_patch,
)
from phase_agent.configuration.session.load_editable_config_json import load_editable_config_json
from phase_agent.configuration.session.split_project_config import (
    split_project_file,
    read_document,
    editable_project_hash,
    write_document,
    INITIAL_ROOTS,
)
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.configuration.schema.validate_search_config import validate_search_config


def setup(tmp_path):
    source = tmp_path / "search_config.project.json"
    create_project_config_json(source)
    before = load_editable_config_json(source, default_layered_search_config())
    runtime = split_project_file(source)
    return source, runtime, before


def test_split_preserves_full_config_and_exposes_environments(tmp_path):
    initial, runtime, before = setup(tmp_path)
    assert load_editable_config_json(initial, before) == before
    document = read_document(initial)
    assert "python_environments" in document["config"] and "supercomputer" in document["config"]
    assert "budgets" not in document["config"] and "convergence" not in document["config"]
    assert not set(read_document(runtime)["config"]) & INITIAL_ROOTS
    assert initial.with_name(initial.name + ".before-split.bak").is_file()
    raw = initial.read_bytes(), runtime.read_bytes()
    assert split_project_file(initial) == runtime
    assert (initial.read_bytes(), runtime.read_bytes()) == raw


def test_agent_routes_policy_without_changing_initial_file(tmp_path):
    initial, runtime, before = setup(tmp_path)
    raw = initial.read_bytes()
    identity = editable_project_hash(initial)
    changes = write_project_config_patch(
        initial, {"run.batch_size": 17}, expected_hash=identity, baseline_config=before
    )
    assert initial.read_bytes() == raw
    assert changes[0]["file"] == str(runtime)
    assert load_editable_config_json(initial, before)["run"]["batch_size"] == 17
    assert editable_project_hash(initial) != identity
    with pytest.raises(ValueError, match="其他编辑"):
        write_project_config_patch(initial, {"run.batch_size": 18}, expected_hash=identity)


def test_manual_run_edit_invalidates_review_and_initial_edit_is_separate(tmp_path):
    initial, runtime, before = setup(tmp_path)
    identity = editable_project_hash(initial)
    document = read_document(runtime)
    document["config"]["run"]["seed"] = 9
    write_document(runtime, document)
    assert editable_project_hash(initial) != identity
    assert load_editable_config_json(initial, before)["run"]["seed"] == 9
    raw = runtime.read_bytes()
    write_project_config_patch(
        initial, {"python_environments.remote_python": "hpc-dft"}, baseline_config=before
    )
    assert runtime.read_bytes() == raw
    assert (
        load_editable_config_json(initial, before)["python_environments"]["remote_python"]
        == "hpc-dft"
    )


def test_run_cannot_override_boundaries_or_unknown_fields(tmp_path):
    initial, runtime, before = setup(tmp_path)
    doc = read_document(runtime)
    doc["config"]["system"] = {"boundary": {}}
    write_document(runtime, doc)
    with pytest.raises(ValueError, match="初始配置"):
        load_editable_config_json(initial, before)
    del doc["config"]["system"]
    doc["config"]["unknown_policy"] = True
    write_document(runtime, doc)
    with pytest.raises(ValueError, match="未知配置字段"):
        load_editable_config_json(initial, before)


def test_startup_defers_later_thresholds_but_full_validation_requires_them():
    config = default_layered_search_config()
    config["convergence"]["final_energy_mae_tolerance"] = None
    config["mlip_finetune"]["validation"]["max_energy_mae"] = None
    startup = validate_search_config(config, stage="startup")
    assert startup["valid"]
    assert "convergence.final_energy_mae_tolerance" in startup["deferred_stage_fields"]
    assert not validate_search_config(config)["valid"]
    config["convergence"]["hull_change_tolerance"] = -0.1
    assert not validate_search_config(config, stage="startup")["valid"]


def test_chat_hash_covers_both_files(tmp_path):
    from phase_agent.runtime.configuration_chat import ConfigurationChatHandler
    from phase_agent.configuration.session.create_config_draft import create_config_draft

    initial, runtime, before = setup(tmp_path)
    handler = ConfigurationChatHandler(
        {"state_path": str(tmp_path / "state.json"), "config_session": create_config_draft(before)},
        config_session_path=tmp_path / "session.json",
        base_directory=tmp_path,
        editable_config_path=initial,
    )
    identity = handler._editable_config_hash()
    doc = read_document(runtime)
    doc["config"]["run"]["seed"] = 10
    write_document(runtime, doc)
    assert handler._editable_config_hash() != identity


def test_legacy_project_remains_single_file_until_explicit_split(tmp_path):
    source = tmp_path / "legacy.project.json"
    create_project_config_json(source)
    write_project_config_patch(source, {"run.seed": 12})
    assert "run_config_file" not in read_document(source)
    assert load_editable_config_json(source, default_layered_search_config())["run"]["seed"] == 12
    assert not (tmp_path / "run_config.project.json").exists()


def test_remote_validation_settings_are_editable_in_run_file(tmp_path):
    initial, runtime, before = setup(tmp_path)
    raw = initial.read_bytes()
    write_project_config_patch(
        initial,
        {
            "remote_training_validation": {
                "data_path": "independent.xyz",
                "data_version": "v1",
                "criteria": {},
                "ranking_pairs": [[0, 1]],
            }
        },
        baseline_config=before,
    )
    assert initial.read_bytes() == raw
    assert (
        load_editable_config_json(initial, before)["remote_training_validation"]["data_version"]
        == "v1"
    )


def test_running_chat_import_enters_revision_without_llm_or_workflow(tmp_path):
    from phase_agent.runtime.chat_application import RunWorkflowChatHandler
    from phase_agent.tools.step_runner.file_protocol import write_json

    path = tmp_path / "state.json"
    write_json(path, {"active_model_version": "base"})
    calls = []

    def factory(state):
        calls.append(state)
        return lambda messages, conversation_id=None: "已读取两份配置，待检查"

    handler = RunWorkflowChatHandler(
        {"state_path": str(path)},
        workflow=lambda **kwargs: pytest.fail("读取配置不能执行搜索"),
        config_revision_factory=factory,
    )
    reply = handler([{"role": "user", "content": "读取运行配置"}])
    assert calls and "已读取两份配置" in reply


def test_new_project_leaves_unknown_chemistry_and_hpc_model_for_user(tmp_path):
    from phase_agent.runtime.local_project_launcher import create_project

    target = create_project(tmp_path / "project", registry=tmp_path / "registry.json")
    initial = target.parent / "parameters/search_config.project.json"
    doc = read_document(initial)
    assert doc["config"]["mlip"]["model_path"] is None
    assert doc["config"]["system"]["boundary"]["TM_ratio"] == {}
    assert doc["config"]["python_environments"]["local_python"] == "py1"
    assert doc["config"]["python_environments"]["remote_mlip"] is None
    assert (initial.parent / "run_config.project.json").is_file()


def test_split_template_migration_keeps_saved_defaults_and_independent_edits(tmp_path):
    initial, runtime, before = setup(tmp_path)
    doc = read_document(initial)
    doc["profile_digest"] = "old-template"
    write_document(initial, doc)
    run = read_document(runtime)
    run["config"].setdefault("run", {})["seed"] = 47
    run["config"]["run"].pop("initial_states_per_branch", None)
    write_document(runtime, run)
    before["run"]["initial_states_per_branch"] = 7
    write_project_config_patch(
        initial, {"system.H_generation.size_max": 12}, baseline_config=before
    )
    actual = load_editable_config_json(initial, before)
    assert actual["run"]["seed"] == 47
    assert actual["run"]["initial_states_per_branch"] == 7
    assert actual["system"]["H_generation"]["size_max"] == 12
    assert read_document(initial)["run_config_file"] == runtime.name
    assert len(list((initial.parent / "backups/configs").glob("*.bak"))) == 2


def test_split_template_migration_without_baseline_preserves_both_files(tmp_path):
    initial, runtime, _ = setup(tmp_path)
    doc = read_document(initial)
    doc["profile_digest"] = "old-template"
    write_document(initial, doc)
    old = (initial.read_bytes(), runtime.read_bytes())
    with pytest.raises(ValueError, match="基线"):
        write_project_config_patch(initial, {"run.seed": 47})
    assert (initial.read_bytes(), runtime.read_bytes()) == old
