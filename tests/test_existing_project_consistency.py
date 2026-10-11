from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

from phase_agent.analysis.state.project_authority import configuration_changes
from phase_agent.analysis.state.project_authority import current_project
from phase_agent.configuration.defaults.default_layered_search_config import (
    default_layered_search_config,
)
from phase_agent.runtime.editable_config_baseline import editable_comparison_baseline
from phase_agent.persistence.models.activate_validated_model import (
    activate_validated_model,
)


def test_added_profile_fields_do_not_masquerade_as_scientific_edits(tmp_path):
    source = tmp_path / "search_config.project.json"
    source.write_text(
        json.dumps({"_format": "phase-search-project-v2"}), encoding="utf-8"
    )
    confirmed = default_layered_search_config()
    confirmed["system"]["constraints"]["TM_ratio"] = {"Co": 1}
    confirmed.pop("python_environments")
    before = deepcopy(confirmed)
    expected = editable_comparison_baseline(confirmed, source, tmp_path)
    assert expected["system"]["constraints"]["TM_ratio"] == {"Co": 1}
    editable = deepcopy(expected)
    assert not configuration_changes(expected, editable)
    editable["round_strategy"]["maximum_mc_budget"] += 1
    assert "round_strategy.maximum_mc_budget" in configuration_changes(
        expected, editable
    )
    editable["python_environments"]["local_python"] = "unexpected_environment"
    assert "python_environments.local_python" in configuration_changes(
        expected, editable
    )
    assert confirmed == before


def test_only_recorded_workspace_move_is_ignored(tmp_path):
    root = tmp_path / "new"
    root.mkdir()
    old = tmp_path / "old"
    source = root / "settings.json"
    source.write_text("{}", encoding="utf-8")
    config = {"system": {"phase_reference_directory": str(old / "structures")}}
    (root / "agent_runtime.json").write_text(
        json.dumps(
            {
                "local_path_relocations": [{"from": str(old), "to": str(root)}],
                "runtime_storage_override": {
                    "workspace_root": str(root / "diagrams/default"),
                    "paths": {},
                },
            }
        ),
        encoding="utf-8",
    )
    expected = editable_comparison_baseline(config, source, root)
    assert expected["system"]["phase_reference_directory"] == str(root / "structures")
    assert expected["storage"]["workspace_root"] == str(root)
    changed = deepcopy(expected)
    changed["system"]["phase_reference_directory"] = str(root / "other_structures")
    assert configuration_changes(expected, changed)
    assert config["system"]["phase_reference_directory"] == str(old / "structures")


def test_activation_synchronizes_candidate_and_registry():
    state = {
        "active_model_version": "m1",
        "active_model": {"version": "m1"},
        "model_registry": {"m1": {"status": "active", "model": {"version": "m1"}}},
        "candidate_models": {"m2": {"status": "validated_candidate"}},
        "tasks": [],
        "branches": {},
    }
    result = activate_validated_model(
        state,
        {"version": "m2"},
        {"status": "completed", "passed": True},
        approval_reason="explicit user approval",
    )
    assert result["status"] == "activated"
    updated = result["state"]
    assert updated["model_registry"]["m1"]["status"] == "historical_active"
    assert updated["candidate_models"]["m2"]["status"] == "activated"
    assert state["candidate_models"]["m2"]["status"] == "validated_candidate"


def test_maintenance_mother_baseline_is_bound_to_confirmed_version(tmp_path):
    from phase_agent.runtime.configuration_read_policy import confirmed_file_unchanged
    from phase_agent.configuration.session.replace_config_from_json import (
        mother_structure_digest,
    )

    config = {"system": {"phase_references": {}}}
    handler = SimpleNamespace(
        editable_config_path=tmp_path / "settings.json", base_directory=tmp_path
    )
    session = {
        "confirmed_snapshot": {"config": config, "config_version": "confirmed-1"},
        "confirmed_file_baseline": {
            "config_version": "confirmed-1",
            "mother_digest": mother_structure_digest(config),
        },
    }
    with patch(
        "phase_agent.runtime.configuration_read_policy.load_editable_config_json",
        return_value=config,
    ):
        assert confirmed_file_unchanged(handler, session)
        session["confirmed_snapshot"]["config_version"] = "confirmed-2"
        assert not confirmed_file_unchanged(handler, session)


def test_workflow_state_junction_does_not_rename_project_as_diagram(tmp_path):
    state_path = tmp_path / "FeMn" / "workflow_state" / "state.json"
    resolved = (
        tmp_path
        / "FeMn"
        / "diagrams"
        / "diagram_default"
        / "workflow_state"
        / "state.json"
    )
    with patch(
        "phase_agent.analysis.state.project_authority.Path.resolve",
        return_value=resolved,
    ):
        card = current_project({}, state_path=state_path)
    assert card["project"] == "FeMn"
    assert card["project_path"] == str(tmp_path / "FeMn")
