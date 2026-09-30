import json
from unittest.mock import patch

from execution_layer.cost.runtime_observation import start_timer, runtime_observation
from execution_layer.remote.run_mlip_batch import run_mlip_batch
from execution_layer.workflows.attach_dft_preview import attach_dft_preview
from execution_layer.state.state_manager import update_state_snapshot
from config_layer.defaults.default_dft_decision_config import default_dft_decision_config


def test_runtime_uses_executed_mc_settings():
    result = runtime_observation(start_timer(), {"parameters": {"max_steps": 100, "patience": 20}},
                                 {"max_mc_steps": 80, "patience_steps": 15, "actual_mc_steps": 30})
    assert result["max_mc_steps"] == 80
    assert result["patience"] == 15
    assert result["actual_mc_steps"] == 30


def test_statistics_write_failure_is_nonfatal(tmp_path):
    manifest = tmp_path / "batch" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("[]")
    with patch("pathlib.Path.write_text", side_effect=OSError("read only")):
        assert run_mlip_batch(manifest) == []


def test_revised_dft_replaces_old_preview_and_cost():
    state = {"qbc_candidates": [{"candidate_id": "s", "atom_count": 40}]}
    action = {"tool": "select_dft_candidates", "budget": 999,
        "parameters": {"dft_input_preview": {"relative_cost": 999},
                       "decisions": [{"candidate_id": "s", "action": "DFT_SINGLE_POINT"}]}}
    updated, error = attach_dft_preview(action, state, {"qbc": default_dft_decision_config()})
    assert error is None and updated["budget"] == 30
    assert updated["parameters"]["dft_input_preview"]["schema_version"] == 2
    assert action["budget"] == 999


def test_new_history_summaries_keep_full_current_and_legacy():
    old = {"snapshot_id": "legacy", "decision_context": {"legacy_evidence": [1]}}
    state = update_state_snapshot({"state_snapshots": [old],
        "branch_candidates": [{"branch_id": "b"}], "decision_memory": {"long_term_advice": ["keep"]}})
    assert state["state_snapshots"][0]["snapshot_id"] == "legacy"
    assert state["state_snapshots"][0]["decision_context"] == old["decision_context"]
    assert "decision_context" not in state["state_snapshots"][-1]
    assert state["state_snapshots"][-1]["history_format"] == "summary-v1"
    assert state["current_state_snapshot"]["available_branches"] == [{"branch_id": "b"}]


def test_chat_compatibility_import():
    from run.open_webui_api import brief_chat_state
    from run.chat_state_presentation import brief_chat_state as implementation
    assert brief_chat_state is implementation


def test_missing_manager_does_not_invent_generation_evidence():
    from execution_layer.state.restore_generation_gate import restore_generation_gate
    source = {"generation_history": [{"summary": {"unique_structures": 1,
              "registered_structures": 1}, "registered_ids": ["unknown"]}]}
    result = restore_generation_gate(source, None)
    assert result == source
    assert "dedup_gate" not in result


def test_template_review_missing_ledger_reports_clear_error():
    import pytest
    from execution_layer.workflows.dft_template_review import create_review
    with pytest.raises(ValueError, match="缺少结构台账"):
        create_review({}, {}, {}, None)
