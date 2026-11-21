import json
import subprocess
import sys

import pytest

from phase_agent.runtime.context_overview import project_overview


def test_overview_rebuilds_after_source_change(tmp_path):
    state = tmp_path / "state.json"
    config = tmp_path / "config.json"
    ledger = tmp_path / "ledger.json"
    state.write_text(
        json.dumps(
            {
                "active_model_version": "m1",
                "tasks": [{"status": "completed"}],
                "generation_history": [
                    {
                        "model_version": "m1",
                        "new_branch_ids": ["b1", "b2"],
                        "registered_ids": ["s1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config.write_text("{}", encoding="utf-8")
    ledger.write_text("{}", encoding="utf-8")
    first = project_overview(state, config, ledger)
    assert first["cache_status"] == "rebuilt"
    assert first["generation_batches"][0]["branch_count"] == 2
    assert "b1" not in json.dumps(first)
    assert project_overview(state, config, ledger)["cache_status"] == "hit"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{\"updated\":true}')",
            str(ledger),
        ],
        check=True,
    )
    assert project_overview(state, config, ledger)["cache_status"] == "rebuilt"


def test_overview_rejects_change_during_build(tmp_path, monkeypatch):
    import phase_agent.runtime.context_overview as mod

    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    original = mod.read_json

    def mutate(path, default=None):
        value = original(path, default)
        state.write_text('{"changed": true}', encoding="utf-8")
        return value

    monkeypatch.setattr(mod, "read_json", mutate)
    with pytest.raises(ValueError, match="state_changed_during_view_build|sources_changed"):
        mod.project_overview(state, None, None)
    assert not (tmp_path / "context_overview.json").exists()


def test_overview_excludes_retired_and_old_model_batches(tmp_path):
    state = tmp_path / "state.json"
    state.write_text(
        json.dumps(
            {
                "active_model_version": "new",
                "generation_history": [
                    {"model_version": "old", "task_key": "old", "new_branch_ids": ["b1"]},
                    {"model_version": "new", "task_key": "retired", "new_branch_ids": ["b2"]},
                    {"model_version": "new", "task_key": "active", "new_branch_ids": ["b3", "b4"]},
                ],
                "branch_region_review": {"retired_batches": [{"task_key": "retired"}]},
            }
        ),
        encoding="utf-8",
    )
    result = project_overview(state, None, None)
    assert [row["task_key"] for row in result["generation_batches"]] == ["active"]
    assert result["generation_batches"][0]["branch_count"] == 2


def test_read_only_overview_does_not_write(tmp_path):
    state = tmp_path / "state.json"
    state.write_text("{}", encoding="utf-8")
    before = {p.name for p in tmp_path.iterdir()}
    result = project_overview(state, None, None, publish=False)
    assert result["cache_status"] == "computed_read_only"
    assert {p.name for p in tmp_path.iterdir()} == before
