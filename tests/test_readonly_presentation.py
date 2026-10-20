from pathlib import Path
from unittest.mock import Mock

import pytest

from phase_agent.tools.workflows.readonly_views import read_only_view, is_read_only
from phase_agent.tools.workflows.generation_artifacts import proposal_artifacts
from phase_agent.tools.workflows.region_review import path_for


def test_query_does_not_create_or_rewrite_branch_plan_artifacts(tmp_path, monkeypatch):
    from phase_agent.tools.workflows import region_review

    state = {"workspace_layout": {"workspace_root": str(tmp_path)}, "active_model_version": "m"}
    proposal = {
        "raw_action": {
            "tool": "generate_branches",
            "task_key": "p",
            "parameters": {
                "sampling_mode": "region_adaptive",
                "regions": [],
                "defer_reason": "review",
            },
        }
    }
    writer = Mock(side_effect=AssertionError("read-only query attempted file write"))
    monkeypatch.setattr(region_review, "write", writer)
    read = read_only_view(proposal_artifacts)
    assert read(proposal, state) == {} and not list(tmp_path.rglob("*"))
    path = path_for(state, proposal["raw_action"])
    path.parent.mkdir(parents=True)
    path.write_text('{"original":"approval"}')
    before = path.read_bytes()
    artifacts = read(proposal, state)
    assert artifacts["json"] == str(path) and path.read_bytes() == before
    assert not path.with_suffix(".txt").exists() and not writer.called
    assert not is_read_only()


def test_read_only_mode_is_restored_after_error():
    @read_only_view
    def broken():
        assert is_read_only()
        raise ValueError("query failure")

    with pytest.raises(ValueError):
        broken()
    assert not is_read_only()
