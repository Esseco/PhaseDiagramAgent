from decision_layer.scoring.score_hull_improvement import score_hull_improvement

from experiments.compare_rule_and_agent import compare_rule_and_agent
from execution_layer.workflows.run_agent_scheduler import load_scheduler_state, run_agent_scheduler


def _state():
    return {"iteration": 0, "remaining_budget": 2.0, "known_target_ids": ["S1"], "candidates": [{"candidate_id": "S1"}], "tasks": [], "allowed_actions": ["generate", "run_stage", "select_dft", "wait"]}


def test_invalid_agent_falls_back():
    result = run_agent_scheduler(_state(), agent_client=lambda _: {"action": "run_stage", "target_ids": ["BAD"], "budget": 9})
    assert result["record"]["fallback_used"] is True
    assert result["record"]["validation"]["valid"] is True


def test_save_and_restore(tmp_path):
    path = tmp_path / "state.json"
    run_agent_scheduler(_state(), save_path=path)
    restored = load_scheduler_state(path)
    assert restored["iteration"] == 1
    run_agent_scheduler(restored, save_path=path)
    resumed = load_scheduler_state(path)
    assert resumed["iteration"] == 2
    assert len(resumed["decision_history"]) == 2


def test_inconsistent_hull_basis_is_unknown():
    result = score_hull_improvement({"energy_basis_id": "a"}, {"energy_basis_id": "b"})
    assert result["status"] == "unknown" and result["score"] is None


def test_comparison_uses_same_budget():
    result = compare_rule_and_agent(_state(), agent_client=lambda payload: {"action": "run_stage", "target_ids": ["S1"], "budget": 1.0, "reason": "test"})
    assert result["same_initial_budget"] == 2.0
