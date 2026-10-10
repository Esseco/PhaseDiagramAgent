"""Hull improvement requires a consistent scientific energy basis."""
from phase_agent.decisions.scoring.score_hull_improvement import score_hull_improvement


def test_inconsistent_hull_basis_is_unknown():
    result = score_hull_improvement({"energy_basis_id": "a"}, {"energy_basis_id": "b"})
    assert result["status"] == "unknown"
    assert result["score"] is None
