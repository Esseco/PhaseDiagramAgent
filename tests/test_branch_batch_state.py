import pytest

from phase_agent.tools.state.branch_batch_state import create_branch_batch, transition_branch_batch


def test_branch_batch_happy_path_and_invalid_transition():
    batch = create_branch_batch(["B2", "B1"], selected_by="agent", reason="near hull")
    assert batch["branch_ids"] == ["B1", "B2"]
    for status in ("relax_pending", "relax_completed", "hull_ready", "hb_active", "completed"):
        batch = transition_branch_batch(batch, status)
    assert batch["status"] == "completed"
    assert [row["status"] for row in batch["history"]] == [
        "selected", "relax_pending", "relax_completed", "hull_ready", "hb_active", "completed"
    ]
    with pytest.raises(ValueError, match="invalid branch batch transition"):
        transition_branch_batch(batch, "hb_active")
