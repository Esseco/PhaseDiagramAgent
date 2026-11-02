"""Only the new single-action entry may own scientific dispatch."""

import importlib.util
import pytest


def test_retired_multistep_modules_are_absent():
    assert importlib.util.find_spec("phase_agent.graphs.actions.iteration") is None
    assert importlib.util.find_spec("phase_agent.tools.workflows.run_event_loop") is None


@pytest.mark.parametrize("limit", [0, 2, 3, -1, True, "1", 1.0])
def test_action_turn_rejects_non_single_limit_before_loading_or_writing(limit, tmp_path):
    from phase_agent.tools.workflows.run_action_turn import run_action_turn

    missing = tmp_path / "missing" / "state.json"
    with pytest.raises(ValueError, match="exactly one action"):
        run_action_turn(missing, {}, registry={}, max_steps=limit, state_path=missing)
    assert not missing.parent.exists()


def test_action_runtime_keeps_the_supported_entry_only():
    from phase_agent.tools.workflows import run_action_turn as module

    assert callable(module.run_action_turn)
    assert not hasattr(module, "run_event_loop")
    assert not hasattr(module, "prepare_event_loop")
