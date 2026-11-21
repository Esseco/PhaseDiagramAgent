import pytest


@pytest.mark.parametrize(
    "result",
    [
        {"status": "paused_for_review", "registered": [{"branch_id": "b"}], "summary": {}},
        {
            "status": "completed",
            "registered": [{"branch_id": "b"}],
            "summary": {"sampling_report": {"status": "paused_for_review"}},
        },
        {
            "status": "completed",
            "registered": [{"branch_id": "b"}],
            "summary": {"initialization_failed_branches": 1},
        },
    ],
)
def test_paused_generation_never_prepares_bundled_inputs(result):
    from phase_agent.tools.workflows.scientific_module_scope import prepare_generated_module

    action = {"parameters": {"module_execution_scope": {"branch_upper": 1}}}
    out = prepare_generated_module(action, result, {})
    assert out["status"] == "paused_for_review"
    assert "requires_review" in out.get("reason", "")
