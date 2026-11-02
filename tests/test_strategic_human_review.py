from tests.test_simulation_agent import SimulationModel
from tests.test_event_loop import _session
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry


def test_direction_acceptance_is_not_execution_approval():
    model = SimulationModel()
    model.supports_strategic_review = True
    model.review_required = True
    executed = []
    options = dict(
        session=_session(),
        registry=create_tool_registry(
            {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}}
        ),
        agent_client=model,
        execution_mode="interactive",
        invocation_id="direction",
    )
    first = run_tool_step({}, **options)
    assert first["agent_proposal"]["strategic_review"]["status"] == "awaiting_human_input"
    count = len(model.calls)
    accepted = run_tool_step(first["state"], human_feedback="approve", **options)
    assert accepted["status"] == "awaiting_approval"
    assert accepted["agent_proposal"]["strategic_review"]["status"] == "accepted"
    assert not executed and len(model.calls) == count
    executed_result = run_tool_step(accepted["state"], human_feedback="approve", **options)
    assert executed_result["status"] == "completed"
    assert len(executed) == 1 and len(model.calls) == count
