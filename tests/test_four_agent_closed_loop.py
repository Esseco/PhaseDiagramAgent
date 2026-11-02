"""Four-role reasoning and execution share Scientific State without real compute."""

from copy import deepcopy

from tests.test_decision_team import FakeTeam
from tests.test_simulation_agent import SimulationModel
from tests.test_event_loop import _session
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.workflows.run_tool_step import run_tool_step


def test_independent_specialists_strategy_human_simulation_and_next_iteration():
    class FourRoles(FakeTeam):
        supports_simulation = True
        supports_strategic_review = True

        def __init__(self):
            super().__init__()
            self.expert_client = self
            self.proposals = 0

        def __call__(self, request):
            if request["mode"] in {"simulation_plan", "simulation_diagnosis"}:
                self.calls.append(deepcopy(request))
                return SimulationModel()(request)
            response = super().__call__(request)
            if request["mode"] == "autonomous_search":
                self.proposals += 1
                if self.proposals > 1:
                    return {
                        "tool": "pause_search",
                        "reason": "Review the updated evidence before further work",
                    }
                response.update(
                    tool="check_convergence",
                    operation="check_convergence",
                    parameters={},
                    budget=0,
                    human_review_request={
                        "required": True,
                        "reason": "Resolve a major research-direction uncertainty",
                    },
                    research_plan={
                        "active_step_id": "inspect",
                        "steps": [
                            {
                                "step_id": "inspect",
                                "operation": "check_convergence",
                                "priority": 1,
                                "reason": "Inspect existing results",
                            },
                            {
                                "step_id": "explore",
                                "operation": "generate_branches",
                                "priority": 2,
                                "reason": "Consider exploration after the evidence review",
                                "depends_on": ["inspect"],
                            },
                        ],
                    },
                )
            return response

    model = FourRoles()
    executed = []
    registry = create_tool_registry(
        {
            "check_convergence": lambda **kw: executed.append(kw) or {"converged": False},
            "pause_search": lambda **_: {"status": "paused"},
        }
    )
    options = dict(
        session=_session(),
        registry=registry,
        agent_client=model,
        execution_mode="interactive",
        invocation_id="first",
    )
    first = run_tool_step({}, **options)
    assert first["agent_proposal"]["strategic_review"]["status"] == "awaiting_human_input"
    accepted = run_tool_step(first["state"], human_feedback="approve", **options)
    assert accepted["status"] == "awaiting_approval" and not executed
    finished = run_tool_step(accepted["state"], human_feedback="approve", **options)
    assert len(executed) == 1
    assert finished["simulation_report"]["scientific_completion_claimed"] is False
    assert (
        finished["state"]["action_records"][-1]["final_action"]["research_plan"]["active_step_id"]
        == "inspect"
    )
    next_turn = run_tool_step(finished["state"], **{**options, "invocation_id": "second"})
    assert next_turn["status"] == "awaiting_approval"
    experts = [request for request in model.calls if request["mode"] == "expert_assessment"]
    assert [request["agent_role"] for request in experts] == [
        "exploration",
        "reliability",
        "exploration",
        "reliability",
    ]
    assert (
        experts[0]["evidence_snapshot"]["snapshot_id"]
        != experts[2]["evidence_snapshot"]["snapshot_id"]
    )
    assert all(
        not (request["evidence_snapshot"]["facts"].get("decision_context") or {}).get(
            "expert_advice"
        )
        for request in experts
    )
    rejected = run_tool_step(
        next_turn["state"], human_feedback="reject", **{**options, "invocation_id": "second"}
    )
    assert rejected["status"] == "rejected_by_user" and len(executed) == 1
