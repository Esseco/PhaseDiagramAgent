from copy import deepcopy
import pytest

from phase_agent.tools.workflows.run_tool_step import run_tool_step
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from tests.test_event_loop import _session


class SimulationModel:
    supports_simulation = True

    def __init__(self):
        self.calls = []
        self.bad = False

    def __call__(self, request):
        self.calls.append(deepcopy(request))
        if request.get("mode") == "simulation_plan":
            return {
                "action_hash": request["action_hash"],
                "rationale": "Inspect recorded convergence evidence.",
                "steps": [
                    {"kind": "preflight", "description": "Check frozen configuration"},
                    {
                        "kind": "invoke",
                        "description": "Call registered tool",
                        "tool": "update_mlip" if self.bad else request["action"]["tool"],
                    },
                    {"kind": "verify", "description": "Check returned evidence"},
                ],
                "checks": ["Read existing result status"],
                "failure_policy": "return_to_strategic",
            }
        if request.get("mode") == "simulation_diagnosis":
            return {
                "assessment": "Tool failed; completion unknown.",
                "evidence": ["execution.error"],
                "recommendation": "return_to_strategic",
                "limitations": "No remote verification performed.",
            }
        result = {
            "tool": "check_convergence",
            "parameters": {},
            "budget": 0,
            "reason": "Review evidence",
        }
        if getattr(self, "review_required", False):
            result["human_review_request"] = {"required": True, "reason": "Review a major scientific conflict"}
        return result


def test_simulation_plan_frozen_before_approval_and_checked_after_execution():
    model = SimulationModel()
    executed = []
    registry = create_tool_registry(
        {"check_convergence": lambda **kw: executed.append(kw) or {"converged": False}}
    )
    opts = dict(
        session=_session(),
        registry=registry,
        invocation_id="simulation",
        execution_mode="interactive",
        agent_client=model,
    )
    first = run_tool_step({}, **opts)
    assert first["agent_proposal"]["simulation_plan"]["steps"][1]["kind"] == "invoke"
    assert not executed
    count = len(model.calls)
    result = run_tool_step(first["state"], human_feedback="approve", **opts)
    assert len(executed) == 1 and len(model.calls) == count
    assert result["simulation_report"]["tool_completed"] is True
    assert result["simulation_report"]["scientific_completion_claimed"] is False


def test_simulation_cannot_expand_approved_tool_scope():
    model = SimulationModel()
    model.bad = True
    result = run_tool_step(
        {},
        _session(),
        registry=create_tool_registry({"check_convergence": lambda **_: {}}),
        agent_client=model,
        execution_mode="interactive",
    )
    assert result["status"] == "rejected"
    assert "simulation" in result["reason"].lower()


def test_failure_diagnosis_does_not_retry_tool():
    model = SimulationModel()
    calls = []

    def fail(**kw):
        calls.append(kw)
        raise RuntimeError("synthetic failure")

    opts = dict(
        session=_session(),
        registry=create_tool_registry({"check_convergence": fail}),
        agent_client=model,
        execution_mode="interactive",
        invocation_id="failure",
    )
    first = run_tool_step({}, **opts)
    result = run_tool_step(first["state"], human_feedback="approve", **opts)
    assert len(calls) == 1
    assert result["simulation_report"]["diagnosis"]["recommendation"] == "return_to_strategic"
    assert model.calls[-1]["mode"] == "simulation_diagnosis"


def test_changed_action_rejects_stale_frozen_simulation_plan():
    model = SimulationModel()
    calls = []
    opts = dict(
        session=_session(),
        registry=create_tool_registry({"check_convergence": lambda **kw: calls.append(kw) or {}}),
        agent_client=model,
        execution_mode="interactive",
        invocation_id="modify",
    )
    first = run_tool_step({}, **opts)
    result = run_tool_step(
        first["state"],
        human_feedback={
            "decision": "modify",
            "modifications": {"reason": "changed scope"},
            "comment": "approve",
        },
        **opts,
    )
    assert result["status"] == "rejected" and not calls


def test_old_coordinator_descriptor_resolves_strategic():
    from phase_agent.compatibility.module_references import import_saved_module

    assert (
        import_saved_module("phase_agent.agents.coordinator.agent").__name__
        == "phase_agent.agents.strategic.agent"
    )


def test_real_chat_reuses_strategy_but_still_runs_simulation_planner(tmp_path):
    from tests.test_decision_team import FakeTeam
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import read_json

    class FourAgents(FakeTeam):
        supports_simulation = True
        supports_strategic_review = True

        def __call__(self, request):
            if request.get("mode") in {"simulation_plan", "simulation_diagnosis"}:
                self.calls.append(deepcopy(request))
                return SimulationModel()(request)
            result = super().__call__(request)
            if request.get("mode") == "autonomous_search":
                result.update(
                    tool="check_convergence",
                    parameters={},
                    budget=0,
                    reason="Inspect current evidence",
                    human_review_request={"required": True, "reason": "Review a major scientific conflict"},
                )
            return result

    model = FourAgents()
    model.expert_client = model
    calls = []
    handler = make_handler(
        tmp_path,
        model,
        {"check_convergence": lambda **kw: calls.append(kw) or {"converged": False}},
    )
    handler([{"role": "user", "content": "分析并建议下一步"}], conversation_id="four")
    pending = next(iter(read_json(handler.state_path)["pending_execution_policies"].values()))
    assert pending["agent_proposal"]["simulation_plan"]
    assert pending["agent_proposal"]["strategic_review"]["status"] == "awaiting_human_input"
    count = len(model.calls)
    assert sum(r["mode"] == "simulation_plan" for r in model.calls) == 1
    handler([{"role": "user", "content": "同意"}], conversation_id="four")
    assert not calls and len(model.calls) == count
    handler([{"role": "user", "content": "同意"}], conversation_id="four")
    assert len(calls) == 1 and len(model.calls) == count


def test_malformed_paid_plan_still_records_usage():
    model = SimulationModel()
    original = model.__class__.__call__

    class PaidInvalid(SimulationModel):
        def __call__(self, request):
            if request.get("mode") == "simulation_plan":
                return {
                    "bad": True,
                    "_llm_usage": {"calls": 1, "input_tokens": 2, "output_tokens": 3, "cost": 0.1},
                }
            return original(self, request)

    result = run_tool_step(
        {},
        _session(),
        registry=create_tool_registry({"check_convergence": lambda **_: {}}),
        agent_client=PaidInvalid(),
        execution_mode="interactive",
    )
    assert result["status"] == "rejected"
    assert result["state"]["budget_usage"]["llm"]["calls"] == 1


def test_planner_network_failure_is_closed_without_dispatch():
    class Broken(SimulationModel):
        def __call__(self, request):
            if request.get("mode") == "simulation_plan":
                raise ConnectionError("synthetic unavailable provider")
            return super().__call__(request)

    calls = []
    result = run_tool_step(
        {},
        _session(),
        registry=create_tool_registry({"check_convergence": lambda **kw: calls.append(kw)}),
        agent_client=Broken(),
        execution_mode="interactive",
    )
    assert result["status"] == "rejected" and not calls


@pytest.mark.parametrize(
    "result",
    [None, {"status": "awaiting_approval"}, {"status": "execution_reconciliation_required"}],
)
def test_incomplete_result_is_not_reported_success(result):
    from phase_agent.agents.simulation.agent import inspect_execution

    report = inspect_execution({"tool": "pause_search"}, {"status": "completed", "result": result})
    assert report["tool_completed"] is False
    assert report["next"] == "return_to_strategic"
