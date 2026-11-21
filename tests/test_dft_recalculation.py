from copy import deepcopy
import json
from types import SimpleNamespace
from threading import RLock

import pytest

from phase_agent.agents.simulation.dft_recalculation import review, validate_action
from phase_agent.agents.simulation.agent import prepare_proposal
from phase_agent.policies.workflow_policy import needs_human
from phase_agent.tools.workflows.create_workflow_handlers import _restart_failed_task


def make_state():
    return {
        "confirmed_config_version": "c1",
        "active_model_version": "m1",
        "tasks": [
            {
                "task_id": name,
                "task_key": name,
                "stage": "dft_single_point",
                "status": "failed",
                "model_version": "m1",
                "config_version": "c1",
                "structure_id": "structure-" + name,
                "planned_relative_cost": 3,
                "error": "SCF failed",
            }
            for name in ("a", "b")
        ],
    }


class Simulation:
    supports_simulation = True

    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        if request["mode"] == "simulation_diagnosis":
            return {
                "assessment": "SCF convergence failed in the recorded tasks",
                "evidence": ["execution.operation_outcome"],
                "recommendation": "request_human_review",
                "limitations": "The actual cause requires logs",
            }
        return {
            "action_hash": request["action_hash"],
            "context_binding": request["context_binding"],
            "rationale": "Review frozen retry tasks",
            "checks": ["Budget and task identity"],
            "steps": [
                {"kind": "preflight", "description": "Check targets"},
                {
                    "kind": "invoke",
                    "description": "Register retries",
                    "tool": "restart_failed_task",
                },
                {"kind": "verify", "description": "Check receipts"},
            ],
            "failure_policy": "return_to_strategic",
        }


def action(state):
    result = review(state, ["a", "b"], Simulation())
    return {
        "tool": "restart_failed_task",
        "target_ids": ["a", "b"],
        "task_key": "reviewed-batch",
        "parameters": {
            "repair_profile": "dft_recalculation",
            "strategies": {"a": "original", "b": "original"},
            "failure_evidence_digest": result["review"]["failure_evidence_digest"],
            "diagnosis": result["diagnosis"],
        },
        "budget": 6,
    }


def test_review_works_below_failure_threshold_and_requires_real_diagnosis():
    state = make_state()
    before = deepcopy(state)
    model = Simulation()
    result = review(state, ["a", "b"], model)
    assert state == before and result["execution_authorized"] is False
    assert result["review"]["estimated_relative_cost"] == 6
    assert "recalculation feasibility" in model.requests[0]["instruction"]
    assert result["diagnosis"]["evidence"] == ["execution.operation_outcome"]
    assert review(state, ["a"], None)["diagnosis_unavailable"]
    assert needs_human(action(state), {}) is True


@pytest.mark.parametrize("change", ["status", "model", "config", "attempt", "cost"])
def test_frozen_review_rejects_changed_facts(change):
    state = make_state()
    proposal = action(state)
    if change == "status":
        state["tasks"][0]["status"] = "completed"
    elif change == "model":
        state["active_model_version"] = "m2"
    elif change == "config":
        state["confirmed_config_version"] = "c2"
    elif change == "attempt":
        state["tasks"].append({"task_id": "r", "retry_of": "a", "status": "pending"})
    else:
        state["tasks"][0]["planned_relative_cost"] = 9
    with pytest.raises(ValueError):
        validate_action(proposal, state)



def test_new_review_blocks_target_with_existing_retry():
    state = make_state()
    state["tasks"].append({"task_id": "r", "retry_of": "a", "status": "pending"})
    proposal = action(state)
    assert review(state, ["a"])["review"]["prior_attempts"][0]["task_id"] == "r"
    with pytest.raises(ValueError, match="existing_retry_requires_review"):
        validate_action(proposal, state)


def test_review_rejects_unknown_cost_duplicate_and_non_dft():
    state = make_state()
    state["tasks"][0]["planned_relative_cost"] = None
    proposal = action(state)
    with pytest.raises(ValueError, match="cost_unknown"):
        validate_action(proposal, state)
    with pytest.raises(ValueError, match="unique"):
        review(state, ["a", "a"])
    state["tasks"][0]["stage"] = "deep_search"
    with pytest.raises(ValueError, match="terminal_dft"):
        review(state, ["a"])


def test_batch_registration_is_atomic_and_reuses_budget():
    state = make_state()
    proposal = action(state)
    original = deepcopy(state)
    context = {
        "event_state": state,
        "config_version": "c1",
        "effective_config": {"budgets": {"stage_limits": {"dft_single_point": {"max_tasks": 10}}}},
    }
    result = _restart_failed_task(action=proposal, context=context)
    assert result["status"] == "prepared"
    assert state == original
    assert [t["retry_of"] for t in result["tasks"]] == ["a", "b"]
    assert all(t["status"] == "pending" for t in result["tasks"])
    assert len(result["state"]["budget_reservations"]) == 2
    with pytest.raises(ValueError, match="evidence_changed"):
        _restart_failed_task(action=proposal, context={**context, "event_state": result["state"]})


def test_ordinary_simulation_precedes_frozen_plan_and_unavailable_blocks():
    state = make_state()
    proposal = {
        "raw_action": {
            "tool": "restart_failed_task",
            "target_ids": ["a", "b"],
            "budget": 6,
            "parameters": {
                "repair_profile": "dft_recalculation",
                "strategies": {"a": "original", "b": "original"},
            },
        }
    }
    model = Simulation()
    result = prepare_proposal(
        proposal, model, state=state, config={}, registry={"restart_failed_task": {}}
    )
    assert [r["mode"] for r in model.requests] == ["simulation_diagnosis", "simulation_plan"]
    assert result["raw_action"]["parameters"]["diagnosis"]["assessment"].startswith("SCF")
    assert "failure_evidence_digest" not in proposal["raw_action"]["parameters"]
    with pytest.raises(ValueError, match="diagnosis_unavailable"):
        prepare_proposal(
            proposal, None, state=state, config={}, registry={"restart_failed_task": {}}
        )


def test_native_read_only_mcp_review_uses_no_model_or_project_write(tmp_path):
    from phase_agent.runtime.codex.mcp import CodexMcpServer

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(make_state()), encoding="utf8")
    config = tmp_path / "agent_runtime.json"
    config.write_text(json.dumps({"state_path": str(state_path)}), encoding="utf8")
    server = CodexMcpServer(runtime_config=config, read_only=True)
    before = state_path.read_bytes()
    result = server._call_tool("phase_dft_recalculation_review", {"target_ids": ["a", "b"]})
    assert result["diagnosis_request"]["agent_role"] == "simulation"
    assert result["diagnosis_request"]["allowed_tools"] == []
    assert not result["execution_authorized"] and state_path.read_bytes() == before


def test_approval_and_replay_through_existing_workflow():
    from tests.test_event_loop import _session
    from tests.test_decision_team import FakeTeam
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
    from phase_agent.tools.workflows.run_tool_step import run_tool_step

    session = _session()
    state = make_state()
    state["confirmed_config_version"] = session["confirmed_snapshot"]["config_version"]
    for t in state["tasks"]:
        t["config_version"] = state["confirmed_config_version"]
    for n in range(10):
        state["tasks"].append(
            {
                "task_id": "ok" + str(n),
                "task_key": "ok" + str(n),
                "stage": "dft_single_point",
                "status": "completed",
                "model_version": "m1",
                "upload_operation_id": "batch",
            }
        )
    for t in state["tasks"][:2]:
        t["upload_operation_id"] = "batch"

    class FourRole(FakeTeam):
        supports_simulation = True

        def __init__(self):
            super().__init__(["exploration", "reliability"])
            self.expert_client = self

        def __call__(self, request):
            if request.get("mode") in {"simulation_diagnosis", "simulation_plan"}:
                self.calls.append(deepcopy(request))
                return Simulation()(request)
            if request.get("mode") == "autonomous_search":
                self.calls.append(deepcopy(request))
                return {
                    "tool": "restart_failed_task",
                    "task_key": "retry-review",
                    "target_ids": ["a", "b"],
                    "budget": 6,
                    "parameters": {
                        "repair_profile": "dft_recalculation",
                        "strategies": {"a": "original", "b": "original"},
                    },
                    "reason": "Review failed DFT tasks",
                    "post_dft_review": {
                        "choice": "retry_failed",
                        "stop_status": "continue",
                        "finetune_recommendation": "insufficient_evidence",
                        **{
                            name: "Evidence remains incomplete"
                            for name in (
                                "error_assessment",
                                "coverage_assessment",
                                "finetune_assessment",
                                "search_assessment",
                                "reference_assessment",
                                "round_findings",
                                "limitations",
                                "dft_assessment",
                                "convergence_assessment",
                            )
                        },
                    },
                }
            return super().__call__(request)

    model = FourRole()
    calls = []

    def handler(*, action, context):
        calls.append(deepcopy(action))
        return _restart_failed_task(action=action, context=context)

    opts = {
        "session": session,
        "agent_client": model,
        "registry": create_tool_registry({"restart_failed_task": handler}),
        "invocation_id": "retry-test",
        "execution_mode": "interactive",
        "context": {"effective_config": session["confirmed_snapshot"]["config"]},
    }
    first = run_tool_step(state, **opts)
    assert first["status"] == "awaiting_approval", first
    assert not calls
    frozen = first["agent_proposal"]["raw_action"]
    assert frozen["parameters"]["diagnosis"]["evidence"] == ["execution.operation_outcome"]
    count = len(model.calls)
    rejected = run_tool_step(deepcopy(first["state"]), human_feedback="reject", **opts)
    assert not calls and len(model.calls) == count
    assert not rejected.get("execution")
    stale = deepcopy(first["state"])
    next(t for t in stale["tasks"] if t["task_id"] == "a")["error"] = "changed evidence"
    denied = run_tool_step(stale, human_feedback="approve", **opts)
    assert denied["status"] == "rejected" and not calls
    approved = run_tool_step(first["state"], human_feedback="approve", **opts)
    assert len(calls) == 1 and len(model.calls) == count, approved
    retries = [t for t in approved["state"]["tasks"] if t.get("retry_of") in {"a", "b"}]
    assert len(retries) == 2 and all(t["status"] == "pending" for t in retries), approved.get(
        "execution"
    )
    run_tool_step(approved["state"], human_feedback="approve", **opts)
    assert len(calls) == 1 and len(model.calls) == count


def test_native_team_uses_bound_diagnosis_and_requests_simulation(tmp_path):
    from tests.test_codex_native_team import make_control, team, advice

    state = make_state()
    control, context = make_control(tmp_path)
    control.state_path.write_text(json.dumps(state), encoding="utf8")
    context["allowed_tools"] = ["restart_failed_task", "pause_search"]
    t = team(control)
    view = t.start(question="Review failed DFT retries", consult=[])
    proposal = action(state)
    view = t.draft(view["team_id"], 1, proposal, rationale="Review limited retries")
    assert set(view["requests"]) == {"exploration", "reliability"}
    for role in ("exploration", "reliability"):
        opinion = advice(view, role)
        opinion["options"] = []
        t.advise(view["team_id"], 1, opinion)
    draft = t.draft(view["team_id"], 1, proposal, rationale="Review limited retries")
    assert draft["status"] == "planning_simulation"
    assert draft["requests"]["simulation"]["agent_role"] == "simulation"
    assert draft["execution_approved"] is False
    before = control.state_path.read_bytes()
    plan = Simulation()(draft["requests"]["simulation"])
    reviewed = t.simulate(view["team_id"], 1, plan)
    assert reviewed["status"] == "awaiting_human_input"
    assert reviewed["execution_approved"] is False
    assert control.state_path.read_bytes() == before


def test_batch_budget_shortfall_and_mid_batch_failure_are_closed():
    state = make_state()
    proposal = action(state)
    proposal["budget"] = 5
    with pytest.raises(ValueError, match="budget_below"):
        validate_action(proposal, state)
    proposal["budget"] = 6
    before = deepcopy(state)
    context = {
        "event_state": state,
        "config_version": "c1",
        "effective_config": {
            "budgets": {
                "enforcement": "hard",
                "stage_limits": {"dft_single_point": {"max_tasks": 1}},
            }
        },
    }
    result = _restart_failed_task(action=proposal, context=context)
    assert result["status"] == "budget_exhausted"
    assert state == before and result["state"] is state


def test_review_message_is_not_threshold_claim_and_invalid_usage_is_counted():
    state = make_state()

    class Invalid(Simulation):
        def __call__(self, request):
            self.requests.append(deepcopy(request))
            return {
                "invalid": True,
                "_llm_usage": {"calls": 1, "input_tokens": 10, "output_tokens": 1, "cost": None},
            }

    model = Invalid()
    result = review(state, ["a"], model)
    assert "threshold" not in model.requests[0]["observations"]["execution.error"]
    assert result["diagnosis_unavailable"]
    assert result["llm_usage"]["calls"] == 1


def test_deterministic_blocker_never_reports_success_or_calls_model():
    from phase_agent.agents.simulation.agent import inspect_execution

    model = Simulation()
    report = inspect_execution(
        {"tool": "restart_failed_task"},
        {
            "status": "completed",
            "result": {"status": "budget_exhausted", "reason": ["stage_task_limit"]},
        },
        model,
    )
    assert not report["tool_completed"] and report["diagnosis_source"] == "program"
    assert not model.requests
    report = inspect_execution(
        {"tool": "restart_failed_task"},
        {"status": "completed", "result": {"status": "already_prepared"}},
        model,
    )
    assert not report["tool_completed"] and not model.requests


def test_retry_pending_dft_preserves_other_waits():
    from phase_agent.tools.workflows.lifecycle_recovery import _failed_retry_input_only

    state = make_state()
    state["tasks"].append({"task_id": "p", "stage": "dft_single_point", "status": "pending"})
    state["pending_dft_recovery_question"] = {"pending_task_ids": ["p"]}
    state["pending_execution_policies"] = {"r": {"agent_proposal": {"raw_action": action(state)}}}
    snapshot = deepcopy(state)
    frame = {"feedback": {"state": state}, "agent_client": None, "human_feedback": "approve"}
    assert _failed_retry_input_only(frame)
    assert state == snapshot
