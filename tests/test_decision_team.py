"""Three-role scientific reasoning with fake clients and no execution tools."""

from copy import deepcopy
import pytest
from phase_agent.evidence.snapshot import evidence_snapshot
from phase_agent.graphs.decision_team.graph import request_with_langgraph
from phase_agent.models.capabilities import copy_model_capabilities


def payload():
    return {
        "mode": "autonomous_search",
        "allowed_tools": ["pause_search"],
        "state": {
            "confirmed_config_version": "c1",
            "active_model_version": "m1",
            "current_project": {"project_path": "synthetic-project"},
            "tasks": [],
        },
        "decision_context": {},
        "instruction": "Assess this synthetic project.",
    }


class FakeTeam:
    supports_expert_advice = True

    def __init__(self, roles=None):
        self.roles = roles if roles is not None else ["exploration", "reliability"]
        self.calls = []
        self.bad = None

    def __call__(self, request):
        self.calls.append(deepcopy(request))
        mode = request.get("mode")
        if mode == "team_consultation":
            result = {
                "consult": self.roles,
                "question": "What evidence is still missing?",
                "snapshot_id": request["evidence_snapshot"]["snapshot_id"],
            }
        elif mode == "expert_assessment":
            if self.bad == "failure":
                error = RuntimeError("synthetic expert failure")
                error.llm_usage = {"calls": 1, "input_tokens": 2, "output_tokens": 1, "cost": None}
                raise error
            result = {
                "role": request["agent_role"],
                "snapshot_id": request["evidence_snapshot"]["snapshot_id"],
                "assessment": "Evidence is insufficient.",
                "recommendation": "defer",
                "options": [],
                "evidence_refs": [],
                "expected_outcome": "Unknown until measured.",
                "limitations": "Synthetic evidence only.",
            }
            if self.bad == "stale":
                result["snapshot_id"] = "old"
            if self.bad == "role":
                result["role"] = "coordinator"
            if self.bad == "reference":
                result["evidence_refs"] = ["invented-reference"]
            if self.bad == "execute":
                result["tool"] = "update_mlip"
        else:
            result = {"tool": "pause_search", "parameters": {}, "reason": "Synthetic assessment"}
        return {
            **result,
            "_llm_usage": {"calls": 1, "input_tokens": 2, "output_tokens": 1, "cost": None},
        }


def test_team_consults_both_experts_then_validates_one_final_action():
    client = FakeTeam()
    original = payload()
    before = deepcopy(original)
    result = request_with_langgraph(client, original)
    assert [r["mode"] for r in client.calls] == [
        "team_consultation",
        "expert_assessment",
        "expert_assessment",
        "autonomous_search",
    ]
    assert [r.get("agent_role") for r in client.calls[1:3]] == ["exploration", "reliability"]
    assert result["tool"] == "pause_search"
    assert result["_llm_usage"]["calls"] == 4
    assert result["_llm_usage"]["cost"] is None
    assert len(result["_decision_team"]["advice"]) == 2
    assert len(client.calls[-1]["decision_context"]["expert_advice"]) == 2
    assert original == before


@pytest.mark.parametrize("roles", [[], ["exploration"], ["reliability"]])
def test_coordinator_chooses_consultation_scope_without_selecting_science_in_code(roles):
    client = FakeTeam(roles)
    result = request_with_langgraph(client, payload())
    assert len(client.calls) == 2 + len(roles)
    assert len(result["_decision_team"]["advice"]) == len(roles)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("_validates_original_facts", True),
        ("_codex_structured", True),
        ("_decision_source", "codex"),
    ],
)
def test_frozen_reuse_and_codex_operator_never_consult_project_model(flag, value):
    client = FakeTeam()
    setattr(client, flag, value)
    result = request_with_langgraph(client, payload())
    assert len(client.calls) == 1
    assert "_decision_team" not in result


def test_legacy_callable_retains_existing_single_proposal_contract():
    calls = []

    def legacy(request):
        calls.append(request)
        return {"tool": "pause_search", "parameters": {}}

    assert request_with_langgraph(legacy, payload())["tool"] == "pause_search"
    assert len(calls) == 1


@pytest.mark.parametrize("bad", ["stale", "role", "reference", "execute"])
def test_invalid_expert_output_blocks_final_proposal_and_preserves_usage(bad):
    client = FakeTeam()
    client.bad = bad
    with pytest.raises(ValueError) as caught:
        request_with_langgraph(client, payload())
    assert all(r["mode"] != "autonomous_search" for r in client.calls)
    assert caught.value.llm_usage["calls"] == 2


def test_expert_failure_never_falls_back_to_unreviewed_action():
    client = FakeTeam()
    client.bad = "failure"
    with pytest.raises(RuntimeError) as caught:
        request_with_langgraph(client, payload())
    assert caught.value.llm_usage["calls"] == 2
    assert len(client.calls) == 2


def test_snapshot_changes_when_same_model_receives_new_results():
    first = payload()
    second = deepcopy(first)
    second["state"]["tasks"] = [{"task_id": "t1", "status": "completed"}]
    assert evidence_snapshot(first)["snapshot_id"] != evidence_snapshot(second)["snapshot_id"]
    assert evidence_snapshot(first) == evidence_snapshot(deepcopy(first))


def test_capabilities_survive_runtime_wrapper():
    client = FakeTeam()
    client.expert_client = client
    client._validates_original_facts = True

    def wrapper(request):
        return client(request)

    copy_model_capabilities(client, wrapper)
    assert wrapper.supports_expert_advice is True
    assert wrapper.expert_client is client
    assert wrapper._validates_original_facts is True


def test_routine_semantic_dialogue_does_not_consult_experts():
    client = FakeTeam()
    client._semantic_dialogue = True

    def answer(request):
        return {"kind": "answer", "answer": "Synthetic status."}

    copy_model_capabilities(client, answer)
    answer._semantic_dialogue = True
    data = payload()
    data["unified_dialogue"] = True
    assert request_with_langgraph(answer, data)["kind"] == "answer"


def test_actual_chat_runs_team_once_and_approval_reuses_frozen_action(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    from phase_agent.tools.step_runner.file_protocol import read_json

    class ConversationTeam(FakeTeam):
        def __call__(self, request):
            response = super().__call__(request)
            if request.get("mode") == "autonomous_search":
                response.update(
                    tool="check_convergence",
                    parameters={},
                    budget=0,
                    reason="Assess recorded evidence",
                    expected_purpose="Check remaining gaps",
                )
            return response

    client = ConversationTeam()
    executed = []

    def check(**kwargs):
        executed.append(kwargs)
        return {"converged": False, "budget_exhausted": False}

    handler = make_handler(tmp_path, client, {"check_convergence": check})
    reply = handler([{"role": "user", "content": "分析并建议下一步"}], conversation_id="team-entry")
    saved = read_json(handler.state_path, {})
    pending = saved.get("pending_execution_policies") or {}
    assert len(pending) == 1, reply
    raw = next(iter(pending.values()))["agent_proposal"]["raw_action"]
    assert raw["tool"] == "check_convergence"
    assert len(raw["_decision_team"]["advice"]) == 2
    assert sum(r["mode"] == "team_consultation" for r in client.calls) == 1
    assert not executed
    call_count = len(client.calls)
    handler([{"role": "user", "content": "同意"}], conversation_id="team-entry")
    assert len(executed) == 1
    assert len(client.calls) == call_count
    handler([{"role": "user", "content": "同意"}], conversation_id="team-entry")
    assert len(executed) == 1 and len(client.calls) == call_count


def test_studio_exposes_the_same_decision_graph_without_a_second_scheduler():
    from phase_agent.graphs.dialogue.graph import build_project_turn_graph
    from phase_agent.graphs.decision_team.graph import build_proposal_graph
    from phase_agent.runtime.loaded_code import WATCHED_MODULES

    view = dict(build_project_turn_graph().get_subgraphs())["decide"]
    actual = build_proposal_graph()
    assert set(view.nodes) == set(actual.nodes)
    assert all(view.nodes[name] is actual.nodes[name] for name in actual.nodes)
    assert actual.checkpointer is False and view.checkpointer is None
    assert {"exploration", "reliability", "coordinated_proposal"} <= set(
        build_proposal_graph().nodes
    )
    assert "phase_agent.agents.exploration.prompts" in WATCHED_MODULES
    assert "phase_agent.agents.reliability.prompts" in WATCHED_MODULES


def test_summary_first_transport_preserves_validated_expert_advice_for_coordinator():
    from phase_agent.evidence.requests import request_summary_first

    client = FakeTeam()
    client.expert_client = client

    def contextualized(request):
        return request_summary_first(client, request)

    copy_model_capabilities(client, contextualized)
    data = payload()
    data["decision_context"].update(
        available_budget={"relative_cost": 0}, latest_convergence_check={"converged": False}
    )
    result = request_with_langgraph(contextualized, data)
    final = [r for r in client.calls if r["mode"] == "autonomous_search"][-1]
    assert len(final["decision_context"].get("expert_advice") or []) == 2
    assert (
        final["decision_context"]["evidence_snapshot_id"] == result["_decision_team"]["snapshot_id"]
    )


def test_legacy_proposal_does_not_deepcopy_large_owned_history():
    class OwnedHistory(list):
        def __deepcopy__(self, memo):
            raise AssertionError("Owned history must not be copied at the proposal boundary")

    data = payload()
    data["state"]["event_history"] = OwnedHistory([{"event": "synthetic"}])

    def legacy(request):
        return {"tool": "pause_search", "parameters": {}}

    assert request_with_langgraph(legacy, data)["tool"] == "pause_search"
