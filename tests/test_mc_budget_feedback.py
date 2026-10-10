"""Budget feedback must stay attached to the pending MC action."""

from types import SimpleNamespace

from phase_agent.analysis.phase.branch_relax_hull import build_relax_hull
from phase_agent.configuration.defaults.default_layered_search_config import default_layered_search_config
from phase_agent.decisions.agent.revise_tool_proposal import revise_tool_proposal
from phase_agent.decisions.agent.resolve_mc_budget_feedback import resolve_mc_full_plan_steps
from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry
from phase_agent.tools.policy.execution_policy import build_agent_proposal
from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.tools.workflows.run_tool_step import run_tool_step
from phase_agent.runtime.agent_api import RunWorkflowChatHandler


def _pending_mc_state(maximum):
    action = {"tool": "allocate_mc_bohb", "task_key": "mc-old",
              "target_ids": ["B1"], "parameters": {"mc_budget": 300}, "budget": 300}
    return {"confirmed_config_version": "config-old",
            "confirmed_config": {"round_strategy": {"maximum_mc_budget": maximum}},
            "pending_execution_policies": {"review-1": {
                "agent_proposal": {"raw_action": action}}}}


def test_full_plan_choice_reads_frozen_preview_only():
    action = {"tool": "allocate_mc_bohb", "parameters": {
        "budget_preview": {"full_plan_steps": 9540}}}
    assert resolve_mc_full_plan_steps("完整运行", action) == 9540
    assert resolve_mc_full_plan_steps("继续", action) is None
    assert resolve_mc_full_plan_steps("完整运行", {"tool": "generate_branches"}) is None


def test_full_plan_above_cap_enters_targeted_revision(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    calls, received = [], []
    def model(payload):
        calls.append(payload)
        return {"kind": "configure", "answer": '完整方案超出配置上限，先修订预算。'}
    handler = make_handler(tmp_path, model)
    handler.config_revision_factory = lambda state: (
        lambda messages, conversation_id=None: received.append(messages[-1]["content"]) or "revision-started")
    assert handler([{"role": "user", "content": '完整运行'}]).endswith("revision-started")
    assert len(calls) == 1
    assert calls[0]["user_instruction"] == '完整运行'
    assert received == ['完整运行']


def test_pending_mc_feedback_above_confirmed_cap_enters_targeted_revision(tmp_path):
    from tests.test_unified_react_dialogue import make_handler
    calls, received = [], []
    def model(payload):
        calls.append(payload)
        return {"kind": "configure", "answer": '需要修订配置中的MC预算。'}
    handler = make_handler(tmp_path, model)
    handler.config_revision_factory = lambda state: (
        lambda messages, conversation_id=None: received.append(messages[-1]["content"]) or "revision-started")
    assert handler([{"role": "user", "content": '按 5000 步预算'}]).endswith("revision-started")
    assert len(calls) == 1
    assert calls[0]["user_instruction"] == '按 5000 步预算'
    assert received == ['按 5000 步预算']


def test_pending_mc_feedback_within_cap_revises_current_action(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, _pending_mc_state(5000))
    seen = {}

    def workflow(**kwargs):
        seen.update(kwargs)
        return {"status": "awaiting_approval", "agent_proposal": {
            "recommended_action": "allocate_mc_bohb",
            "raw_action": _pending_mc_state(5000)["pending_execution_policies"]["review-1"]["agent_proposal"]["raw_action"]}}

    handler = RunWorkflowChatHandler({"state_path": str(path)}, workflow=workflow)
    handler([{"role": "user", "content": "按 5000 步预算"}])

    assert seen["invocation_id"] == "review-1"
    assert seen["human_feedback"] == {"decision": "comment", "comment": "按 5000 步预算"}


def test_mc_budget_revision_recomputes_frozen_hull_preview():
    config = default_layered_search_config()
    config["round_strategy"]["maximum_mc_budget"] = 5000
    pool = build_relax_hull([{"branch_id": "B1", "structure_id": "S1",
        "structure_path": "final.vasp", "composition": {"Na": 1, "Fe": 1, "Mn": 1, "O": 4},
        "energy": -40.0, "energy_unit": "eV", "converged": True,
        "model_version": "m1", "stage": "relax_and_feature"}], model_version="m1")
    state = {"branch_hull_batches": {pool["version"]: pool},
                 "current_branch_hull_version": pool["version"]}
    state["phase_diagrams"] = {"mlip": {"method": "mlip", "status": "completed",
        "model_version": "m1", "version": "test-phase-v1", "entries": [{
            "structure_id": "S1", "structure_path": "final.vasp",
            "normalized_total_energy": -40.0, "ehull": 0.0,
            "ehull_unit": "eV/atom", "phase": "O3",
            "phase_identification_status": "identified"}]}}
    manager = SimpleNamespace(data={"branches": {"B1": {"P": "O3", "x": "1/2"}}})
    original = {"tool": "allocate_mc_bohb", "task_key": "mc-old", "target_ids": ["B1"],
                "parameters": {"mc_budget": 300, "seed": 7,
                               "hull_reference_version": pool["version"],
                               "exploration_fraction": 0.1}, "budget": 300}

    revision = revise_tool_proposal({"raw_action": original}, "按 5000 步预算",
        state={}, allowed_tools=["allocate_mc_bohb"], agent_client=None,
        source_state=state, manager=manager, config=config)

    action = revision["action"]
    assert revision["revision_status"] == "user_mc_budget_applied"
    assert action["parameters"]["mc_budget"] == 5000
    assert action["parameters"]["budget_preview"]["target_steps"] == 5000
    assert action["budget"] == action["parameters"]["budget_preview"]["estimated_relative_cost"]
    assert action["task_key"] != original["task_key"]

    config["agent"]["allowed_tools"] = ["allocate_mc_bohb"]
    state.update({"confirmed_config_version": "v1", "confirmed_config": config,
                  "pending_execution_policies": {"review-1": {
                      "record_id": "review-1", "agent_proposal": build_agent_proposal(original, {}),
                      "revision": 0}}})
    result = run_tool_step(state, {"status": "confirmed", "confirmed_snapshot": {
        "config_version": "v1", "config": config}},
        registry=create_tool_registry({"allocate_mc_bohb": lambda **_: {"status": "completed"}}),
        context={"manager": manager, "effective_config": config, "event_state": state},
        invocation_id="review-1", execution_mode="interactive",
        human_feedback={"decision": "comment", "comment": "按 5000 步预算"})
    assert result["status"] == "awaiting_approval"
    assert result["agent_proposal"]["raw_action"]["parameters"]["mc_budget"] == 5000
