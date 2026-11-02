from copy import deepcopy
import json
from pathlib import Path
import zipfile
import pytest

from tests.test_simulation_agent import SimulationModel
from tests.test_codex_native_team import make_control, team
from phase_agent.agents.simulation.agent import validate_plan
from phase_agent.evidence.snapshot import evidence_snapshot
from phase_agent.agents.contracts import validate_advice
from phase_agent.policies.training_handoff_scope import proposed_scope, bind_job, check_handoff


def test_native_simulation_must_finish_before_human_direction(tmp_path):
    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["prepare_local_batch_files"]
    t = team(control)
    started = t.start(question="prepare reviewed inputs", consult=[])
    draft = t.draft(
        started["team_id"],
        1,
        {"tool": "prepare_local_batch_files", "task_key": "test", "parameters": {"batch_id": "b"}},
        rationale="prepare only",
    )
    assert draft["status"] == "planning_simulation"
    request = draft["requests"]["simulation"]
    model = SimulationModel()
    planned = t.simulate(started["team_id"], 1, model(request))
    assert planned["status"] == "awaiting_human_input"
    assert planned["execution_approved"] is False
    accepted = t.feedback(started["team_id"], 1, "accept", "accept direction")
    assert accepted["status"] == "ready_to_propose"
    assert accepted["direction_review"]["execution_approved"] is False


def test_native_stale_plan_cannot_be_registered(tmp_path):
    control, context = make_control(tmp_path)
    context["allowed_tools"] = ["prepare_local_batch_files"]
    t = team(control)
    view = t.start(question="prepare", consult=[])
    draft = t.draft(view["team_id"], 1, {"tool": "prepare_local_batch_files"}, rationale="prepare")
    plan = SimulationModel()(draft["requests"]["simulation"])
    plan["action_hash"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        t.simulate(view["team_id"], 1, plan)
    assert t.read()["status"] == "planning_simulation"


def test_operation_catalog_changes_snapshot():
    data = {
        "allowed_tools": ["pause_search"],
        "scientific_operations": [{"operation": "a", "tool": "pause_search"}],
    }
    changed = deepcopy(data)
    changed["scientific_operations"][0]["operation"] = "b"
    assert evidence_snapshot(data)["snapshot_id"] != evidence_snapshot(changed)["snapshot_id"]


def test_semantic_draft_without_expert_advice_is_not_regenerated():
    from tests.test_decision_team import FakeTeam, payload
    from phase_agent.graphs.decision_team.graph import request_with_langgraph

    client = FakeTeam([])
    client._semantic_dialogue = True
    request_with_langgraph(client, payload())
    assert [r["mode"] for r in client.calls] == ["autonomous_search", "team_consultation"]


def test_training_scope_is_bound_to_original_report_and_config():
    config = {"remote_training_validation": {"data_version": "v1"}}
    job = {"directory": "synthetic", "report": {"training_round": "r1"}}
    job["handoff_authorization"] = bind_job(proposed_scope({"parameters": {}}, config), "j", job)
    check_handoff(job, config, "manifest")
    with pytest.raises(ValueError, match="new_approval"):
        check_handoff(job, config, "validation")
    changed = deepcopy(job)
    changed["report"]["training_round"] = "r2"
    with pytest.raises(ValueError, match="authorization_changed"):
        check_handoff(changed, config, "manifest")
    with pytest.raises(ValueError, match="authorization_changed"):
        check_handoff(job, {"remote_training_validation": {"data_version": "v2"}}, "manifest")


def test_real_dispatch_supplies_simulation_execution_role():
    from tests.test_event_loop import _session
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.dispatch.create_tool_registry import create_tool_registry

    seen = []
    result = run_tool_step(
        {},
        _session(),
        registry=create_tool_registry(
            {
                "check_convergence": lambda **kw: (
                    seen.append(kw["context"].get("executor_role")) or {"converged": False}
                )
            }
        ),
        agent_client=lambda _: {"tool": "check_convergence"},
        execution_mode="autonomous",
    )
    assert result["status"] == "completed"
    assert seen == ["simulation"]


def test_explicit_training_followup_uses_existing_preparation(tmp_path):
    from tests.test_training_handoff import setup, config
    from phase_agent.tools.local.prepare_training_followup import (
        request_binding,
        prepare_training_followup,
    )

    state, _, inputs = setup(tmp_path)
    settings = config(tmp_path)
    action = {
        "tool": "prepare_local_batch_files",
        "parameters": {"mode": "training_handoff_inputs", "job_key": "j", "kind": "validation"},
    }
    action["_training_handoff_request"] = request_binding(action, state, settings)
    result = prepare_training_followup(
        action=action, context={"event_state": state, "effective_config": settings}
    )
    assert result["status"] == "prepared" and result["submitted"] is False
    assert (inputs / "GPU_validation.sh").is_file()
    assert (
        "validation"
        in result["state"]["remote_finetune_jobs"]["j"]["handoff_authorization"]["kinds"]
    )


def test_ordering_export_and_import_are_minimal_and_checksummed(tmp_path, monkeypatch):
    from tests.test_ordering_agent import fixture
    from phase_agent.tools.workflows.run_tool_step import run_tool_step
    from phase_agent.tools.local.prepare_ordering_search import recover_ordering_results
    from phase_agent.tools.remote.export_ordering_result import export_ordering_result
    from Process_AL_MC import run_ordering_search

    session, context, action, registry = fixture(tmp_path)
    options = dict(
        session=session,
        registry=registry,
        context=context,
        agent_client=lambda _: action,
        execution_mode="interactive",
        invocation_id="ordering",
    )
    first = run_tool_step({}, **options)
    prepared = run_tool_step(first["state"], human_feedback="approve", **options)
    record = next(iter(prepared["state"]["ordering_searches"].values()))
    root = Path(record["directory"])
    assert not (root / "runtime/mc_backend").exists()

    def evaluate(structure, context):
        return {
            "structure": structure.as_dict(),
            "energy_per_atom": -1,
            "phase": "P2",
            "converged": True,
        }

    run_ordering_search(root, evaluator=evaluate)
    (root / "large.model").write_text("not needed locally")
    from phase_agent.tools.local.prepare_ordering_search import prepare_ordering_inputs

    # Exercise repackaging independently of the backend's plan-reuse policy.
    monkeypatch.setattr(
        "Process_AL_MC.prepare_ordering_search", lambda plan, directory: {"status": "reused"}
    )
    reused = prepare_ordering_inputs(
        action=first["agent_proposal"]["raw_action"],
        context={**context, "event_state": prepared["state"]},
    )
    with zipfile.ZipFile(reused["input_archive"]) as package:
        assert "result.json" not in package.namelist()
        assert "large.model" not in package.namelist()
    archive = export_ordering_result(root)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        assert "transfer_manifest.json" in names and "result_manifest.json" in names
        assert not any(
            "runtime" in name or name.endswith(".model") or name.endswith(".py") for name in names
        )
    result = recover_ordering_results(
        action={"parameters": {"plan_id": record["plan_id"], "results_archive": archive}},
        context={"event_state": prepared["state"]},
    )
    assert result["status"] == "completed"
    assert Path(result["ordering_result"]["global_ranking_csv"]).is_file()
