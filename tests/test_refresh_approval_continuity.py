from copy import deepcopy

from tests.test_model_refresh_workflow import setup
from phase_agent.tools.workflows.model_refresh_gate import run_model_refresh_gate
from phase_agent.tools.local.prepare_model_refresh import prepare_model_refresh


def test_budget_block_preserves_approval_and_resumes_once(tmp_path):
    state, session, context, registry = setup(tmp_path)
    limit = context["effective_config"]["budgets"]["total_relative_cost"]
    state["budget_reservations"] = {
        "other": {"stage": "relax_and_feature", "status": "reserved", "reserved_cost": limit}
    }
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    blocked = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert blocked["status"] == "budget_exhausted"
    assert blocked["state"]["model_refresh"]["input_authorization"]["plan"]
    assert not blocked["state"]["tasks"]
    blocked["state"]["budget_reservations"]["other"]["status"] = "released"
    resumed = run_model_refresh_gate(blocked["state"], session, registry=registry, context=context)
    assert resumed["status"] == "prepared", resumed
    assert len(resumed["state"]["tasks"]) == 1
    again = run_model_refresh_gate(resumed["state"], session, registry=registry, context=context)
    assert len(again["state"]["tasks"]) == 1


def test_frozen_geometry_is_used_after_new_ledger_geometry(tmp_path):
    state, session, context, registry = setup(tmp_path)
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    action = first["action"]
    # Later ledger discovery must not replace the original approved inputs.
    context["manager"].data["structures"]["s"]["source_path"] = str(tmp_path / "missing.vasp")
    context["event_state"] = first["state"]
    result = prepare_model_refresh(action=action, context=context)
    assert result["status"] == "prepared", result
    assert (
        result["state"]["tasks"][0]["input_structure_sha256"]
        == action["parameters"]["refresh_preview"]["candidates"][0]["structure_sha256"]
    )


def test_changed_model_does_not_reuse_blocked_approval(tmp_path):
    state, session, context, registry = setup(tmp_path)
    state["budget_reservations"] = {
        "other": {
            "stage": "relax_and_feature",
            "status": "reserved",
            "reserved_cost": context["effective_config"]["budgets"]["total_relative_cost"],
        }
    }
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    blocked = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    changed = deepcopy(blocked["state"])
    changed["active_model"]["model_path"] = "/models/different.model"
    changed["budget_reservations"]["other"]["status"] = "released"
    result = run_model_refresh_gate(changed, session, registry=registry, context=context)
    assert result["status"] == "awaiting_approval"
    assert not result["state"]["tasks"]


def test_delivered_native_approval_survives_restart_before_execution(tmp_path):
    import json

    state, session, context, registry = setup(tmp_path)
    state_path = tmp_path / "state.json"
    context["state_path"] = str(state_path)
    state["confirmed_config_version"] = "c"
    state["confirmed_config"] = deepcopy(session["confirmed_snapshot"]["config"])
    state["budget_reservations"] = {
        "other": {
            "stage": "relax_and_feature",
            "status": "reserved",
            "reserved_cost": context["effective_config"]["budgets"]["total_relative_cost"],
        }
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    first = run_model_refresh_gate(state, session, registry=registry, context=context)
    blocked = run_model_refresh_gate(
        first["state"], session, registry=registry, context=context, human_feedback="同意"
    )
    assert blocked["status"] == "budget_exhausted"
    assert (tmp_path / "langgraph_approvals.sqlite").exists()
    assert not (tmp_path / "execution_receipts.sqlite").exists()
    state_path.write_text(json.dumps(blocked["state"]), encoding="utf-8")
    restored = json.loads(state_path.read_text(encoding="utf-8"))
    restored["budget_reservations"]["other"]["status"] = "released"
    result = run_model_refresh_gate(restored, session, registry=registry, context=context)
    assert result["status"] == "prepared", result


def test_generated_inputs_are_presented_as_waiting_results():
    from phase_agent.runtime.chat_state_presentation import brief_chat_state

    reply = brief_chat_state({"model_refresh": {"status": "waiting_results", "wave": 0}})
    assert "输入已生成" in reply
    assert "待确认" not in reply
