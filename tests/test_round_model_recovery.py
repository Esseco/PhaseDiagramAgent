"""Original-model DFT validation, simple round records and human partial recovery."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from pymatgen.core import Lattice, Structure
import pytest

from analysis_layer.feedback.dft_result_products import record_dft_products, export_dft_products
from analysis_layer.state.summarize_model_rounds import summarize_model_rounds
from analysis_layer.state.dft_round_status import dft_recovery_rounds
from execution_layer.state.dft_recovery_decision import classify_dft_recovery_reply, update_dft_recovery_question
from execution_layer.state.reconcile_task_results import reconcile_task_results
from execution_layer.remote.summarize_manual_upload_wait import summarize_manual_upload_wait
from execution_layer.workflows.comparison_model_registry import remember_comparison_models
from execution_layer.workflows.create_dft_comparison_evaluator import create_dft_comparison_evaluator
from run.main import run_workflow
from run.agent_api import RunWorkflowChatHandler
from run.workflow_reply_presentation import format_workflow_reply


def dft_result(task_id="D1", version="m1"):
    structure = Structure(Lattice.cubic(4), ["Na", "O"], [[0, 0, 0], [.5, .5, .5]])
    return {"task_id": task_id, "task_key": "key-"+task_id, "structure_id": "S-"+task_id,
            "model_version": version, "stage": "dft_single_point", "status": "completed", "converged": True,
            "outputs": {"structure": structure.as_dict(), "composition": {"Na": 1, "O": 1}, "atom_count": 2,
                        "energy": -8., "training_energy": -8., "energy_unit": "eV", "forces": [[0, 0, 0]]*2,
                        "forces_unit": "eV/angstrom", "training_ready": True, "training_frame_index": 0,
                        "final_frame_index": 0, "final_frame_valid": True, "actual_phase": "O3"}}


def task(task_id="D1", status="pending", operation="op1", version="m1"):
    return {"task_id": task_id, "task_key": "key-"+task_id, "structure_id": "S-"+task_id,
            "stage": "dft_single_point", "status": status, "model_version": version, "batch_id": "remote-"+task_id,
            "upload_operation_id": operation, "search_group_index": 1,
            "input_path": f"upload/MLIP-round-0001_{version}/Search-group-0001/DFT-round-0001_{operation}/"
                          f"DFT-single-point/submission/{task_id}/task.json"}


def partial_state():
    return {"tasks": [task("D1", "completed"), task("D2"), task("D3")],
            "processed_task_ids": ["D1"], "budget_reservations": {"key-D2": {"status": "reserved", "reserved_cost": 30}},
            "upload_layout": {"model_rounds": {"m1": 1}}}


def prediction(**kwargs):
    return {"model_version": kwargs["result"]["model_version"], "energy": -7., "energy_unit": "eV",
            "forces": [[.2]*3]*2, "forces_unit": "eV/angstrom"}


def test_old_round_uses_original_model_after_active_model_changes(tmp_path, monkeypatch):
    old, new = tmp_path / "old.model", tmp_path / "new.model"
    old.write_text("old weights")
    new.write_text("new weights")
    state = {}
    remember_comparison_models(state, {"mlip": {"version": "m1", "model_path": str(old)}})
    original_digest = state["model_registry"]["m1"]["comparison_model_sha256"]
    state["active_model"] = {"version": "m2", "model_path": str(new)}
    config = {"state_path": str(tmp_path / "state.json"), "mlip": state["active_model"], "dft_comparison_location": "local"}
    calls = []
    def run(structure, **kwargs):
        calls.append(kwargs)
        return {"status": "completed", "energy": -7., "energy_unit": "eV",
                "forces": [[0, 0, 0]]*2, "forces_unit": "eV/angstrom"}
    monkeypatch.setattr("scientific_layer.mlip.run_mace_subprocess.run_mace_with_py_mace", run)
    evaluated = create_dft_comparison_evaluator(config, state=state)(result=dft_result(), structure_id="S-D1", manager=None)
    assert calls[0]["model_path"] == str(old)
    assert calls[0]["operation"] == "predict"
    assert evaluated["model_version"] == "m1" and evaluated["model_sha256"] == original_digest
    assert evaluated["geometry"] == "DFT_final_frame"


def test_replaced_original_model_is_not_silently_reused(tmp_path):
    model = tmp_path / "weights.model"
    model.write_text("original")
    config = {"state_path": str(tmp_path / "state.json"), "mlip": {"version": "m1", "model_path": str(model)}, "dft_comparison_location": "local"}
    state = {}
    remember_comparison_models(state, config)
    model.write_text("different model under the same version")
    with pytest.raises(ValueError, match="content changed|fingerprint"):
        create_dft_comparison_evaluator(config, state=state)(result=dft_result(), structure_id="S-D1", manager=None)


def test_missing_old_version_does_not_fall_back_to_new_model(tmp_path):
    model = tmp_path / "new.model"
    model.write_text("new")
    config = {"state_path": str(tmp_path / "state.json"), "mlip": {"version": "m2", "model_path": str(model)}}
    with pytest.raises(ValueError, match="version unavailable"):
        create_dft_comparison_evaluator(config)(result=dft_result(), structure_id="S-D1", manager=None)


def test_round_summary_counts_and_errors_are_model_specific(tmp_path):
    state = {"tasks": [task("D1", "completed"), task("D2"), task("D3", "failed")],
             "generation_history": [{"model_version": "m1", "summary": {
                 "proposed_branches": 150, "selected_branches": 119, "registered_branches": 119}}],
             "upload_layout": {"model_rounds": {"m1": 1, "m2": 2}}}
    state["tasks"] += [{"task_id": f"M{i}", "stage": "deep_search", "status": "completed", "model_version": "m1", "branch_id": f"B{i}"} for i in range(3)]
    record_dft_products(state, dft_result(), prediction)
    state["tasks"].append(task("D4", "completed", version="m2"))
    def new_prediction(**kwargs):
        return {**prediction(**kwargs), "energy": -4., "forces": [[.6]*3]*2}
    record_dft_products(state, dft_result("D4", "m2"), new_prediction)
    export_dft_products(state, tmp_path)
    rows = {r["model_version"]: r for r in state["model_round_summary"]}
    first, second = rows["m1"], rows["m2"]
    assert first["mlip_round"] == 1 and second["mlip_round"] == 2
    assert first["branches_proposed"] == 150 and first["branches_selected"] == 119
    assert first["mc_tasks"] == first["mc_completed"] == 3
    assert first["dft_tasks"] == 3 and first["dft_recovered"] == 2 and first["dft_failed"] == 1
    assert first["dft_recovery_ratio"] == pytest.approx(2/3)
    assert first["energy_mae_eV_per_atom"] == .5 and second["energy_mae_eV_per_atom"] == 2
    assert first["force_rmse_eV_per_A"] == pytest.approx(.2)
    assert second["force_rmse_eV_per_A"] == pytest.approx(.6)
    assert first["matched_dft_tasks"] == 1
    with (tmp_path / "round_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 2


def test_unknown_legacy_branch_lineage_is_not_guessed():
    state = {"tasks": [task()], "generation_history": [{"registered_ids": ["unknown-S"],
            "summary": {"proposed_branches": 100, "selected_branches": 90}}]}
    row = summarize_model_rounds(state)[0]
    assert row["branches_proposed"] is None and row["branch_count_status"] == "unknown_legacy_lineage"
    assert row["energy_mae_eV_per_atom"] is None


def test_legacy_registered_ids_can_prove_the_generation_model():
    state = {"tasks": [task()], "generation_history": [{"registered_ids": ["S-D1"],
             "summary": {"proposed_branches": 12, "selected_branches": 10, "registered_branches": 9}}]}
    assert summarize_model_rounds(state)[0]["branches_selected"] == 10


def test_partial_recovery_question_is_saved_and_continue_is_not_a_choice():
    state = update_dft_recovery_question(partial_state())
    question = state["pending_dft_recovery_question"]
    assert question["recovered_tasks"] == 1 and question["expected_tasks"] == 3
    assert question["recovery_ratio"] == pytest.approx(1/3)
    assert classify_dft_recovery_reply("继续", state) is None
    reply = format_workflow_reply({"status": "awaiting_dft_recovery_decision", "state": state,
                                  "dft_recovery_question": question}, "state.json")
    assert "1/3" in reply and "33.3%" in reply and "不再回收" in reply and "第 1 轮" in reply


def test_decline_waives_only_asked_tasks_without_faking_completion_or_settlement():
    original = update_dft_recovery_question(partial_state())
    decision = classify_dft_recovery_reply("否", original)
    state = update_dft_recovery_question(original, decision)
    assert "pending_dft_recovery_question" not in state
    assert state["pending_tasks"] == []
    assert state["budget_reservations"] == original["budget_reservations"]
    assert state["tasks"][1]["status"] == "pending" and state["tasks"][1]["recovery_wait_waived"] is True
    assert summarize_manual_upload_wait(state) is None
    assert reconcile_task_results(state)["state"]["pending_tasks"] == []
    assert dft_recovery_rounds(state)[0]["recovery_ratio"] == pytest.approx(1/3)
    assert len(dft_recovery_rounds(state)[0]["waived_task_ids"]) == 2


def test_new_round_does_not_inherit_a_previous_waiver():
    state = update_dft_recovery_question(partial_state())
    state = update_dft_recovery_question(state, classify_dft_recovery_reply("不再回收", state))
    state["tasks"].extend([task("E1", "completed", "op2"), task("E2", operation="op2")])
    state = update_dft_recovery_question(state)
    assert state["pending_dft_recovery_question"]["scope"]["upload_operation_id"] == "op2"
    assert [t["task_id"] for t in state["pending_tasks"]] == ["E2"]


def test_late_result_is_still_reconciled_after_waiving_wait():
    state = update_dft_recovery_question(partial_state())
    state = update_dft_recovery_question(state, classify_dft_recovery_reply("否", state))
    late = dft_result("D2")
    result = reconcile_task_results(state, [late])
    assert result["state"]["tasks"][1]["status"] == "completed"
    assert "D2" in result["state"]["processed_task_ids"]
    assert dft_recovery_rounds(result["state"])[0]["recovered_tasks"] == 2


def test_wait_choice_is_not_reasked_until_recovery_changes():
    state = update_dft_recovery_question(partial_state())
    state = update_dft_recovery_question(state, classify_dft_recovery_reply("继续回收", state))
    assert "pending_dft_recovery_question" not in state
    state = update_dft_recovery_question(state)
    assert "pending_dft_recovery_question" not in state
    state["tasks"][1]["status"] = "completed"
    state = update_dft_recovery_question(state)
    assert state["pending_dft_recovery_question"]["recovered_tasks"] == 2


def test_no_question_rejection_cannot_waive_tasks():
    assert classify_dft_recovery_reply("否", partial_state()) is None
    state = update_dft_recovery_question(partial_state())
    with pytest.raises(ValueError, match="changed"):
        update_dft_recovery_question(state, {"decision": "close", "question_id": "wrong"})
    assert not state["tasks"][1].get("recovery_wait_waived")


def test_ambiguous_action_approval_is_not_used_as_recovery_decision():
    state = update_dft_recovery_question(partial_state())
    state["pending_execution_policies"] = {"action1": {}}
    assert classify_dft_recovery_reply("拒绝", state) is None
    assert classify_dft_recovery_reply("同意", state) is None
    assert classify_dft_recovery_reply("不再回收", state)["decision"] == "close"


def test_chat_routes_actual_recovery_choice_without_approving_an_action(tmp_path):
    state = update_dft_recovery_question(partial_state())
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    calls = []
    def workflow(**kwargs):
        calls.append(kwargs)
        updated = update_dft_recovery_question(state, kwargs["dft_recovery_decision"])
        return {"status": "awaiting_approval", "state": updated,
                "agent_proposal": {"recommended_action": "check_convergence", "expected_purpose": "evaluate existing results"}}
    handler = RunWorkflowChatHandler({"state_path": str(path)}, workflow=workflow)
    reply = handler([{"role": "user", "content": "否"}])
    assert len(calls) == 1 and calls[0]["human_feedback"] is None
    assert calls[0]["dft_recovery_decision"]["decision"] == "close"
    assert "不再等待" in reply


def test_workflow_moves_to_llm_only_after_declining_partial_recovery(monkeypatch):
    from config_layer.defaults.default_layered_search_config import default_layered_search_config
    from config_layer.session.create_config_draft import create_config_draft
    from config_layer.session.confirm_config_snapshot import confirm_config_snapshot
    session = confirm_config_snapshot(create_config_draft(default_layered_search_config()), user_confirmed=True)
    state = partial_state()
    state["confirmed_config_version"] = session["confirmed_snapshot"]["config_version"]
    calls = []
    def event_loop(current, *args, **kwargs):
        calls.append(current)
        return {"status": "awaiting_approval", "state": current, "events": [], "steps_executed": 1}
    monkeypatch.setattr("run.main.run_event_loop", event_loop)
    first = run_workflow(None, {}, {}, session, state=state, execution_mode="interactive")
    assert first["status"] == "awaiting_dft_recovery_decision" and calls == []
    choice = classify_dft_recovery_reply("否", first["state"])
    second = run_workflow(None, {}, {}, session, state=first["state"], execution_mode="interactive", dft_recovery_decision=choice)
    assert second["status"] == "awaiting_approval" and len(calls) == 1
    assert calls[0]["pending_tasks"] == []
