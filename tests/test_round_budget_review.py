import csv
import json
from copy import deepcopy
import pytest
from analysis_layer.state.training_result_evidence import training_result_evidence
from analysis_layer.state.round_budget_evidence import round_budget_evidence, budget_decision_outcomes
from decision_layer.agent.round_budget_review import round_budget_review_errors
from decision_layer.agent.proposal_validation import proposal_errors
from data_layer.memory.collect_memory_candidates import collect_memory_candidates
from data_layer.memory.verify_evidence_refs import verify_evidence_refs


def review(choice="supplement_dft"):
    return {"choice": choice, "reason": "Compare remaining pool and exploration", "uncertainty": "Costs unknown",
            "revisit_when": "New DFT labels arrive", "alternatives": [
                {"path": path, "expected_benefit": "Potential hull or coverage improvement",
                 "benefit_status": "hypothesis", "expected_relative_cost": None,
                 "cost_basis": "Unknown downstream cost; request estimates"}
                for path in ("supplement_dft", "new_search")]}


def report_job(tmp_path):
    directory = tmp_path / "results"; directory.mkdir()
    (directory / "training.finished.json").write_text(json.dumps({"status": "completed",
        "original_model_version": "m1", "out_of_fold_structures": 27, "committee_count": 4}))
    (directory / "models.json").write_text(json.dumps([{"model_id": f"com_{i}"} for i in range(4)]))
    fields = ["fold", "structures", "evaluation_type", "energy_MAE_meV_per_atom", "energy_RMSE_meV_per_atom",
              "force_MAE_meV_per_A", "force_RMSE_meV_per_A"]
    with (directory / "kfold_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        writer.writerow(["all_out_of_fold", 27, "grouped_cross_validation", 3.8, 6.2, 56, 90])
    return {"active_model_version": "m1", "remote_finetune_jobs": {"J1": {
        "directory": str(tmp_path), "original_model_version": "m1", "status": "inputs_prepared", "activated": False}}}


def test_finished_report_visible_without_activating_or_rewriting_job(tmp_path):
    state = report_job(tmp_path); before = deepcopy(state)
    facts = training_result_evidence(state)
    assert facts[0]["structures"] == 27 and facts[0]["model_files_verified"] is False
    assert round_budget_evidence(state)["required"] is True
    assert state == before


def test_invalid_report_never_claims_completed(tmp_path):
    state = report_job(tmp_path)
    path = tmp_path / "results" / "training.finished.json"
    data = json.loads(path.read_text()); data["original_model_version"] = "other"
    path.write_text(json.dumps(data))
    assert training_result_evidence(state)[0]["status"] == "invalid_report"
    assert "report_id" not in training_result_evidence(state)[0]


def test_report_candidate_is_idempotent_and_evidence_is_resolvable(tmp_path):
    state = collect_memory_candidates(report_job(tmp_path))
    assert len(state["memory_candidates"]) == 1
    assert collect_memory_candidates(state)["memory_candidates"] == state["memory_candidates"]
    candidate = state["memory_candidates"][0]
    assert candidate["status"] == "candidate"
    assert verify_evidence_refs(state, candidate["evidence_refs"], scope="project") == []


def test_first_round_not_forced_and_later_round_requires_both_paths():
    action = {"tool": "pause_search"}
    assert not round_budget_review_errors(action, {})
    context = {"round_budget_evidence": {"required": True}}
    assert round_budget_review_errors(action, context)
    action["round_budget_review"] = review("stop")
    assert not round_budget_review_errors(action, context)
    action["round_budget_review"]["alternatives"].pop()
    assert round_budget_review_errors(action, context)


@pytest.mark.parametrize("cost", [-1, float("nan"), float("inf"), True])
def test_false_precision_and_invalid_cost_rejected(cost):
    action = {"tool": "select_dft_candidates", "round_budget_review": review()}
    action["round_budget_review"]["alternatives"][0]["expected_relative_cost"] = cost
    assert round_budget_review_errors(action, {"round_budget_evidence": {"required": True}})


def test_outcomes_use_explicit_links_and_do_not_invent_alternative_returns():
    state = {"action_records": [{"record_id": "D1", "final_action": {"round_budget_review": review()}}],
             "tasks": [{"task_id": "T1", "parent_decision_id": "D1", "status": "completed",
                        "actual_cost": 5, "model_version": "m1"},
                       {"task_id": "unrelated", "status": "completed", "actual_cost": 999}]}
    outcome = budget_decision_outcomes(state)[0]
    assert outcome["actual_relative_cost"] == 5
    assert outcome["unselected_path_actual_benefit"] is None
    state["tasks"][0]["actual_cost"] = None
    assert budget_decision_outcomes(state)[0]["actual_relative_cost"] is None
    state["tasks"][0].update(status="running", actual_cost=2)
    assert budget_decision_outcomes(state)[0]["actual_relative_cost"] is None


def test_missing_links_remain_pending_and_old_versions_are_preserved():
    state = {"active_model_version": "m2", "action_records": [{"record_id": "D1",
             "final_action": {"round_budget_review": review()}}]}
    result = budget_decision_outcomes(state)[0]
    assert result["status"] == "awaiting_task_links" and result["model_version"] is None
    assert result["actual_relative_cost"] is None


def test_tool_must_match_budget_choice_and_expected_benefits_are_hypotheses():
    action = {"tool": "generate_branches", "round_budget_review": review()}
    ctx = {"round_budget_evidence": {"required": True}}
    assert round_budget_review_errors(action, ctx)
    action["round_budget_review"] = review("new_search")
    assert not round_budget_review_errors(action, ctx)
    action["round_budget_review"]["alternatives"][0]["benefit_status"] = "observed"
    assert round_budget_review_errors(action, ctx)


def test_completed_task_creates_budget_memory_candidate():
    state = {"active_model_version": "m1", "action_records": [{"record_id": "D1",
             "final_action": {"round_budget_review": review()}}],
             "tasks": [{"task_id": "T1", "parent_decision_id": "D1", "status": "completed",
                        "model_version": "m1", "actual_cost": {"value": 5, "unit": "relative_cost"}}]}
    updated = collect_memory_candidates(state)
    row = next(row for row in updated["memory_candidates"] if row["kind"] == "round_budget_outcome")
    assert row["facts"]["actual_relative_cost"] == 5 and row["status"] == "candidate"
    assert verify_evidence_refs(updated, row["evidence_refs"], scope="project") == []


def test_langgraph_validation_enforces_review_even_without_post_dft_assessment():
    action = {"tool": "pause_search", "parameters": {}, "budget": 0}
    payload = {"allowed_tools": ["pause_search"], "decision_context": {"round_budget_evidence": {"required": True}}}
    assert proposal_errors(action, payload)
    action["round_budget_review"] = review("stop")
    assert not proposal_errors(action, payload)


def test_large_error_is_not_a_new_search_gate():
    action = {"tool": "generate_branches", "round_budget_review": review("new_search")}
    context = {"round_budget_evidence": {"required": True, "energy_error": 1000, "coverage": 0.1}}
    assert round_budget_review_errors(action, context) == []
