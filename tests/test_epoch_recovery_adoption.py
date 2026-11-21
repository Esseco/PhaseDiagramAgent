from copy import deepcopy
import pytest
from phase_agent.runtime.epoch_recovery_adoption import adopt_state
from phase_agent.agents.reliability import extrapolation_round as rnd
from phase_agent.evidence import dft_failure_policy as failure
from phase_agent.evidence import extrapolation_recovery as recovery
from phase_agent.evidence import parent_first_recovery as parent
from phase_agent.evidence import model_epoch


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(parent, "valid_parent_first_cycle", lambda state: True)
    monkeypatch.setattr(failure, "unreviewed_failure_batches", lambda state: [])
    monkeypatch.setattr(rnd, "assess_current_rounds", lambda state: {"rounds":[
        {"scope_id":"epoch_refresh", "evidence_digest":"abc", "direct_sampling_required":True}]})
    monkeypatch.setattr(rnd, "recovery_dft_failure_settled", lambda state, cycle: True)
    monkeypatch.setattr(model_epoch, "model_epoch", lambda state, model: "epoch1")
    monkeypatch.setattr(recovery, "recovery_label_facts", lambda state: {
        "status":"waiting", "all_returned":True, "threshold_met":False,
        "selected_count":100, "labelled_count":41, "failed_count":69})
    return {"active_model_version":"m", "extrapolation_recovery_cycle":{
        "scope_id":"epoch_refresh","status":"awaiting_reviewed_dft_labels",
        "invalid_model_version":"m","batch_id":"b"},
        "parent_first_recovery_batches":{"b":{"status":"waiting_dft_results"}},
        "tasks":[{"status":"failed"},{"status":"completed"}]}


def test_adoption_is_bound_idempotent_and_preserves_results(case):
    original = deepcopy(case)
    updated, receipt = adopt_state(case, "abc")
    assert case == original
    assert updated["tasks"] == original["tasks"]
    assert receipt["progress"]["labelled_count"] == 41
    assert updated["parent_first_recovery_batches"]["b"]["status"] == "results_received"
    assert updated["extrapolation_recovery_cycle"]["legacy_policy_adoption"]["decision"] == "exclude_failed_without_retry"
    assert adopt_state(updated, "abc")[1]["status"] == "already_adopted"


def test_refuses_wrong_digest_and_unreviewed_failures(case, monkeypatch):
    with pytest.raises(ValueError, match="round_mismatch"):
        adopt_state(case, "wrong")
    monkeypatch.setattr(failure, "unreviewed_failure_batches", lambda state: [{}])
    with pytest.raises(ValueError, match="failure_analysis_required"):
        adopt_state(case, "abc")


def test_verified_direct_cycle_adopts_only_settled_50_labels(case, monkeypatch):
    from phase_agent.tools.workflows import validate_dft_agent_decisions as validation

    case["extrapolation_recovery_cycle"].update(recovery_route="extrapolation", selected=[{"candidate_id": "c"}])
    case["direct_sampling_results"] = {"epoch_refresh": {"round_evidence_digest": "abc"}}
    monkeypatch.setattr(validation, "_is_direct_recovery_selection", lambda state, ids: ids == {"c"})
    monkeypatch.setattr(recovery, "recovery_label_facts", lambda state: {
        "status": "complete", "all_returned": True, "threshold_met": True,
        "selected_count": 100, "labelled_count": 60, "failed_count": 40})
    updated, receipt = adopt_state(case, "abc")
    assert receipt["progress"]["labelled_count"] == 60
    assert updated["extrapolation_recovery_cycle"]["policy_version"] == "epoch_refresh_v1"
    assert updated["tasks"] == case["tasks"]
    case["direct_sampling_results"]["epoch_refresh"]["round_evidence_digest"] = "other"
    with pytest.raises(ValueError, match="legacy_cycle_invalid"):
        adopt_state(case, "abc")
    case["direct_sampling_results"]["epoch_refresh"]["round_evidence_digest"] = "abc"
    monkeypatch.setattr(validation, "_is_direct_recovery_selection", lambda state, ids: False)
    with pytest.raises(ValueError, match="legacy_cycle_invalid"):
        adopt_state(case, "abc")
