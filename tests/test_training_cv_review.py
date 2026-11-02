import json
from phase_agent.tools.local.training_cv_review import prepare_cv_review
from tests.test_remote_training_recovery import returned_job


def test_missing_baseline_is_not_reported_as_improvement(tmp_path):
    job, root = returned_job(tmp_path)
    report, message = prepare_cv_review({}, job, {"fingerprint": "test"})
    assert report["structures"] == 0
    assert report["new"]["energy_per_atom"] is None
    assert report["activatable"] is False
    assert json.loads((root / "cv_baseline_review.json").read_text(encoding="utf-8")) == report
    assert "缺少匹配数据" in message


def test_recovery_preserves_review_only_for_identical_evidence(tmp_path):
    from phase_agent.tools.local.training_cv_review import register_cv_candidate
    from phase_agent.tools.local.recover_remote_training import inspect_training_results

    job, root = returned_job(tmp_path)
    result = inspect_training_results(job)
    report = {
        "structures": 1,
        "returned_structures": 1,
        "force_components": 1,
        "old": {
            "energy_per_atom": {"mae": 0.01, "rmse": 0.02},
            "forces": {"mae": 0.1, "rmse": 0.2},
        },
        "new": {
            "energy_per_atom": {"mae": 0.001, "rmse": 0.002},
            "forces": {"mae": 0.01, "rmse": 0.02},
        },
        "limitations": [],
    }
    (root / "cv_baseline_review.json").write_text("review")
    state = {"active_model_version": "base"}
    handoff = {}
    assert register_cv_candidate(state, "j", job, result, report, handoff)
    version = handoff["candidate_model_version"]
    assert state["candidate_models"][version]["evidence_status"] == "research_candidate"
    assert state["candidate_models"][version]["validation"]["independent_test_completed"] is False
    state["candidate_models"][version]["agent_review"] = {
        "choice": "activate",
        "reason": "improved",
    }
    assert register_cv_candidate(state, "j", job, result, report, {})
    assert state["candidate_models"][version]["agent_review"]["choice"] == "activate"
    changed = {**report, "limitations": ["new caveat"]}
    assert register_cv_candidate(state, "j", job, result, changed, {})
    assert "agent_review" not in state["candidate_models"][version]


def test_research_and_independent_evidence_are_distinct():
    from phase_agent.persistence.models.activate_validated_model import activate_validated_model

    state = {"active_model_version": "old"}
    model = {"version": "new"}
    research = {
        "status": "completed",
        "passed": True,
        "evidence_status": "research_candidate",
        "independent_test_completed": False,
        "evaluation_type": "grouped_cross_validation_review",
    }
    result = activate_validated_model(state, model, research, approval_reason="research refresh")
    assert result["status"] == "activated"
    assert (
        result["state"]["model_registry"]["new"]["validation"]["independent_test_completed"]
        is False
    )
    invalid = {**research, "evidence_status": "independently_validated"}
    assert (
        activate_validated_model(state, model, invalid, approval_reason="x")["reason"]
        == "independent_evidence_inconsistent"
    )
    independent = {**invalid, "independent_test_completed": True}
    assert (
        activate_validated_model(state, model, independent, approval_reason="x")["status"]
        == "activated"
    )
    assert activate_validated_model(state, model, research)["status"] == "awaiting_user_approval"
