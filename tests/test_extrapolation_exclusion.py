from phase_agent.analysis.convergence.mlip_relaxation import (
    relaxation_failure, reconcile_relaxation_quality,
)


def test_force_triggered_predict_excludes_extrapolation_without_relabeling_convergence():
    task = {"task_id": "t", "status": "completed", "stage": "relax_and_feature",
            "converged": True, "parameters": {"model_refresh_operation": "predict"},
            "outputs": {"optimizer_converged": True, "single_point_completed": True,
                        "extrapolation_monitor": {"extrapolation_failure": True}}}
    assert relaxation_failure(task) == "extrapolation_failure"
    state = {"tasks": [task], "phase_records": [{"record_id": "r", "source_task_id": "t", "status": "completed"}],
             "phase_diagrams": {"mlip": {"entries": [{"record_id": "r"}]}}, "phase_diagrams_by_model": {}}
    result = reconcile_relaxation_quality(state)
    assert result["tasks"][0]["converged"] is True
    assert result["tasks"][0]["checks_passed"] is False
    assert result["phase_records"][0]["status"] == "excluded_extrapolation_failure"
    assert result["phase_diagrams"]["mlip"]["status"] == "quality_review_required"
    assert state["phase_records"][0]["status"] == "completed"


def test_no_extrapolation_keeps_valid_predict():
    task = {"status": "completed", "stage": "relax_and_feature",
            "parameters": {"model_refresh_operation": "predict"},
            "outputs": {"single_point_completed": True, "extrapolation_monitor": {"extrapolation_failure": False}}}
    assert relaxation_failure(task) is None


def test_unconverged_extrapolation_keeps_original_disposition():
    task = {"task_id": "t", "status": "completed", "stage": "relax_and_feature",
            "result_disposition": "discarded_unconverged", "converged": False,
            "extrapolation_review": {"status": "extrapolation_failure"}}
    result = reconcile_relaxation_quality({"tasks": [task]})
    assert result["tasks"][0]["result_disposition"] == "discarded_unconverged"
