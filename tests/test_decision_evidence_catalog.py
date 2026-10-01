from analysis_layer.state.decision_evidence_catalog import decision_evidence_catalog, check_decision_evidence_refs


def test_only_existing_versions_and_approved_context_are_indexed():
    catalog = decision_evidence_catalog({"current_phase_diagram": {
        "mlip": {"version": "v1", "status": "completed"}, "dft": {}},
        "relevant_approved_knowledge": [{"knowledge_id": "k1"}]})
    assert set(catalog["references"]) == {"phase_diagram:mlip:v1", "knowledge:k1"}
    assert check_decision_evidence_refs(["phase_diagram:mlip:v0"], catalog)["missing"] == ["phase_diagram:mlip:v0"]


def test_empty_or_invalid_refs_are_not_validated_scientific_proofs():
    assert check_decision_evidence_refs([], {})["status"] == "no_references"
    assert check_decision_evidence_refs("a", {})["status"] == "invalid_reference_format"


def test_task_lineage_reference_is_stable_but_status_stays_current():
    lineage = {"model_version": "m1", "segment_index": 0}
    first = decision_evidence_catalog({"task_round_evidence": {"groups": [
        {"lineage": lineage, "task_count": 1, "stage_evidence": [{"status": "running"}]}]}})
    second = decision_evidence_catalog({"task_round_evidence": {"groups": [
        {"lineage": lineage, "task_count": 1, "stage_evidence": [{"status": "completed"}]}]}})
    assert set(first["references"]) == set(second["references"])
    assert first["references"] != second["references"]


def test_cost_reference_is_not_presented_as_measured_sample():
    row = decision_evidence_catalog({"calculation_cost_reference": {"reports": [
        {"stage": "deep_search", "status": "fallback"}]}})["references"]["cost_reference:deep_search"]
    assert row["kind"] == "reference_cost_estimate"
    assert "not an individual measured sample" in row["instruction"]
