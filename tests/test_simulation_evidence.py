from phase_agent.evidence.advisory_projection import project_advisory_payload
from phase_agent.evidence.snapshot import evidence_snapshot, simulation_facts


def test_native_simulation_defers_large_tasks_under_original_snapshot_binding():
    tasks = [{"task_id": str(i), "status": "completed", "outputs": {"forces": [0.1] * 100}} for i in range(150)]
    raw = {"state": {"tasks": tasks}, "decision_context": {}}
    snapshot = evidence_snapshot(project_advisory_payload(raw, {"sha256": "source"}))
    facts = simulation_facts({"tool": "prepare_local_batch_files", "parameters": {"mode": "relax_mc_results"}},
                             state=raw["state"], context=snapshot["facts"]["decision_context"], snapshot=snapshot)
    assert facts["snapshot_id"] == snapshot["snapshot_id"]
    assert facts["facts"]["state"]["tasks"]["count"] == 150
    assert "state.tasks" in facts["facts"]["decision_context"]["deferred_evidence"]["fields"]
    assert tasks[0]["outputs"]["forces"] == [0.1] * 100


def test_projected_evidence_retains_preexisting_deferred_refs():
    payload = {"state": {"tasks": [{}] * 129}, "decision_context": {"deferred_evidence": {"fields": ["state.old"]}}}
    result = project_advisory_payload(payload, {"sha256": "source"})
    assert result["decision_context"]["deferred_evidence"]["fields"] == ["state.old", "state.tasks"]


def test_large_batch_inventory_does_not_bloat_review_request():
    batches = {str(i): {"diagnostics": "x" * 3000} for i in range(100)}
    result = project_advisory_payload({"state": {"relax_mc_batches": batches}}, {"sha256": "s"})
    assert result["state"]["relax_mc_batches"]["count"] == 100
    assert result["state"]["relax_mc_batches"]["source_ref"] == "state.relax_mc_batches"
