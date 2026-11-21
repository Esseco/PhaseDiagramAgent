from copy import deepcopy

from phase_agent.analysis.phase.outside_refresh_hull import rejected_refresh_inputs


def bound():
    task = {
        "task_id": "t1",
        "task_checksum": "receipt",
        "status": "completed",
        "stage": "relax_and_feature",
        "checks_passed": False,
        "result_disposition": "excluded_extrapolation_failure",
        "model_version": "new",
        "structure_id": "s",
        "input_structure_sha256": "input",
        "parameters": {"model_refresh_operation": "relax"},
        "outputs": {"extrapolation_monitor": {"extrapolation_failure": True}},
    }
    record = {
        "source_task_id": "t1",
        "checks_passed": False,
        "status": "excluded_extrapolation_failure",
        "model_version": "new",
        "structure_id": "s",
    }
    return task, record


def test_rejected_refresh_blocks_only_bound_newer_input():
    task, record = bound()
    assert rejected_refresh_inputs([record], [task], {"new": 2}) == {("s", "input"): 2}
    assert rejected_refresh_inputs([], [task], {"new": 2}) == {("s", "input"): 2}
    for broken in ("model_version", "structure_id", "checks_passed", "status"):
        changed = deepcopy(record)
        changed[broken] = None
        assert rejected_refresh_inputs([changed], [task], {"new": 2}) == {}
    assert rejected_refresh_inputs([record], [task, task], {"new": 2}) == {}
    assert rejected_refresh_inputs([record, record], [task], {"new": 2}) == {}
    task["outputs"]["extrapolation_monitor"]["extrapolation_failure"] = False
    assert rejected_refresh_inputs([record], [task], {"new": 2}) == {}


def test_unknown_phase_requires_explicit_audited_discard():
    task, record = bound()
    task["result_disposition"] = "discarded_unknown_phase"
    task["outputs"]["phase_identification"] = {
        "status": "unknown",
        "reason": "ValueError: unrecognized_phase:X",
    }
    record["status"] = "excluded_unknown_phase"
    assert rejected_refresh_inputs([record], [task], {"new": 2}) == {}
    task["phase_evidence_recovery"] = {"decision": "discard_unknown"}
    assert rejected_refresh_inputs([record], [task], {"new": 2}) == {("s", "input"): 2}


def test_rejected_newer_result_cannot_revive_old_input_in_hull():
    from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram

    task, rejected = bound()

    def phase_record(rid, sid, sha, model, na, energy):
        return {
            "record_id": rid,
            "structure_id": sid,
            "structure_sha256": sha,
            "model_version": model,
            "source_version": model,
            "energy_method": "mlip",
            "energy_unit": "eV",
            "energy": energy,
            "status": "completed",
            "composition": {"Na": na, "Co": 1, "O": 2},
            "phase": "O3",
        }

    records = [
        phase_record("zero", "z", "zero", "new", 0, -5),
        phase_record("one", "o", "one", "new", 1, -7),
        phase_record("old", "s", "input", "old", 0.5, -10),
        {**phase_record("bad", "s", "newframe", "new", 0.5, -11), **rejected},
    ]
    state = {"tasks": [task], "upload_layout": {"model_rounds": {"old": 1, "new": 2}}}
    diagram = update_phase_diagram(
        records, active_model_version="new", output_state=state, latest_per_structure=True
    )["diagrams"]["mlip"]
    assert diagram["status"] == "completed"
    assert {r["record_id"] for r in diagram["entries"]} == {"zero", "one"}
