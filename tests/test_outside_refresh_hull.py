import copy

import pytest

from phase_agent.analysis.phase.outside_refresh_hull import outside_refresh_inputs
from phase_agent.analysis.phase.update_phase_diagram import update_phase_diagram
from tests.test_latest_mixed_hull import record


def sample():
    evidence = {"status": "outside_boundary", "observed_phase": "OP3",
                "classification_scope": "scope", "structure_sha256": "final"}
    task = {"task_id": "t", "status": "completed", "stage": "relax_and_feature",
            "structure_id": "a", "input_structure_sha256": "aaa", "model_version": "new",
            "converged": True, "phase_identification_status": "outside_boundary",
            "parameters": {"model_refresh_operation": "predict"},
            "outputs": {"phase_identification": evidence, "optimizer_converged": True,
                        "extrapolation_monitor": {"extrapolation_failure": False}}}
    outside = {**record("outside", "a", "final", "new", 0.5, -5),
               "source_task_id": "t", "status": "outside_search_boundary",
               "phase_identification": copy.deepcopy(evidence)}
    old = record("old", "a", "aaa", "old", 0.5, -5)
    other = record("other", "a", "different", "old", 0.5, -4)
    endpoints = [record("lo", "lo", "lohash", "new", 0, -5),
                 record("hi", "hi", "hihash", "new", 1, -7)]
    return task, outside, old, other, endpoints


def test_verified_outside_only_replaces_its_own_older_frame():
    task, outside, old, other, endpoints = sample()
    state = {"tasks": [task], "upload_layout": {"model_rounds": {"old": 1, "new": 2}}}
    records = [old, other, outside, *endpoints, record("lo_old", "lo", "lohash", "old", 0, -4), record("hi_old", "hi", "hihash", "old", 1, -6)]
    # Different hashes are deliberately geometrically distinct; no actual files required.
    import phase_agent.analysis.phase.latest_mlip_lineage as lineage
    original = lineage.structures_equivalent
    try:
        lineage.structures_equivalent = lambda a, b, caches=None: a["structure_sha256"] == b["structure_sha256"]
        out = update_phase_diagram(records, active_model_version="new", output_state=state,
                                   latest_per_structure=True)
        assert {r["record_id"] for r in out["diagrams"]["mlip"]["entries"]} == {"other", "lo", "hi"}
        assert {r["record_id"] for r in out["mlip_by_version"]["old"]["entries"]} == {"old", "other", "lo_old", "hi_old"}
        fast = update_phase_diagram(records, active_model_version="new", output_state=state,
                                    latest_per_structure=True, latest_only=True)
        assert fast["diagrams"]["mlip"]["version"] == out["diagrams"]["mlip"]["version"]
    finally:
        lineage.structures_equivalent = original


@pytest.mark.parametrize("case", ["failed", "no_monitor", "bad_hash", "wrong_model",
                                   "wrong_structure", "missing_scope", "duplicate", "unconverged"])
def test_unverified_outside_does_not_discard_prior_frame(case):
    task, outside, old, other, endpoints = sample()
    if case == "failed":
        task["outputs"]["extrapolation_monitor"]["extrapolation_failure"] = True
    elif case == "no_monitor":
        task["outputs"].pop("extrapolation_monitor")
    elif case == "bad_hash":
        outside["structure_sha256"] = "other"
    elif case == "wrong_model":
        outside["model_version"] = "old"
    elif case == "wrong_structure":
        outside["structure_id"] = "other"
    elif case == "missing_scope":
        task["outputs"]["phase_identification"]["classification_scope"] = ""
        outside["phase_identification"]["classification_scope"] = ""
    elif case == "unconverged":
        task["converged"] = False
    tasks = [task, copy.deepcopy(task)] if case == "duplicate" else [task]
    assert outside_refresh_inputs([outside], tasks, {"old": 1, "new": 2}) == {}
