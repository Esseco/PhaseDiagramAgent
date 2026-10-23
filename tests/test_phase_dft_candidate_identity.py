from copy import deepcopy

from phase_agent.analysis.phase.phase_dft_candidates import phase_review_candidates
from tests.test_phase_dft_candidates import files
from tests.test_phase_quality import sample


def test_qbc_requires_consistent_current_output_identity(tmp_path):
    state = sample()
    files(state, tmp_path)
    entry = state["phase_diagrams"]["mlip"]["entries"][1]
    base = {
        "task_id": "current",
        "branch_id": "b1",
        "model_version": state["active_model_version"],
        "status": "completed",
        "outputs": {
            "structure_path": entry["structure_path"],
            "structure_checksum": entry["structure_sha256"],
            "mlip_version": state["active_model_version"],
            "qbc": {"status": "completed", "energy_per_atom_std": 0.01},
        },
    }
    for field, wrong in [("structure_checksum", "different-geometry"), ("mlip_version", "old")]:
        task = deepcopy(base)
        task["outputs"][field] = wrong
        state["tasks"] = [task]
        row = next(
            x
            for x in phase_review_candidates(state, [])
            if x["candidate_id"] == entry["structure_id"]
        )
        assert row["qbc"]["status"].startswith("not_available")
    state["tasks"] = [base]
    row = next(
        x for x in phase_review_candidates(state, []) if x["candidate_id"] == entry["structure_id"]
    )
    assert row["qbc"] == base["outputs"]["qbc"]
    assert row["branch_id"] == "b1"
