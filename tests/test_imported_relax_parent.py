from copy import deepcopy
import pytest
from phase_agent.tools.remote.batch_runner import _assign_upload_operations


def make_state():
    bundle = "relax-mc-1"
    path = (
        "C:/evidence/jobs/epoch0/calculations/G0001_Relax-0002/results/RELAX-1/initial_relaxed.vasp"
    )
    parent = {
        "task_id": "R",
        "stage": "relax_and_feature",
        "status": "completed",
        "relax_mc_bundle_id": bundle,
        "structure_id": "S",
        "branch_id": "B",
        "model_version": "M",
        "result_path": path,
    }
    task = {
        "task_id": "MC",
        "stage": "deep_search",
        "status": "pending",
        "relax_mc_bundle_id": bundle,
        "structure_id": "S",
        "branch_id": "B",
        "model_version": "M",
        "config_version": "C",
        "structure_path": path,
        "search_group_index": 1,
    }
    state = {
        "tasks": [parent, task],
        "slurm_batches": [
            {"batch_id": "old", "calculation_group": "Relax-screening", "operation_index": 1}
        ],
        "relax_mc_batches": {
            bundle: {
                "model_version": "M",
                "config_version": "C",
                "threshold_transition": {
                    "status": "relax_phase_imported",
                    "mc_first_wave": {"task_ids": ["MC"]},
                },
            }
        },
    }
    return state, task


def test_imported_relax_parent_binds_exact_frame_and_round():
    state, task = make_state()
    _assign_upload_operations(state, [task])
    assert task["parent_relax_round"] == 2


@pytest.mark.parametrize("change", ["path", "model", "branch", "scope", "bad_path", "duplicate"])
def test_imported_relax_parent_rejects_unverified_source(change):
    state, task = make_state()
    if change == "path":
        task["structure_path"] += "changed"
    if change == "model":
        task["model_version"] = "other"
    if change == "branch":
        task["branch_id"] = "other"
    if change == "scope":
        state["relax_mc_batches"]["relax-mc-1"]["threshold_transition"]["mc_first_wave"][
            "task_ids"
        ] = []
    if change == "bad_path":
        task["structure_path"] = "C:/unknown/initial_relaxed.vasp"
        state["tasks"][0]["result_path"] = task["structure_path"]
    if change == "duplicate":
        state["tasks"].append(deepcopy(state["tasks"][0]))
    with pytest.raises(ValueError, match="imported_relax_parent"):
        _assign_upload_operations(state, [task])
