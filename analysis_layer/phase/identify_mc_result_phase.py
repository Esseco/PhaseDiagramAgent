"""Re-identify only phase P after MC; retain the source branch unchanged."""
from copy import deepcopy
from pathlib import Path


def identify_mc_result_phase(result, manager):
    current = deepcopy(result)
    if current.get("stage") != "deep_search" or current.get("status") != "completed":
        return current
    branch_id = current.get("branch_id")
    if not branch_id:
        structure = manager.data.get("structures", {}).get(current.get("structure_id")) or {}
        branch_id = structure.get("branch_id")
    original = (manager.data.get("branches", {}).get(branch_id) or {}).get("P")
    outputs = current.get("outputs") or {}
    path = outputs.get("structure_path")
    actual = None
    reason = None
    if path and Path(path).is_file():
        try:
            from pymatgen.core import Structure
            from scientific_layer.structures.identify_branch import identify_phase
            detected = identify_phase(Structure.from_file(path))
            phase = detected.get("phase")
            if phase in {"O1", "O3", "P3", "OP2"}:
                actual = phase
            else:
                reason = f"unrecognized_phase:{phase}"
        except Exception as error:
            reason = f"identification_failed:{type(error).__name__}"
    else:
        reason = "final_structure_missing"
    current["source_phase"] = original
    current["actual_phase"] = actual
    current["phase_identification_status"] = "identified" if actual else "unknown"
    if reason:
        current["phase_identification_reason"] = reason
    current["outputs"] = {**outputs, "source_phase": original, "actual_phase": actual,
                          "phase_identification": {"status": current["phase_identification_status"],
                                                   "reason": reason}}
    return current
