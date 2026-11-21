"""Validate unlabeled MLIP frames without assigning a crystallographic phase."""

from math import isfinite
from pathlib import Path


def verified_unidentified_mlip(record, task):
    """Require bound completed task, final frame, model, energy and X evidence."""
    if not isinstance(task, dict):
        return False
    evidence = record.get("phase_identification") or {}
    output = task.get("outputs") or {}
    digest = record.get("structure_sha256")
    if (
        record.get("energy_method") != "mlip"
        or record.get("status") != "pending_phase_identification"
        or record.get("phase_identification_status") != "unknown"
        or evidence.get("status") != "unknown"
        or evidence.get("reason") != "ValueError: unrecognized_phase:X"
        or not evidence.get("classification_scope")
        or not isinstance(digest, str)
        or len(digest) != 64
        or evidence.get("structure_sha256") != digest
        or output.get("phase_identification") != evidence
        or output.get("structure_checksum") != digest
        or task.get("task_id") != record.get("source_task_id")
        or task.get("stage") not in {"relax_and_feature", "deep_search"}
        or record.get("stage", task.get("stage")) != task.get("stage")
        or task.get("status") != "completed"
        or task.get("checks_passed", True) is not True
        or record.get("checks_passed", True) is not True
        or task.get("converged") is not True
        or record.get("converged") is not True
        or not task.get("model_version")
        or task.get("model_version") != record.get("source_version")
        or record.get("model_version", task.get("model_version")) != task.get("model_version")
        or output.get("final_frame_valid") is False
        or record.get("final_frame_valid") is False
        or record.get("energy_unit") != "eV"
        or output.get("energy_unit") != "eV"
        or record.get("composition") != output.get("composition")
        or record.get("energy") != output.get("energy")
        or isinstance(record.get("energy"), bool)
        or not isinstance(record.get("energy"), (int, float))
        or not isfinite(record["energy"])
    ):
        return False
    from phase_agent.analysis.convergence.mlip_relaxation import relaxation_failure

    if relaxation_failure(task):
        return False
    try:
        from pymatgen.core import Composition
        from phase_agent.science.structures.load_result_structure import load_result_structure

        if output.get("structure") is None:
            source = Path(record["structure_path"])
            if source.resolve() != Path(output["structure_path"]).resolve():
                return False
        elif record.get("structure") != output["structure"]:
            return False
        structure, actual_digest = load_result_structure(output)
        return actual_digest == digest and structure.composition == Composition(
            record["composition"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def unidentified_hull_entry(row):
    """Recognize the explicit display label produced for a verified frame."""
    return (
        row.get("phase") == "Unidentified"
        and row.get("phase_identification_status") == "unknown"
        and row.get("unidentified_valid") is True
    )
