"""Bridge the shared Py-Code Fe/Mn diagnostic to portable DFT JSON labels."""
from copy import deepcopy
import hashlib
import json

from scientific_layer.dft.layered_spin_scope import LAYERED_PHASES, layered_spin_scope


def extract_magnetic_diagnostics(directory, parsed):
    """Collect final OUTCAR evidence; do not invalidate otherwise valid DFT labels."""
    structure = getattr(parsed, "final_structure", None)
    if structure is None:
        steps = getattr(parsed, "ionic_steps", []) or []
        structure = steps[-1].get("structure") if steps else None
    if structure is None:
        return {"status": "unknown", "reason": "final_structure_unavailable"}
    elements = [site.specie.symbol for site in structure]
    try:
        from Process_Vasp import read_dft_magnetic_data
        report = read_dft_magnetic_data(directory, structure=structure,
                                       parameters=getattr(parsed, "parameters", {}) or {})
        report["final_frame_index"] = len(getattr(parsed, "ionic_steps", []) or []) - 1
        return report
    except (ImportError, OSError, ValueError, TypeError) as error:
        return {"status": "unknown", "reason": "magnetic_reader_unavailable",
                "error": f"{type(error).__name__}: {error}", "elements": elements}


def diagnose_result_magnetism(result, manager=None):
    """Use saved evidence and known system/phase; no OUTCAR reads on local recovery."""
    current = deepcopy(result)
    if current.get("stage") not in {"dft_single_point", "dft_relax"}:
        return current
    outputs = current.setdefault("outputs", {})
    prior = outputs.get("magnetic_check") or {}
    raw = outputs.get("magnetic_moments") or prior  # Legacy diagnosed JSON is still readable.
    structure = outputs.get("structure")
    if not structure:
        # Preserve remote diagnostics even when a failed run lacks final labels.
        return current
    from scientific_layer.structures.load_result_structure import load_result_structure
    try:
        parsed, digest = load_result_structure(outputs)
        elements = [site.specie.symbol for site in parsed]
        layered = layered_spin_scope(outputs, manager)
        if raw.get("elements") and raw["elements"] != elements:
            raise ValueError("magnetic site order differs from final structure")
        if (raw.get("final_frame_index") is not None and outputs.get("final_frame_index") is not None
                and raw["final_frame_index"] != outputs["final_frame_index"]):
            raise ValueError("magnetic evidence differs from actual final frame")
        values = raw.get("moments")
        if values is not None and not raw.get("noncollinear", False):
            values = [float(value) for value in values]
        evidence = {"structure_sha256": digest, "elements": elements, "moments": values,
                    "noncollinear": raw.get("noncollinear", False), "layered": layered,
                    "total": raw.get("total_local_moment"), "final_frame_index": outputs.get("final_frame_index")}
        key = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        if prior.get("evidence_sha256") == key and "reasonable_atoms" in prior:
            return current
        from Process_Vasp import check_layered_oxide_moments
        report = check_layered_oxide_moments(elements, values, is_layered_oxide=layered,
            noncollinear=raw.get("noncollinear", False), total_moment=raw.get("total_local_moment"))
        # Original read errors/source remain visible after local classification.
        report.update({k: raw[k] for k in ("source", "nupdown", "read_error", "error", "final_frame_index") if k in raw})
        report["evidence_sha256"] = key
        outputs["magnetic_check"] = report
    except (ImportError, OSError, ValueError, TypeError, KeyError) as error:
        outputs["magnetic_check"] = {**prior, "status": "unknown", "reason": "magnetic_diagnostic_unavailable",
                                     "error": f"{type(error).__name__}: {error}"}
    return current
