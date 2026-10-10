"""JSON-portable labels from matching VASP ionic frames."""

import numpy as np
from copy import deepcopy


def _frame_labels(frame, index):
    structure = frame.get("structure")
    if structure is None:
        raise ValueError("ionic frame structure missing")
    energy = float(frame["e_0_energy"])
    forces = np.asarray(frame["forces"], dtype=float)
    if (
        not np.isfinite(energy)
        or forces.shape != (len(structure), 3)
        or not np.isfinite(forces).all()
    ):
        raise ValueError("invalid energy or atomic forces")
    labels = {
        "training_schema": "vasp-ionic-frame-v2",
        "structure": structure.as_dict(),
        "composition": structure.composition.get_el_amt_dict(),
        "atom_count": len(structure),
        "training_energy": energy,
        "training_energy_kind": "e_0_energy",
        "forces": forces.tolist(),
        "forces_unit": "eV/angstrom",
        "training_frame_index": index,
        "training_ready": True,
    }
    if frame.get("stress") is not None:
        stress = np.asarray(frame["stress"], dtype=float)
        if stress.shape != (3, 3) or not np.isfinite(stress).all():
            raise ValueError("invalid VASP stress tensor")
        # VASP kbar compression-positive -> ASE eV/A^3 tension-positive.
        labels.update(
            stress=(-stress * 0.0006241509074460763).tolist(),
            stress_unit="eV/angstrom^3",
            stress_convention="ASE_tension_positive",
        )
    return labels


def extract_training_frames(parsed):
    """Follow CHGNet's NELM filtering, keeping original frame indices."""
    accepted, rejected = [], []
    parameters = getattr(parsed, "parameters", {}) or {}
    nelm = parameters.get("NELM")
    for index, frame in enumerate(getattr(parsed, "ionic_steps", None) or []):
        try:
            electronic_steps = frame.get("electronic_steps")
            if nelm is not None and electronic_steps is not None:
                if not electronic_steps or len(electronic_steps) >= int(nelm):
                    raise ValueError("electronic convergence not demonstrated (NELM)")
                evidence = "electronic_steps_below_NELM"
            elif bool(getattr(parsed, "converged", False)):
                evidence = "converged_calculation"
            else:
                raise ValueError("electronic convergence information unavailable")
            labels = _frame_labels(frame, index)
            labels.update(electronic_converged=True, electronic_convergence_evidence=evidence)
            accepted.append(labels)
        except (KeyError, TypeError, ValueError) as error:
            rejected.append({"frame_index": index, "reason": str(error)})
    return {"frames": accepted, "rejected": rejected}


def final_training_labels(parsed):
    extracted = extract_training_frames(parsed)
    final_index = len(getattr(parsed, "ionic_steps", None) or []) - 1
    final = next(
        (frame for frame in extracted["frames"] if frame["training_frame_index"] == final_index),
        None,
    )
    if final is None:
        raise ValueError("final ionic frame has no verified training labels")
    return final


def training_records(record, outputs):
    """Convert portable frames into uniquely identified training examples."""
    frames = outputs.get("training_frames")
    if frames is None:
        frames = (
            [outputs]
            if outputs.get("training_ready") is True and record.get("converged") is True
            else []
        )
    records = []
    for frame in frames:
        if (
            frame.get("training_ready") is not True
            or frame.get("electronic_converged", True) is not True
        ):
            continue
        index = int(frame.get("training_frame_index", 0))
        records.append(
            {
                **deepcopy(record),
                **deepcopy(frame),
                "data_id": f"{record['task_id']}:frame:{index}",
                "source_task_status": record.get("status"),
                "status": "completed",
                "converged": True,
                "ionic_converged": outputs.get("ionic_converged", record.get("converged")),
                "energy": frame.get("training_energy", frame.get("energy")),
                "energy_unit": "eV",
            }
        )
    return records
