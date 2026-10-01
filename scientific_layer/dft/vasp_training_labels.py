"""JSON-portable labels from one matching VASP ionic frame."""

import numpy as np


def final_training_labels(parsed):
    steps = getattr(parsed, "ionic_steps", None) or []
    if not steps:
        raise ValueError("no ionic frame for training")
    frame = steps[-1]
    structure = frame.get("structure")
    if structure is None:
        raise ValueError("ionic frame structure missing")
    energy = float(frame["e_0_energy"])
    forces = np.asarray(frame["forces"], dtype=float)
    if not np.isfinite(energy) or forces.shape != (len(structure), 3) or not np.isfinite(forces).all():
        raise ValueError("invalid energy or atomic forces")
    labels = {
        "training_schema": "vasp-final-frame-v1",
        "structure": structure.as_dict(),
        "composition": structure.composition.get_el_amt_dict(),
        "atom_count": len(structure),
        "training_energy": energy,
        "training_energy_kind": "e_0_energy",
        "forces": forces.tolist(), "forces_unit": "eV/angstrom",
        "training_frame_index": len(steps) - 1,
        "training_ready": True,
    }
    if frame.get("stress") is not None:
        stress = np.asarray(frame["stress"], dtype=float)
        if stress.shape != (3, 3) or not np.isfinite(stress).all():
            raise ValueError("invalid VASP stress tensor")
        # VASP kbar compression-positive -> ASE eV/A^3 tension-positive.
        labels.update(stress=(-stress * 0.0006241509074460763).tolist(),
                      stress_unit="eV/angstrom^3", stress_convention="ASE_tension_positive")
    return labels
