"""Same-geometry committee prediction with an explicit main model."""
from collections import Counter
from pathlib import Path


def predict_structure(job):
    import numpy as np
    from ase.io import read
    from mace.calculators import MACECalculator
    from execution_layer.remote.integrity import file_checksum
    from scientific_layer.qbc.summarize_member_energies import summarize_member_energies
    atoms = read(job["structure_path"])
    parameters = job.get("parameters") or {}
    paths = job.get("model_paths") or [job.get("model_path")]
    index = int(parameters.get("main_model_index", 0))
    if not paths or not all(paths) or not 0 <= index < len(paths):
        raise ValueError("prediction requires valid model paths and main model index")
    energies, forces = [], []
    for path in paths:
        options = {"model_paths": str(path), "device": parameters.get("device", "cpu"),
                   "default_dtype": parameters.get("mace_default_dtype", "float64")}
        if parameters.get("mace_head"):
            options["head"] = parameters["mace_head"]
        atoms.calc = MACECalculator(**options)
        energies.append(float(atoms.get_potential_energy()))
        forces.append(atoms.get_forces().copy())
    if not np.isfinite(energies).all() or not np.isfinite(forces).all():
        raise ValueError("nonfinite committee prediction")
    qbc = summarize_member_energies([e/len(atoms) for e in energies], expected_members=len(paths)) if len(paths) > 1 else {
        "status": "not_available_single_model", "member_count": 1}
    if len(paths) > 1:
        qbc["force_rms_disagreement"] = float(np.sqrt(np.mean(np.var(forces, axis=0))))
    structure_path = Path(job["structure_path"])
    if parameters.get("model_refresh_operation"):
        import shutil
        structure_path = Path(job["output_directory"]) / "evaluated.vasp"
        if Path(job["structure_path"]).resolve() != structure_path.resolve():
            shutil.copy2(job["structure_path"], structure_path)
    return {"status": "completed", "energy": energies[index], "forces": forces[index].tolist(),
            "energy_unit": "eV", "forces_unit": "eV/angstrom", "single_point_completed": True,
            "structure_path": str(structure_path.resolve()),
            "structure_checksum": file_checksum(structure_path),
            "composition": dict(Counter(atoms.get_chemical_symbols())), "atom_count": len(atoms),
            "mlip_version": job.get("model_version"), "mlip_name": "MACE", "qbc": qbc,
            "geometry": "model_refresh" if parameters.get("model_refresh_operation") else "DFT_final_frame"}
