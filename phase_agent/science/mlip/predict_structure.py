"""Same-geometry committee prediction with an explicit main model."""

from collections import Counter
from pathlib import Path


def predict_structure(job):
    import numpy as np
    from ase.io import read
    from phase_agent.tools.remote.integrity import file_checksum
    from phase_agent.science.qbc.summarize_member_energies import summarize_member_energies

    structure_path = Path(job["structure_path"])
    atoms = read(
        structure_path, format="vasp" if structure_path.suffix.lower() == ".vasp" else None
    )
    parameters = job.get("parameters") or {}
    paths = job.get("model_paths") or [job.get("model_path")]
    from phase_agent.science.mlip.model_interface import create_calculator

    index = int(parameters.get("main_model_index", 0))
    if not paths or not all(paths) or not 0 <= index < len(paths):
        raise ValueError("prediction requires valid model paths and main model index")
    energies, forces = [], []
    for path in paths:
        atoms.calc = create_calculator(
            {
                **parameters,
                "framework": parameters.get("framework", "mace"),
                "local_repo_path": parameters.get("local_repo_path"),
                "model_name": parameters.get("model_name") or parameters.get("mlip_version"),
            },
            model_path=path,
        )
        energies.append(float(atoms.get_potential_energy()))
        forces.append(atoms.get_forces().copy())
    if not np.isfinite(energies).all() or not np.isfinite(forces).all():
        raise ValueError("nonfinite committee prediction")
    qbc = (
        summarize_member_energies([e / len(atoms) for e in energies], expected_members=len(paths))
        if len(paths) > 1
        else {"status": "not_available_single_model", "member_count": 1}
    )
    if len(paths) > 1:
        qbc["force_rms_disagreement"] = float(np.sqrt(np.mean(np.var(forces, axis=0))))
    if parameters.get("model_refresh_operation"):
        import shutil

        structure_path = Path(job["output_directory"]) / "evaluated.vasp"
        if Path(job["structure_path"]).resolve() != structure_path.resolve():
            shutil.copy2(job["structure_path"], structure_path)
    return {
        "status": "completed",
        "energy": energies[index],
        "forces": forces[index].tolist(),
        "energy_unit": "eV",
        "forces_unit": "eV/angstrom",
        "single_point_completed": True,
        "structure_path": str(structure_path.resolve()),
        "structure_checksum": file_checksum(structure_path),
        "composition": dict(Counter(atoms.get_chemical_symbols())),
        "atom_count": len(atoms),
        "mlip_version": job.get("model_version"),
        "mlip_name": parameters.get("framework", "mace").upper(),
        "qbc": qbc,
        "geometry": "model_refresh"
        if parameters.get("model_refresh_operation")
        else "DFT_final_frame",
    }
