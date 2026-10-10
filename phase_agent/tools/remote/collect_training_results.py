"""Portable HPC finalizer: evaluate out-of-fold data and list trained models."""

import csv
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np


def metrics(errors):
    values = np.asarray(errors, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("No finite evaluation values")
    return float(np.abs(values).mean()), float(np.sqrt(np.square(values).mean()))


def model_file(directory, name):
    for suffix in (".model", "_compiled.model"):
        for stem in (name + "_ema", name):
            matches = sorted(directory.rglob(stem + suffix))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous trained model: {name}")
    raise FileNotFoundError(f"Trained model missing: {name}")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_model_manifest(root, plan, output):
    """Record HPC identities without copying potentially gigabyte-sized models."""
    main_model = plan["main_model"]["final_model"]
    members = {row["model_id"] for row in plan["committees"]}
    names = list(dict.fromkeys([row["model_id"] for row in plan["committees"]] + [main_model]))
    manifest = []
    for name in names:
        source = model_file(root / name, name).resolve()
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        manifest.append(
            {
                "model_id": name,
                "model_path": str(source),
                "remote_model_path": str(source),
                "storage": "remote",
                "size_bytes": source.stat().st_size,
                "sha256": digest.hexdigest(),
                "is_main_model": name == main_model,
                "is_committee_member": name in members,
                "source_model_version": plan.get("original_model_version"),
                "epoch": plan.get("epoch"),
                "model_version": plan.get("original_model_version"),
                "training_round": plan.get("training_round", root.name),
            }
        )
    output.mkdir(exist_ok=True)
    (output / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    from ase.io import read
    from mace.calculators import MACECalculator

    root = Path(__file__).resolve().parent
    plan = json.loads((root / "training_plan.json").read_text(encoding="utf-8"))
    output = (root.parent if root.name == "inputs" else root) / "results"
    output.mkdir(exist_ok=True)
    energies, forces, summaries, manifest = [], [], [], []
    for job in plan["main_model"]["fold_jobs"]:
        name = job["model_id"]
        model = model_file(root / name, name)
        calculator = MACECalculator(model_paths=str(model), device="cuda", default_dtype="float64")
        e_errors, f_errors = [], []
        for index, atoms in enumerate(read(root / name / "valid.xyz", index=":")):
            ref_energy = float(
                atoms.info[plan["config"]["labels"].get("energy_key", "REF_energy")]
            ) / len(atoms)
            ref_forces = atoms.arrays[
                plan["config"]["labels"].get("forces_key", "REF_forces")
            ].copy()
            atoms.calc = calculator
            predicted_energy = float(atoms.get_potential_energy()) / len(atoms)
            predicted_forces = atoms.get_forces()
            e_errors.append(predicted_energy - ref_energy)
            record_identity = {
                "data_id": atoms.info.get("data_id", ""),
                "task_id": atoms.info.get("task_id", ""),
                "is_current_round": bool(atoms.info.get("is_current_round", False)),
            }
            energies.append(
                {
                    "fold": name,
                    "structure_index": index,
                    "source": atoms.info.get("source", ""),
                    **record_identity,
                    "E_DFT_eV_per_atom": ref_energy,
                    "E_MLIP_eV_per_atom": predicted_energy,
                }
            )
            for atom in range(len(atoms)):
                for axis in range(3):
                    error = float(predicted_forces[atom, axis] - ref_forces[atom, axis])
                    f_errors.append(error)
                    forces.append(
                        {
                            "fold": name,
                            "structure_index": index,
                            "atom_index": atom,
                            **record_identity,
                            "component": "xyz"[axis],
                            "F_DFT_eV_per_A": float(ref_forces[atom, axis]),
                            "F_MLIP_eV_per_A": float(predicted_forces[atom, axis]),
                        }
                    )
        emae, ermse = metrics(e_errors)
        fmae, frmse = metrics(f_errors)
        summaries.append(
            {
                "fold": name,
                "structures": len(e_errors),
                "force_components": len(f_errors),
                "evaluation_type": "grouped_cross_validation",
                "energy_MAE_meV_per_atom": emae * 1000,
                "energy_RMSE_meV_per_atom": ermse * 1000,
                "force_MAE_meV_per_A": fmae * 1000,
                "force_RMSE_meV_per_A": frmse * 1000,
            }
        )
    emae, ermse = metrics([r["E_MLIP_eV_per_atom"] - r["E_DFT_eV_per_atom"] for r in energies])
    fmae, frmse = metrics([r["F_MLIP_eV_per_A"] - r["F_DFT_eV_per_A"] for r in forces])
    summaries.append(
        {
            "fold": "all_out_of_fold",
            "structures": len(energies),
            "force_components": len(forces),
            "evaluation_type": "grouped_cross_validation",
            "energy_MAE_meV_per_atom": emae * 1000,
            "energy_RMSE_meV_per_atom": ermse * 1000,
            "force_MAE_meV_per_A": fmae * 1000,
            "force_RMSE_meV_per_A": frmse * 1000,
        }
    )
    latest_e = [r for r in energies if r["is_current_round"]]
    latest_f = [r for r in forces if r["is_current_round"]]
    if latest_e and latest_f:
        emae, ermse = metrics([r["E_MLIP_eV_per_atom"] - r["E_DFT_eV_per_atom"] for r in latest_e])
        fmae, frmse = metrics([r["F_MLIP_eV_per_A"] - r["F_DFT_eV_per_A"] for r in latest_f])
        summaries.append(
            {
                "fold": "current_round_out_of_fold",
                "structures": len(latest_e),
                "force_components": len(latest_f),
                "evaluation_type": "grouped_cross_validation",
                "energy_MAE_meV_per_atom": emae * 1000,
                "energy_RMSE_meV_per_atom": ermse * 1000,
                "force_MAE_meV_per_A": fmae * 1000,
                "force_RMSE_meV_per_A": frmse * 1000,
            }
        )
    manifest = write_model_manifest(root, plan, output)
    identity = {
        "epoch": plan.get("epoch"),
        "model_version": plan.get("original_model_version"),
        "training_round": plan.get("training_round", root.name),
    }
    for rows in (summaries, energies, forces, manifest):
        for row in rows:
            row.update(identity)
    write_csv(output / "kfold_metrics.csv", summaries)
    write_csv(output / "energy_comparison.csv", energies)
    write_csv(output / "force_comparison.csv", forces)
    (output / "models.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "training.finished.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "original_model_version": plan.get("original_model_version"),
                "folds": len(plan["main_model"]["fold_jobs"]),
                "out_of_fold_structures": len(energies),
                "committee_count": len(plan["committees"]),
                "model_storage": "remote",
                "model_manifest": "models.json",
                "activated": False,
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect lightweight training results; models remain on HPC"
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Update model paths/hashes only; do not evaluate or train",
    )
    args = parser.parse_args()
    if args.manifest_only:
        root = Path(__file__).resolve().parent
        plan = json.loads((root / "training_plan.json").read_text(encoding="utf-8"))
        write_model_manifest(
            root, plan, (root.parent if root.name == "inputs" else root) / "results"
        )
    else:
        main()
