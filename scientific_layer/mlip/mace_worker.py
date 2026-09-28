"""在 py-mace 环境中执行 MACE Relax 或 Process_AL_MC。"""
from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path


def _read_json(path: Path, default=None):
    compressed = Path(f"{path}.gz")
    if compressed.is_file():
        with gzip.open(compressed, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_json_gzip(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(f"{path}.gz"), "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def _final_member_energies(output: Path, row: dict) -> list[float] | None:
    values = row.get("energy_per_model_per_atom")
    if values:
        return [float(value) for value in values]
    name = row.get("relax_traj_file")
    frames = _read_json(output / "pool" / name, []) if name else []
    for frame in reversed(frames or []):
        values = frame.get("energy_per_model_per_atom")
        if values:
            return [float(value) for value in values]
    return None


def _remove_relax_trajectories(output: Path, summary: list[dict]) -> None:
    names = {row.get("relax_traj_file") for row in summary} | {"initial_relax_traj.json"}
    for name in names:
        if not name:
            continue
        for base in (output / "pool" / name, output / name):
            for path in (base, Path(f"{base}.gz")):
                if path.is_file():
                    path.unlink()


def _actual_mc_steps(output: Path) -> int | None:
    trace = _read_json(output / "trace.json", []) or []
    steps = [row.get("step") for row in trace if isinstance(row.get("step"), int)]
    return max(steps) if steps else None


def _qbc_from_summary(best: dict, member_count: int) -> dict:
    if member_count < 2:
        return {"status": "not_available_single_model", "member_count": member_count,
                "interpretation": "single_model_has_no_committee_uncertainty"}
    if best.get("energy_per_model_per_atom"):
        from scientific_layer.qbc.summarize_member_energies import summarize_member_energies
        return summarize_member_energies(best["energy_per_model_per_atom"],
                                         expected_members=member_count)
    return {
        "status": "completed", "member_count": member_count,
        "energy_per_atom_std": best.get("energy_std_per_atom"),
        "energy_per_atom_var": best.get("energy_var_per_atom"),
        "force_rms_disagreement": best.get("force_uncertainty_per_atom"),
        "f_std_max": best.get("force_uncertainty_max"),
        "interpretation": "committee_disagreement_not_true_error",
    }


def run_mace_worker(job: dict) -> dict:
    output = Path(job["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(job.get("parameters") or {})
    operation = job["operation"]
    if operation == "relax":
        model_paths = [str(path) for path in (job.get("model_paths") or [])]
        if model_paths:
            main_index = int(parameters.pop("main_model_index", 0))
            if not 0 <= main_index < len(model_paths):
                raise ValueError("main_model_index out of range")
            model_path = model_paths[main_index]
        else:
            model_path = job.get("model_path")
        if not model_path:
            raise ValueError("Relax task requires model_path or model_paths")
        from Process_AL_MC.relax import relax_structure_mace
        mace_head = parameters.pop("mace_head", None)
        if mace_head is None:
            identity = " ".join(str(value or "") for value in
                                 (job.get("model_version"), model_path)).lower()
            if "mace-mh-1" in identity or "mh-1" in identity or "mh_1" in identity:
                mace_head = "omat_pbe"
        structure_path = output / "initial_relaxed.vasp"
        result = relax_structure_mace(
            job["structure_path"], model_path, output_path=structure_path,
            device=parameters.pop("device", "cuda"), head=mace_head,
            default_dtype=parameters.pop("mace_default_dtype", "float64"),
            fmax=parameters.pop("fmax", 0.05),
            steps=parameters.pop("relax_steps", 150),
            relax_cell=parameters.pop("relax_cell", True),
        )
        from pymatgen.core import Structure
        final = Structure.from_file(structure_path)
        from execution_layer.remote.integrity import file_checksum
        result.update({
            "status": "completed", "structure_path": str(structure_path),
            "structure_checksum": file_checksum(structure_path),
            "composition": final.composition.as_dict(),
            "atom_count": len(final), "mlip_name": "MACE",
            "mlip_version": job.get("model_version"),
            "qbc": {"status": "not_available_single_model", "member_count": 1},
        })
        return result

    if operation != "mc":
        raise ValueError(f"unsupported MLIP operation: {operation}")

    from Process_AL_MC import LayeredOxide_MCOrderingClass

    parameters.setdefault("save_relax_traj", False)
    mode = parameters.pop("mode", "Na_MC_input")
    full_na = parameters.pop("full_na_structure", None)
    if isinstance(full_na, (str, Path)):
        full_na_path = Path(full_na)
        if not full_na_path.is_file():
            raise FileNotFoundError(f"full Na structure missing: {full_na_path}")
        from pymatgen.core import Structure
        full_na = Structure.from_file(full_na_path)
    requested_steps = job.get("segment_budget")
    configured_steps = parameters.pop("max_steps", 30)
    max_steps = int(requested_steps if operation == "mc" and requested_steps is not None else configured_steps)
    parameters.pop("actual_step_limit_parameter", None)
    patience_steps = parameters.get("patience_steps", parameters.get("patience"))
    min_improvement = parameters.get("min_improvement")
    if "patience_steps" in parameters and "patience" not in parameters:
        parameters["patience"] = parameters.pop("patience_steps")

    model_paths = job.get("model_paths")
    mace_head = parameters.pop("mace_head", None)
    if not mace_head:
        identity = " ".join(str(job.get(key) or "") for key in
                             ("model_version", "model_path", "model_paths")).lower()
        if "mace-mh-1" in identity or "mh-1" in identity or "mh_1" in identity:
            mace_head = "omat_pbe"
    constructor = {"model_path": None if model_paths else job.get("model_path"), "model_type": "mace",
                   "device": parameters.pop("device", "cuda"),
                   "mace_head": mace_head, "out_dir": output,
                   "full_na_structure": full_na, "max_steps": max_steps, **parameters}
    if model_paths:
        constructor["model_paths"] = model_paths
    accepted = inspect.signature(LayeredOxide_MCOrderingClass).parameters
    if not any(item.kind == inspect.Parameter.VAR_KEYWORD for item in accepted.values()):
        constructor = {key: value for key, value in constructor.items() if key in accepted}
    runner = LayeredOxide_MCOrderingClass(**constructor)
    returned = runner.run(structure=job["structure_path"], mode=mode, out_dir=output)

    summary = _read_json(output / "pool" / "pool_summary.json", []) or []
    for row in summary:
        energies = _final_member_energies(output, row)
        if energies:
            row["energy_per_model_per_atom"] = energies
        for key in ("relax_traj_file", "relax_traj_contains_structure",
                    "relax_traj_contains_committee_info", "relax_traj_n_frames"):
            row.pop(key, None)
    _remove_relax_trajectories(output, summary)
    _write_json_gzip(output / "pool" / "pool_summary.json", summary)

    structure_path = output / "initial_relaxed.vasp"
    if summary:
        structure_path = output / "pool" / summary[0]["structure_file"]
    status_path = output / "status.json"
    status = _read_json(status_path, {}) or {}
    best = summary[0] if summary else {}
    energy_per_atom = best.get("energy_mean_per_atom")
    composition = None
    atom_count = None
    if structure_path.is_file():
        from pymatgen.core import Structure
        final = Structure.from_file(structure_path)
        composition, atom_count = final.composition.as_dict(), len(final)
    energy = energy_per_atom * atom_count if energy_per_atom is not None and atom_count else None
    member_count = len(model_paths) if model_paths else 1
    qbc = _qbc_from_summary(best, member_count)
    normal_stop = bool(summary and structure_path.is_file())
    structure_checksum = None
    if structure_path.is_file():
        from execution_layer.remote.integrity import file_checksum
        structure_checksum = file_checksum(structure_path)
    compressed_status = Path(f"{status_path}.gz")
    actual_mc_steps = _actual_mc_steps(output) if operation == "mc" else None
    reported_stop = status.get("stop_reason") if isinstance(status, dict) else None
    if not reported_stop and isinstance(status, dict) and status.get("early_stopped") is True:
        reported_stop = "patience"
    return {
        "status": "completed" if normal_stop else "failed",
        "structure_path": str(structure_path),
        "structure_checksum": structure_checksum,
        "checkpoint": str(compressed_status if compressed_status.exists() else status_path)
        if compressed_status.exists() or status_path.exists() else None,
        "segment_complete": normal_stop, "relax_stopped_normally": normal_stop,
        "strict_chain_resume": False if operation == "mc" else None,
        "restart_mode": "new_segment_from_structure" if operation == "mc" else None,
        "summary": status, "pool_summary": summary, "returned_structure": returned is not None,
        "energy": energy, "energy_per_atom": energy_per_atom,
        "energy_unit": "eV" if energy is not None else None,
        "converged": None, "composition": composition, "atom_count": atom_count,
        "qbc": qbc, "max_mc_steps": max_steps if operation == "mc" else None,
        "requested_max_mc_steps": max_steps if operation == "mc" else None,
        "patience_steps": patience_steps if operation == "mc" else None,
        "min_improvement": min_improvement if operation == "mc" else None,
        "actual_mc_steps": actual_mc_steps, "stop_reason": reported_stop or ("unknown" if operation == "mc" else None),
        "actual_gpu_core_hours": None, "actual_cost": None,
    }


if __name__ == "__main__":
    import sys
    job_path, result_path = map(Path, sys.argv[1:3])
    result_path.write_text(json.dumps(run_mace_worker(json.loads(job_path.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2), encoding="utf-8")
