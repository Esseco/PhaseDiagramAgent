"""在 py-mace 环境中执行 MACE Relax 或 Process_AL_MC。"""

from __future__ import annotations

import gzip
import inspect
import json
import os
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


def _pool_member_energies(runner, row, member_count, output):
    """Bind the saved ranking to the exact in-memory committee evaluation."""
    import math

    matches = [
        item
        for item in (getattr(runner, "pool", None) or [])
        if item.get("step") == row.get("step")
    ]
    if len(matches) > 1:
        return None
    values = (
        (matches[0].get("eval") or {}).get("energy_per_model_per_atom")
        if matches
        else _final_member_energies(output, row)
    )
    if not isinstance(values, (list, tuple)) or len(values) != member_count:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        return None
    mean = row.get("energy_mean_per_atom")
    if not isinstance(mean, (int, float)) or not math.isfinite(mean):
        return None
    if not math.isclose(sum(values) / member_count, mean, rel_tol=1e-10, abs_tol=1e-10):
        return None
    return [float(value) for value in values]


def _actual_mc_steps(output: Path) -> int | None:
    trace = _read_json(output / "trace.json", []) or []
    steps = [row.get("step") for row in trace if isinstance(row.get("step"), int)]
    return max(steps) if steps else None


def _qbc_from_summary(best: dict, member_count: int) -> dict:
    if member_count < 2:
        return {
            "status": "not_available_single_model",
            "member_count": member_count,
            "interpretation": "single_model_has_no_committee_uncertainty",
        }
    if best.get("energy_per_model_per_atom"):
        from phase_agent.science.qbc.summarize_member_energies import summarize_member_energies

        result = summarize_member_energies(
            best["energy_per_model_per_atom"], expected_members=member_count
        )
        result["force_rms_disagreement"] = best.get("force_uncertainty_per_atom")
        result["f_std_max"] = best.get("force_uncertainty_max")
        return result
    return {
        "status": "completed",
        "member_count": member_count,
        "energy_per_atom_std": best.get("energy_std_per_atom"),
        "energy_per_atom_var": best.get("energy_var_per_atom"),
        "force_rms_disagreement": best.get("force_uncertainty_per_atom"),
        "f_std_max": best.get("force_uncertainty_max"),
        "interpretation": "committee_disagreement_not_true_error",
    }


def _extrapolation_monitor_factory():
    """Use the audited formation-energy embedding; fail closed on missing weights."""
    import importlib.util

    if os.environ.get("TF_USE_LEGACY_KERAS", "1") != "1":
        raise ValueError("extrapolation_monitor_requires_legacy_keras")
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    spec = importlib.util.find_spec("maml")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("extrapolation_monitor_maml_unavailable")
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "describers/data/m3gnet_models/matbench_mp_e_form/0/m3gnet"
    )
    from Process_MLIP.direct_sampling.features import M3GNetEmbedding
    from Process_MLIP.collapse_monitor import CollapseMonitor
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    encoder = M3GNetEmbedding(path)
    if (
        encoder.metadata["model_sha256"]
        != "87db513eeb8c346d667677f3041982c5fd0a4304d8e5071d9c6c22fded04eb0d"
    ):
        raise ValueError("extrapolation_monitor_model_checksum_mismatch")
    return lambda atoms: CollapseMonitor(atoms, encoder, stop_on_extrapolation=False)


def _threshold_monitor_factory(threshold):
    """Apply a scope-specific threshold while preserving the original monitor."""
    if isinstance(threshold, bool) or threshold not in (1.0, 3.0):
        raise ValueError("invalid_feature_difference_threshold")
    factory = _extrapolation_monitor_factory()
    if threshold == 1.0:
        return factory

    def make(initial):
        monitor = factory(initial)

        class ScopedMonitor:
            @property
            def failures(self):
                return [row for row in monitor.failures if row["score"] > threshold]

            def __call__(self, atoms):
                return monitor(atoms)

            def check_final(self, atoms):
                return monitor.check_final(atoms)

        return ScopedMonitor()

    return make


def run_mace_worker(job: dict) -> dict:
    output = Path(job["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    parameters = dict(job.get("parameters") or {})
    operation = job["operation"]
    feature_threshold = parameters.pop("feature_difference_threshold", 1.0)
    if isinstance(feature_threshold, bool) or feature_threshold not in (1.0, 3.0):
        raise ValueError("invalid_feature_difference_threshold")
    if parameters.get("model_refresh_operation") and feature_threshold != 1.0:
        raise ValueError("model_refresh_feature_threshold_must_be_one")
    if operation == "predict":
        from phase_agent.science.mlip.predict_structure import predict_structure

        if parameters.get("refresh_force_threshold") is not None:
            from phase_agent.science.mlip.refresh_inline import run_inline_refresh

            return run_inline_refresh(job, predict_structure, run_mace_worker)
        return predict_structure(job)
    if operation == "relax":
        optimizer = parameters.pop("optimizer", "LBFGS")
        if optimizer != "LBFGS":
            raise ValueError("Standalone Relax requires LBFGS")
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
        framework = str(parameters.get("framework") or "mace").lower()
        if framework != "mace":
            # Generic ASE path for the selectable local Process_MLIP factory.
            from ase.io import read, write
            from ase.optimize import BFGS
            from phase_agent.science.mlip.model_interface import create_calculator

            input_path = Path(job["structure_path"])
            atoms = read(
                input_path, format="vasp" if input_path.suffix.lower() == ".vasp" else None
            )
            atoms.calc = create_calculator(
                {
                    **parameters,
                    "framework": framework,
                    "model_name": parameters.get("model_name") or job.get("model_version"),
                    "local_repo_path": parameters.get("local_repo_path"),
                },
                model_path=model_path,
            )
            optimizer = BFGS(atoms, logfile=None)
            monitor = _threshold_monitor_factory(feature_threshold)(atoms)
            optimizer.attach(lambda: monitor(atoms), interval=1)
            optimizer.run(
                fmax=float(parameters.pop("fmax", 0.05)),
                steps=int(parameters.pop("relax_steps", 500)),
            )
            structure_path = output / "initial_relaxed.vasp"
            write(structure_path, atoms, format="vasp", direct=True)
            final_difference = monitor.check_final(atoms)
            from phase_agent.tools.remote.integrity import file_checksum

            return {
                "status": "completed",
                "optimizer": "BFGS",
                "extrapolation_monitor": {
                    "final_feature_difference": final_difference,
                    "feature_difference_threshold": feature_threshold,
                    "extrapolation_failure": bool(monitor.failures),
                    "failures": monitor.failures,
                },
                "structure_path": str(structure_path),
                "structure_checksum": file_checksum(structure_path),
                "composition": {
                    s: atoms.get_chemical_symbols().count(s)
                    for s in set(atoms.get_chemical_symbols())
                },
                "atom_count": len(atoms),
                "mlip_name": framework.upper(),
                "mlip_version": job.get("model_version"),
                "qbc": {"status": "not_available_single_model", "member_count": 1},
            }

        from Process_AL_MC.relax import relax_structure_mace

        mace_head = parameters.pop("mace_head", None)
        if mace_head is None:
            identity = " ".join(
                str(value or "") for value in (job.get("model_version"), model_path)
            ).lower()
            if "mace-mh-1" in identity or "mh-1" in identity or "mh_1" in identity:
                mace_head = "omat_pbe"
        structure_path = output / "initial_relaxed.vasp"
        from ase.io import read as read_atoms

        source = Path(job["structure_path"])
        initial_atoms = read_atoms(
            source, format="vasp" if source.suffix.lower() == ".vasp" else None
        )
        monitor = _threshold_monitor_factory(feature_threshold)(initial_atoms)
        result = relax_structure_mace(
            job["structure_path"],
            model_path,
            output_path=structure_path,
            device=parameters.pop("device", "cuda"),
            head=mace_head,
            default_dtype=parameters.pop("mace_default_dtype", "float64"),
            fmax=parameters.pop("fmax", 0.05),
            steps=parameters.pop("relax_steps", 500),
            relax_cell=parameters.pop("relax_cell", True),
            step_callback=monitor,
        )
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        from Process_AL_MC.MC_sample import _relaxation_input_identity

        monitor_result = result.get("extrapolation_monitor") or {}
        if monitor_result:
            monitor_result["feature_difference_threshold"] = feature_threshold
        result["relaxation_observation"] = {
            "structure_id": _relaxation_input_identity(
                AseAtomsAdaptor.get_structure(initial_atoms)
            ),
            "identity_scheme": "periodic_input_v1",
            "monitor_complete": bool(monitor_result),
            "extrapolation_failure": monitor_result.get("extrapolation_failure"),
        }
        final = Structure.from_file(structure_path)
        from phase_agent.tools.remote.integrity import file_checksum

        result.update(
            {
                "status": "completed",
                "optimizer": optimizer,
                "structure_path": str(structure_path),
                "structure_checksum": file_checksum(structure_path),
                "composition": final.composition.as_dict(),
                "atom_count": len(final),
                "mlip_name": "MACE",
                "mlip_version": job.get("model_version"),
                "qbc": {"status": "not_available_single_model", "member_count": 1},
            }
        )
        if (job.get("parameters") or {}).get("model_refresh_operation") and result.get(
            "optimizer_converged"
        ) is True:
            from phase_agent.science.mlip.predict_structure import predict_structure

            final_job = {**job, "structure_path": str(structure_path)}
            prediction = predict_structure(final_job)
            prediction.pop("single_point_completed", None)
            result.update(prediction)
        return result

    if operation != "mc":
        raise ValueError(f"unsupported MLIP operation: {operation}")

    framework = str(parameters.get("framework") or "mace").lower()
    if framework not in {"mace", "chgnet"}:
        raise ValueError(
            f"MLIP framework {framework!r} does not support the current LayeredOxide MC worker; "
            "choose mace/chgnet for MC or use the framework's dedicated MC adapter."
        )

    from Process_AL_MC import LayeredOxide_MCOrderingClass

    parameters.setdefault("save_relax_traj", True)
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
    max_steps = int(
        requested_steps if operation == "mc" and requested_steps is not None else configured_steps
    )
    parameters.pop("actual_step_limit_parameter", None)
    patience_steps = parameters.get("patience_steps", parameters.get("patience"))
    min_improvement = parameters.get("min_improvement")
    if min_improvement is None or job.get("stopping_threshold_source") == "library_default":
        # Leaving this argument absent preserves the calculation library default.
        parameters.pop("min_improvement", None)
        min_improvement = None
    if "patience_steps" in parameters:
        parameters["patience"] = parameters.pop("patience_steps")

    model_paths = job.get("model_paths")
    mace_head = parameters.pop("mace_head", None)
    if not mace_head:
        identity = " ".join(
            str(job.get(key) or "") for key in ("model_version", "model_path", "model_paths")
        ).lower()
        if "mace-mh-1" in identity or "mh-1" in identity or "mh_1" in identity:
            mace_head = "omat_pbe"
    constructor = {
        "model_path": None if model_paths else job.get("model_path"),
        "model_type": "mace",
        "device": parameters.pop("device", "cuda"),
        "mace_head": mace_head,
        "out_dir": output,
        "full_na_structure": full_na,
        "max_steps": max_steps,
        **parameters,
    }
    if framework != "mace":
        raise ValueError("extrapolation_monitor_mc_requires_mace_backend")
    constructor["relaxation_monitor_factory"] = _threshold_monitor_factory(feature_threshold)
    if model_paths:
        constructor["model_paths"] = model_paths
    accepted = inspect.signature(LayeredOxide_MCOrderingClass).parameters
    if "relaxation_monitor_factory" not in accepted and not any(
        item.kind == inspect.Parameter.VAR_KEYWORD for item in accepted.values()
    ):
        raise RuntimeError("mc_extrapolation_monitor_not_supported_by_installed_Process_AL_MC")
    if not any(item.kind == inspect.Parameter.VAR_KEYWORD for item in accepted.values()):
        constructor = {key: value for key, value in constructor.items() if key in accepted}
    if (
        operation == "mc"
        and getattr(LayeredOxide_MCOrderingClass, "MC_ENERGY_BASIS", None) != "main_model"
    ):
        raise RuntimeError("mc_backend_main_model_energy_required")
    runner = LayeredOxide_MCOrderingClass(**constructor)
    from Process_AL_MC.MC_sample import InitialExtrapolationFailure

    try:
        returned = runner.run(structure=job["structure_path"], mode=mode, out_dir=output)
    except InitialExtrapolationFailure as error:
        item = error.evidence
        item["extrapolation_monitor"]["feature_difference_threshold"] = feature_threshold
        return {
            "status": "failed",
            "error": "initial_relaxation_extrapolation_failure",
            "structure_path": item["structure_path"],
            "structure_checksum": item["structure_checksum"],
            "extrapolation_monitor": item["extrapolation_monitor"],
            "extrapolation_failures": [item],
            "mlip_version": job.get("model_version"),
        }

    summary = _read_json(output / "pool" / "pool_summary.json", []) or []
    member_count = len(model_paths) if model_paths else 1
    for row in summary:
        energies = _pool_member_energies(runner, row, member_count, output)
        row.pop("energy_per_model_per_atom", None)
        if energies:
            row["energy_per_model_per_atom"] = energies
    _write_json_gzip(output / "pool" / "pool_summary.json", summary)

    structure_path = output / "initial_relaxed.vasp"
    if summary:
        structure_path = output / "pool" / summary[0]["structure_file"]
    status_path = output / "status.json"
    status = _read_json(status_path, {}) or {}
    best = summary[0] if summary else {}
    committee_energy = best.get("energy_mean_per_atom")
    members = best.get("energy_per_model_per_atom") or []
    main_index = int((job.get("parameters") or {}).get("main_model_index", 0))
    if summary and (not 0 <= main_index < member_count or len(members) != member_count):
        raise ValueError("mc_selected_frame_member_energy_unavailable")
    energy_per_atom = members[main_index] if members else committee_energy
    composition = None
    atom_count = None
    if structure_path.is_file():
        from pymatgen.core import Structure

        final = Structure.from_file(structure_path)
        composition, atom_count = final.composition.as_dict(), len(final)
    energy = energy_per_atom * atom_count if energy_per_atom is not None and atom_count else None
    qbc = _qbc_from_summary(best, member_count)
    normal_stop = bool(summary and structure_path.is_file())
    structure_checksum = None
    if structure_path.is_file():
        from phase_agent.tools.remote.integrity import file_checksum

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
        if compressed_status.exists() or status_path.exists()
        else None,
        "segment_complete": normal_stop,
        "relax_stopped_normally": normal_stop,
        "strict_chain_resume": False if operation == "mc" else None,
        "restart_mode": "new_segment_from_structure" if operation == "mc" else None,
        "summary": status,
        "pool_summary": summary,
        "structure_check_counts": status.get("structure_check_counts"),
        "returned_structure": returned is not None,
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "energy_source": "main_model_selected_frame",
        "main_model_index": main_index,
        "committee_energy_per_atom": committee_energy,
        "energy_unit": "eV" if energy is not None else None,
        "converged": None,
        "composition": composition,
        "atom_count": atom_count,
        "qbc": qbc,
        "max_mc_steps": max_steps if operation == "mc" else None,
        "requested_max_mc_steps": max_steps if operation == "mc" else None,
        "patience_steps": patience_steps if operation == "mc" else None,
        "min_improvement": min_improvement if operation == "mc" else None,
        "actual_mc_steps": actual_mc_steps,
        "stop_reason": reported_stop or ("unknown" if operation == "mc" else None),
        "actual_gpu_core_hours": None,
        "actual_cost": None,
        "feature_difference_threshold": feature_threshold,
        "extrapolation_failures": list(getattr(runner, "extrapolation_failures", [])),
        "relaxation_observations": [
            {**row, "feature_difference_threshold": feature_threshold}
            for row in getattr(runner, "relaxation_observations", [])
        ],
        "relaxation_observations_complete": bool(
            normal_stop
            and type(actual_mc_steps) is int
            and actual_mc_steps >= 0
            and len(getattr(runner, "relaxation_observations", [])) == actual_mc_steps + 1
            and all(row.get("monitor_complete") is True for row in runner.relaxation_observations)
        ),
    }


if __name__ == "__main__":
    import sys

    job_path, result_path = map(Path, sys.argv[1:3])
    result_path.write_text(
        json.dumps(
            run_mace_worker(json.loads(job_path.read_text(encoding="utf-8"))),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
