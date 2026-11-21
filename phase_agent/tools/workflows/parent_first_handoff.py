"""Build a minimal HPC input archive from verified parent and DFT reference frames."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_initial_parents(state, ledger, explicit_parents):
    """Normalize dict or structure_id initial_parents entries against state and ledger."""
    structures = (ledger or {}).get("structures") or {}
    branches = (ledger or {}).get("branches") or {}
    registered = {
        reg.get("structure_id"): reg
        for task in state.get("tasks") or []
        for reg in task.get("registered") or []
        if isinstance(reg, dict) and reg.get("structure_id")
    }
    resolved = []
    for item in explicit_parents:
        if isinstance(item, dict):
            resolved.append(item)
            continue
        if isinstance(item, str) and item:
            reg = registered.get(item) or {}
            struct = structures.get(item) or {}
            branch_id = reg.get("branch_id") or struct.get("branch_id")
            branch = branches.get(branch_id) or {}
            source_path = (
                reg.get("structure_path")
                or struct.get("source_path")
                or struct.get("structure_path")
            )
            resolved.append(
                {
                    "structure_id": item,
                    "branch_id": branch_id,
                    "phase": struct.get("phase") or branch.get("phase"),
                    "H": struct.get("H") or branch.get("H"),
                    "x_Na_per_O2": (
                        struct.get("x_Na_per_O2")
                        if struct.get("x_Na_per_O2") is not None
                        else branch.get("x_Na_per_O2")
                    ),
                    "source_path": source_path,
                    "structure_sha256": struct.get("structure_sha256"),
                }
            )
    return resolved


def prepare_parent_first_bundle(
    state,
    ledger,
    parameters,
    output_dir,
    *,
    feature_model_path,
    feature_model_sha256,
    dft_settings=None,
):
    """Package approved inputs only. No perturbation, relaxation or DIRECT runs here."""
    from ase.io import write
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from phase_agent.agents.reliability.extrapolation_round import required_direct_round
    from phase_agent.evidence.perturbation_recovery import build_perturbation_recovery_pool
    from phase_agent.tools.remote.transfer_bundle import create_bundle
    from phase_agent.tools.workflows.comparison_model_registry import original_round_model
    from phase_agent.tools.workflows.direct_extrapolation_sampling import successful_dft_references

    initial_direct = parameters.get("scope_id") == "initial_direct"
    if initial_direct:
        required = {"scope_id": "initial_direct", "evidence_digest": "initial_direct_v1"}
        sampling_method = "direct"
    else:
        required = required_direct_round(state)
        sampling_method = parameters.get("sampling_method", "kmeans_pp")
        if (
            not required
            or parameters.get("confirmed") is not True
            or parameters.get("scope_id") != required["scope_id"]
            or parameters.get("round_evidence_digest") != required["evidence_digest"]
        ):
            raise ValueError("parent_first_round_or_approval_invalid")
    if parameters.get("confirmed") is not True:
        raise ValueError("parent_first_round_or_approval_invalid")
    target = parameters.get("direct_target")
    frames = parameters.get("frames_per_parent", 36)
    seed = parameters.get("seed", 42)
    distance = parameters.get("min_distance_A", 1.0)
    volume = parameters.get("volume_ratio", (0.75, 1.25))
    feature_threshold = parameters.get("feature_difference_threshold", 1.0)
    if (
        type(target) is not int
        or not 1 <= target <= 100
        or type(frames) is not int
        or frames < 1
        or type(seed) is not int
        or seed < 0
        or isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(distance)
        or distance <= 0
        or isinstance(feature_threshold, bool)
        or not isinstance(feature_threshold, (int, float))
        or float(feature_threshold) not in {1.0, 3.0}
        or not isinstance(volume, (list, tuple))
        or len(volume) != 2
        or not 0 < volume[0] <= 1 <= volume[1]
    ):
        raise ValueError("parent_first_parameters_invalid")
    if sampling_method == "direct" and (
        not PurePosixPath(str(feature_model_path or "")).is_absolute()
        or not re.fullmatch(r"[a-fA-F0-9]{64}", str(feature_model_sha256 or ""))
    ):
        raise ValueError("parent_first_feature_model_invalid")
    if initial_direct:
        hull_version = "initial"
        explicit_parents = parameters.get("initial_parents")
        if not isinstance(explicit_parents, list) or not explicit_parents:
            raise ValueError("initial_direct_requires_explicit_initial_parents")
        explicit_parents = _resolve_initial_parents(state, ledger, explicit_parents)
        available = {
            row["structure_id"]: {
                "record_id": row["structure_id"],
                "branch_id": row.get("branch_id"),
                "phase": row.get("phase"),
                "H": row.get("H"),
                "x_Na_per_O2": row.get("x_Na_per_O2"),
                "structure_path": row.get("source_path") or row.get("structure_path"),
                "structure_sha256": row.get("structure_sha256")
                or (
                    _hash(row["source_path"])
                    if row.get("source_path") and Path(row["source_path"]).is_file()
                    else None
                ),
            }
            for row in explicit_parents
            if isinstance(row, dict)
            and row.get("structure_id")
            and (row.get("source_path") or row.get("structure_path"))
        }
        ids = parameters.get("parent_record_ids")
        if ids is None:
            ids = [
                row["structure_id"]
                for row in explicit_parents
                if isinstance(row, dict) and row.get("structure_id")
            ]
    else:
        pool = build_perturbation_recovery_pool(
            state, ledger, max_det_H=parameters.get("max_det_H")
        )
        hull_version = pool["hull_version"]
        available = {row["record_id"]: row for row in pool["candidates"]}
        ids = parameters.get("parent_record_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != len(set(ids))
        or any(i not in available for i in ids)
    ):
        raise ValueError("parent_first_parent_selection_invalid")
    model, model_sha, _ = original_round_model(state, {}, "mace-mh-1")
    if initial_direct and not model_sha:
        model_sha = parameters.get("model_sha256") or model.get("model_sha256")
    if (
        model.get("framework", "").lower() != "mace"
        or model.get("mace_head") != "omat_pbe"
        or not PurePosixPath(str(model.get("model_path") or "")).is_absolute()
        or not re.fullmatch(r"[a-fA-F0-9]{64}", str(model_sha or ""))
    ):
        raise ValueError("parent_first_mh1_binding_invalid")
    refs = successful_dft_references(state)
    if dft_settings is not None:
        from phase_agent.science.dft.parent_first_settings import validate_parent_first_dft_settings

        validate_parent_first_dft_settings(dft_settings)
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError("parent_first_bundle_exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parent-first-input-", dir=out.parent) as temp:
        root = Path(temp) / "payload"
        root.mkdir()
        (root / "parents").mkdir()
        (root / "references").mkdir()
        parents = []
        for index, key in enumerate(ids):
            row = available[key]
            source = Path(row["structure_path"])
            if not source.is_file() or _hash(source) != row["structure_sha256"]:
                raise ValueError("parent_first_parent_changed")
            relative = f"parents/{index:04d}.vasp"
            shutil.copyfile(source, root / relative)
            parents.append(
                {
                    "record_id": key,
                    "branch_id": row["branch_id"],
                    "phase": row["phase"],
                    "H": row["H"],
                    "x_Na_per_O2": row["x_Na_per_O2"],
                    "path": relative,
                    "sha256": row["structure_sha256"],
                }
            )
        references = []
        for index, row in enumerate(refs):
            relative = f"references/{index:04d}.vasp"
            write(
                root / relative,
                AseAtomsAdaptor.get_atoms(Structure.from_dict(row["structure"])),
                format="vasp",
                direct=True,
            )
            references.append(
                {"task_id": row["task_id"], "path": relative, "sha256": _hash(root / relative)}
            )
        manifest = {
            "schema": "parent-first-hpc-input-v1",
            "scope_id": required["scope_id"],
            "sampling_method": sampling_method,
            "round_evidence_digest": required["evidence_digest"],
            "baseline_hull_version": hull_version,
            "model_path": str(model["model_path"]),
            "model_sha256": model_sha.lower(),
            "feature_model_path": str(feature_model_path) if feature_model_path else None,
            "feature_model_sha256": feature_model_sha256.lower() if feature_model_sha256 else None,
            "target": target,
            "frames_per_parent": frames,
            "seed": seed,
            "min_distance_A": distance,
            "volume_ratio": list(volume),
            "feature_difference_threshold": float(feature_threshold),
            "parents": parents,
            "references": references,
        }
        if dft_settings is not None:
            manifest["dft_settings"] = dft_settings
        (root / "input_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
        )
        export = Path(temp) / "export"
        export.mkdir()
        archive = export / "inputs.zip"
        files = [p for p in root.rglob("*") if p.is_file()]
        create_bundle(root, files, archive, kind="inputs")
        receipt = {
            "schema": "parent-first-hpc-bundle-v1",
            "scope_id": required["scope_id"],
            "round_evidence_digest": required["evidence_digest"],
            "parent_count": len(parents),
            "dft_reference_count": len(references),
            "target": target,
            "file_count": len(files),
            "archive_sha256": _hash(archive),
            "archive_size": archive.stat().st_size,
        }
        (export / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8"
        )
        export.rename(out)
    return {
        "status": "prepared",
        "archive": str(out / "inputs.zip"),
        "receipt": receipt,
        "submitted": False,
    }
