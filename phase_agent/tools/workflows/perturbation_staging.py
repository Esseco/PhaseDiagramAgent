"""Immutable preparation of verified perturbation-recovery inputs.

This module prepares structures only. It deliberately does not clear the
reliability gate or pretend that relaxation, DIRECT, DFT or training ran.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from copy import deepcopy
from pathlib import Path

from phase_agent.agents.reliability.extrapolation_round import required_direct_round
from phase_agent.evidence.perturbation_recovery import build_perturbation_recovery_pool
from phase_agent.tools.workflows.comparison_model_registry import original_round_model


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_perturbation_inputs(state, ledger, parameters, output_dir):
    """Freeze a reliability-bound batch, without modifying the project state."""
    required = required_direct_round(state)
    if not required or any(parameters.get(k) != required[v] for k, v in
                           (("scope_id", "scope_id"), ("round_evidence_digest", "evidence_digest"))):
        raise ValueError("perturbation_round_evidence_mismatch")
    if parameters.get("confirmed") is not True:
        raise ValueError("perturbation_selection_not_confirmed")
    pool = build_perturbation_recovery_pool(state, ledger, max_det_H=parameters.get("max_det_H"))
    available = {row["record_id"]: row for row in pool["candidates"]}
    ids = parameters.get("parent_record_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(x not in available for x in ids):
        raise ValueError("perturbation_parent_selection_invalid")
    model, digest, _ = original_round_model(state, {}, "mace-mh-1")
    if (str(model.get("framework", "")).lower() != "mace"
        or model.get("mace_head") != "omat_pbe"
        or not model.get("model_path")
        or not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest)):
        raise ValueError("perturbation_mh1_binding_invalid")
    digest = digest.lower()
    # These are passed through to Process_MLIP; the Agent does not implement
    # perturbation or geometry filtering itself.
    from phase_agent.tools.workflows.direct_extrapolation_sampling import successful_dft_references
    references = successful_dft_references(state)
    frames_per_parent = parameters.get("frames_per_parent", 36)
    seed = parameters.get("seed", 42)
    min_distance_A = parameters.get("min_distance_A", 1.0)
    volume_ratio = parameters.get("volume_ratio", (0.75, 1.25))
    if type(frames_per_parent) is not int or frames_per_parent < 1 or type(seed) is not int or seed < 0:
        raise ValueError("perturbation_parameters_invalid")
    target = Path(output_dir)
    if target.exists():
        try:
            saved_path = target / "manifest.json"
            saved = json.loads(saved_path.read_text(encoding="utf8"))
            expected_params = {"frames_per_parent": frames_per_parent, "seed": seed,
                               "max_det_H": parameters["max_det_H"], "min_distance_A": min_distance_A,
                               "volume_ratio": list(volume_ratio)}
            binding_ok = (saved.get("schema") == "perturbation-recovery-inputs-v1"
                and saved.get("scope_id") == required["scope_id"]
                and saved.get("round_evidence_digest") == required["evidence_digest"]
                and saved.get("failed_model_version") == state.get("active_model_version")
                and saved.get("baseline_hull_version") == pool["hull_version"]
                and saved.get("baseline_model_version") == pool["model_version"]
                and saved.get("relax_model", {}).get("model") == model
                and saved.get("relax_model", {}).get("sha256") == digest
                and saved.get("parameters") == expected_params
                and saved.get("parents") == [available[x] for x in ids]
                and saved.get("direct_dft_references") == references
                and saved.get("status") == "structures_prepared")
            frames = saved.get("frames") or []
            for parent in saved.get("parents") or []:
                binding_ok = binding_ok and _sha(parent["structure_path"]) == parent["structure_sha256"]
            for frame in frames:
                path = (target / frame["path"]).resolve()
                binding_ok = binding_ok and path.is_relative_to(target.resolve()) and path.is_file() and _sha(path) == frame["sha256"]
            if binding_ok and frames:
                return {"status": "structures_prepared", "directory": str(target),
                        "manifest_path": str(saved_path), "frame_count": len(frames), "submitted": False,
                        "reused": True}
        except (OSError, ValueError, KeyError, TypeError):
            pass
        raise FileExistsError("perturbation_existing_inputs_require_recovery:" + str(target))
    target.parent.mkdir(parents=True, exist_ok=True)
    from ase.io import read, write
    from Process_MLIP.nepkit_sampling import perturb_layered_oxide_structure

    with tempfile.TemporaryDirectory(prefix=".perturbation-", dir=target.parent) as scratch:
        root = Path(scratch) / "batch"
        root.mkdir()
        rows = []
        for parent_index, record_id in enumerate(ids):
            parent = available[record_id]
            # Recheck immediately before use; the earlier pool check is not a lock.
            if _sha(parent["structure_path"]) != parent["structure_sha256"]:
                raise ValueError("perturbation_parent_changed")
            work = root / ("parent_%04d" % parent_index)
            generated = perturb_layered_oxide_structure(parent["structure_path"], work,
                frames_per_parent=frames_per_parent, seed=seed + parent_index,
                min_distance_A=min_distance_A, volume_ratio=volume_ratio,
                metadata={"record_id": record_id, "parent_sha256": parent["structure_sha256"]})
            frames = read(generated["structures"], index=":")
            sampler = json.loads(Path(generated["manifest"]).read_text(encoding="utf8"))
            if len(frames) != generated["n_frames"] or len(frames) != len(sampler["identities"]):
                raise ValueError("perturbation_sampler_frame_count_mismatch")
            # The sampler records an absolute temporary path; make it portable
            # before the batch is atomically moved to its final directory.
            sampler["structures"] = "structures.extxyz"
            Path(generated["manifest"]).write_text(
                json.dumps(sampler, ensure_ascii=False, indent=2), encoding="utf8")
            for index, (frame, identity) in enumerate(zip(frames, sampler["identities"])):
                path = work / ("frame_%04d.vasp" % index)
                write(path, frame, format="vasp", direct=True)
                rows.append({"parent_record_id": record_id, "parent_sha256": parent["structure_sha256"],
                    "branch_id": parent["branch_id"], "phase": parent["phase"],
                    "x_Na_per_O2": parent["x_Na_per_O2"], "H": parent["H"],
                    "perturbation_index": identity["perturbation_index"],
                    "path": path.relative_to(root).as_posix(), "sha256": _sha(path)})
        manifest = {"schema": "perturbation-recovery-inputs-v1", "scope_id": required["scope_id"],
            "round_evidence_digest": required["evidence_digest"], "failed_model_version": state.get("active_model_version"),
            "baseline_hull_version": pool["hull_version"], "baseline_model_version": pool["model_version"],
            "relax_model": {"version": "mace-mh-1", "model": deepcopy(model), "sha256": digest,
                            "remote_content_verified": False},
            "parameters": {"frames_per_parent": frames_per_parent, "seed": seed, "max_det_H": parameters["max_det_H"], "min_distance_A": min_distance_A, "volume_ratio": list(volume_ratio)},
            "parents": [available[x] for x in ids], "frames": rows,
            "direct_dft_references": references,
            "status": "structures_prepared", "submitted": False}
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
        root.rename(target)
    return {"status": "structures_prepared", "directory": str(target),
            "manifest_path": str(target / "manifest.json"), "frame_count": len(rows), "submitted": False}
