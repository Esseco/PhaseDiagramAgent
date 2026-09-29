"""Identify a completed final structure once per file content, with a persistent cache."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


STAGES = {"relax_and_feature", "deep_search", "dft_single_point", "dft_relax"}


def identify_result_phase(result, manager, *, phase_references=None, cache=None):
    current = deepcopy(result)
    if current.get("stage") not in STAGES or current.get("status") != "completed":
        return current
    outputs = current.get("outputs") or {}
    structure_id = current.get("structure_id")
    branch_id = current.get("branch_id") or (
        manager.data.get("structures", {}).get(structure_id) or {}).get("branch_id")
    source_phase = (manager.data.get("branches", {}).get(branch_id) or {}).get("P")
    file_path = outputs.get("structure_path") or outputs.get("final_structure_path")
    path = Path(file_path) if file_path else None
    if path is None or not path.is_file():
        evidence = {"status": "unknown", "reason": "final_structure_missing"}
    else:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        scope = _classification_scope(manager, phase_references)
        cache_key = f"{scope}:{digest}"
        prior = outputs.get("phase_identification") or {}
        if (prior.get("structure_sha256") == digest and
                prior.get("classification_scope") == scope and
                prior.get("status") in {"identified", "outside_boundary"}):
            evidence = deepcopy(prior)
        elif (cache is not None and cache_key in cache
              and cache[cache_key].get("status") in {"identified", "outside_boundary"}):
            evidence = deepcopy(cache[cache_key])
        else:
            try:
                from pymatgen.core import Structure
                from scientific_layer.structures.identify_branch import (
                    calculate_x, identify_phase, identify_supercell)
                from scientific_layer.structures.identify_layered_phase_fast import identify_layered_phase_fast
                from scientific_layer.structures.boundary_utils import allowed_phases, allowed_phases_at_x
                structure = Structure.from_file(path)
                mobile = (((manager.data.get("system_config") or {}).get("species") or {})
                          .get("mobile") or ["Na"])[0]
                if any(site.is_ordered and site.specie.symbol == mobile for site in structure):
                    detected = identify_layered_phase_fast(structure, cation=mobile)
                    if detected.get("phase") == "X":
                        detected = identify_phase(structure, cation=mobile)
                    phase, method = detected.get("phase"), detected.get("method")
                else:
                    references = phase_references or (
                        (manager.data.get("system_config") or {}).get("phase_references") or {})
                    if not references:
                        raise ValueError("phase_references_required_for_empty_mobile_sites")
                    boundary = manager.boundary
                    allowed = allowed_phases_at_x(boundary["P"], calculate_x(structure))
                    from scientific_layer.structures.boundary_utils import load_structure
                    loaded = {name: load_structure(value) for name, value in references.items()}
                    hint = next(iter(allowed)) if len(allowed) == 1 else None
                    phase, _, _ = identify_supercell(
                        structure, loaded, boundary, phase_hint=hint,
                        tolerance=0.25, ambiguity_tolerance=1e-6)
                    if phase not in allowed:
                        raise ValueError("identified_phase_outside_composition_boundary")
                    method = "reference_supercell"
                boundary = getattr(manager, "boundary", None)
                if not phase or phase == "X":
                    raise ValueError(f"unrecognized_phase:{phase}")
                if boundary and phase not in allowed_phases(boundary["P"]):
                    evidence = {"status": "outside_boundary", "observed_phase": phase,
                                "method": method, "reason": "phase_outside_search_boundary"}
                elif boundary and mobile == "Na" and phase not in allowed_phases_at_x(
                        boundary["P"], calculate_x(structure)):
                    evidence = {"status": "outside_boundary", "observed_phase": phase,
                                "method": method, "reason": "phase_outside_composition_boundary"}
                else:
                    evidence = {"status": "identified", "phase": phase, "method": method}
            except Exception as error:
                evidence = {"status": "unknown", "reason": f"{type(error).__name__}: {error}"}
            evidence["structure_sha256"] = digest
            evidence["classification_scope"] = scope
            if cache is not None and evidence.get("status") in {"identified", "outside_boundary"}:
                cache[cache_key] = deepcopy(evidence)
    current["source_phase"] = source_phase
    current["actual_phase"] = evidence.get("phase") if evidence.get("status") == "identified" else None
    current["observed_phase"] = evidence.get("phase") or evidence.get("observed_phase")
    current["phase_identification_status"] = evidence["status"]
    if evidence.get("reason"):
        current["phase_identification_reason"] = evidence["reason"]
    current["outputs"] = {**outputs, "source_phase": source_phase,
                          "actual_phase": current["actual_phase"],
                          "observed_phase": current["observed_phase"],
                          "phase_identification": evidence}
    return current


def _classification_scope(manager, phase_references):
    references = phase_references or (
        (manager.data.get("system_config") or {}).get("phase_references") or {})
    described = {}
    for name, value in sorted(references.items()):
        if isinstance(value, (str, Path)):
            path = Path(value)
            stat = path.stat() if path.is_file() else None
            described[name] = [str(path), stat.st_size if stat else None,
                               stat.st_mtime_ns if stat else None]
        else:
            described[name] = str(value)
    payload = {"boundary": getattr(manager, "boundary", None),
               "mobile": (((manager.data.get("system_config") or {}).get("species") or {})
                          .get("mobile") or ["Na"]), "references": described}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
