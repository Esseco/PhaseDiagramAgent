"""Identify a completed final structure once per structure content."""

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
        manager.data.get("structures", {}).get(structure_id) or {}
    ).get("branch_id")
    source_phase = (manager.data.get("branches", {}).get(branch_id) or {}).get("P")
    try:
        from phase_agent.science.structures.load_result_structure import (
            load_result_structure,
        )

        cached_file = _cached_file_identity(outputs, manager, phase_references)
        if cached_file is None:
            structure, digest = load_result_structure(outputs)
            configured_phase = _configured_phase(structure, manager)
        else:
            # Exact file content and classification rules still match a prior
            # successful identification. No structure parsing/detection is needed.
            digest, configured_phase = cached_file
            structure = None
    except FileNotFoundError:
        evidence = {"status": "unknown", "reason": "final_structure_missing"}
    except Exception as error:
        evidence = {"status": "unknown", "reason": f"{type(error).__name__}: {error}"}
    else:
        scope = _classification_scope(manager, phase_references, configured_phase=configured_phase)
        cache_key = f"{scope}:{digest}"
        prior = outputs.get("phase_identification") or {}
        if (
            prior.get("structure_sha256") == digest
            and prior.get("classification_scope") == scope
            and prior.get("status") in {"identified", "outside_boundary"}
        ):
            evidence = deepcopy(prior)
        elif (
            cache is not None
            and cache_key in cache
            and cache[cache_key].get("status") in {"identified", "outside_boundary"}
        ):
            evidence = deepcopy(cache[cache_key])
        else:
            try:
                from phase_agent.science.structures.identify_branch import (
                    calculate_x,
                    identify_phase,
                    identify_supercell,
                )
                from phase_agent.science.structures.identify_layered_phase_fast import (
                    identify_layered_phase_fast,
                )
                from phase_agent.science.structures.boundary_utils import (
                    allowed_phases,
                    allowed_phases_at_x,
                )

                mobile = (
                    ((manager.data.get("system_config") or {}).get("species") or {}).get("mobile")
                    or ["Na"]
                )[0]
                if configured_phase:
                    phase, method = configured_phase, "unique_configured_phase"
                    if phase == "O1" and _legacy_na0_default_enabled(manager):
                        method = "Na_free_O1_default"
                elif any(site.is_ordered and site.specie.symbol == mobile for site in structure):
                    detected = identify_layered_phase_fast(structure, cation=mobile)
                    if detected.get("phase") == "X":
                        detected = identify_phase(structure, cation=mobile)
                    phase, method = detected.get("phase"), detected.get("method")
                else:
                    boundary = manager.boundary
                    allowed = allowed_phases_at_x(boundary["P"], calculate_x(structure))
                    references = phase_references or (
                        (manager.data.get("system_config") or {}).get("phase_references") or {}
                    )
                    if not references:
                        raise ValueError("phase_references_required_for_empty_mobile_sites")
                    boundary = manager.boundary
                    allowed = allowed_phases_at_x(boundary["P"], calculate_x(structure))
                    from phase_agent.science.structures.boundary_utils import (
                        load_structure,
                    )

                    loaded = {name: load_structure(value) for name, value in references.items()}
                    hint = next(iter(allowed)) if len(allowed) == 1 else None
                    phase, _, _ = identify_supercell(
                        structure,
                        loaded,
                        boundary,
                        phase_hint=hint,
                        tolerance=0.25,
                        ambiguity_tolerance=1e-6,
                    )
                    if phase not in allowed:
                        raise ValueError("identified_phase_outside_composition_boundary")
                    method = "reference_supercell"
                boundary = getattr(manager, "boundary", None)
                if not phase or phase == "X":
                    raise ValueError(f"unrecognized_phase:{phase}")
                if boundary and phase not in allowed_phases(boundary["P"]):
                    evidence = {
                        "status": "outside_boundary",
                        "observed_phase": phase,
                        "method": method,
                        "reason": "phase_outside_search_boundary",
                    }
                elif (
                    boundary
                    and mobile == "Na"
                    and phase not in allowed_phases_at_x(boundary["P"], calculate_x(structure))
                ):
                    evidence = {
                        "status": "outside_boundary",
                        "observed_phase": phase,
                        "method": method,
                        "reason": "phase_outside_composition_boundary",
                    }
                else:
                    evidence = {
                        "status": "identified",
                        "phase": phase,
                        "method": method,
                    }
            except Exception as error:
                evidence = {
                    "status": "unknown",
                    "reason": f"{type(error).__name__}: {error}",
                }
            evidence["structure_sha256"] = digest
            evidence["classification_scope"] = scope
            if cache is not None and evidence.get("status") in {
                "identified",
                "outside_boundary",
            }:
                cache[cache_key] = deepcopy(evidence)
    current["source_phase"] = source_phase
    current["actual_phase"] = (
        evidence.get("phase") if evidence.get("status") == "identified" else None
    )
    current["observed_phase"] = evidence.get("phase") or evidence.get("observed_phase")
    current["phase_identification_status"] = evidence["status"]
    if evidence.get("reason"):
        current["phase_identification_reason"] = evidence["reason"]
    else:
        current.pop("phase_identification_reason", None)
    current["outputs"] = {
        **outputs,
        "source_phase": source_phase,
        "actual_phase": current["actual_phase"],
        "observed_phase": current["observed_phase"],
        "phase_identification": evidence,
    }
    return current


def _cached_file_identity(outputs, manager, phase_references):
    # Inline geometry has a canonical structural digest and retains its existing
    # parser. Invalid/mismatched final-frame metadata must still be rejected there.
    if outputs.get("structure") is not None or outputs.get("final_frame_valid") is False:
        return None
    if (
        outputs.get("final_frame_index") is not None
        and outputs.get("training_frame_index") != outputs["final_frame_index"]
    ):
        return None
    prior = outputs.get("phase_identification") or {}
    if prior.get("status") not in {"identified", "outside_boundary"}:
        return None
    configured = (
        prior.get("phase")
        if prior.get("method") in {"unique_configured_phase", "Na_free_O1_default"}
        else None
    )
    scope = _classification_scope(manager, phase_references, configured_phase=configured)
    if prior.get("classification_scope") != scope:
        return None
    value = outputs.get("structure_path") or outputs.get("final_structure_path")
    if not value or not Path(value).is_file():
        return None
    digest = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    return (digest, configured) if prior.get("structure_sha256") == digest else None


def _classification_scope(manager, phase_references, *, configured_phase=None):
    references = phase_references or (
        (manager.data.get("system_config") or {}).get("phase_references") or {}
    )
    described = {}
    for name, value in sorted(references.items()):
        if isinstance(value, (str, Path)):
            path = Path(value)
            stat = path.stat() if path.is_file() else None
            described[name] = [
                str(path),
                stat.st_size if stat else None,
                stat.st_mtime_ns if stat else None,
            ]
        else:
            described[name] = str(value)
    payload = {
        "boundary": getattr(manager, "boundary", None),
        "mobile": (
            ((manager.data.get("system_config") or {}).get("species") or {}).get("mobile") or ["Na"]
        ),
        "references": described,
    }
    if configured_phase:
        payload["unique_phase_rule"] = ["current_composition_v1", configured_phase]
    if _legacy_na0_default_enabled(manager):
        payload["Na_free_O1_default"] = 1
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :16
    ]


def _configured_phase(structure, manager):
    """Use final composition rules, including the legacy Na-free O1 default."""
    boundary = getattr(manager, "boundary", None)
    if not boundary or not boundary.get("P"):
        return None
    from phase_agent.science.structures.boundary_utils import allowed_phases_at_x
    from phase_agent.science.structures.identify_branch import calculate_x

    x = calculate_x(structure)
    allowed = allowed_phases_at_x(boundary["P"], x)
    if x == "0" and _legacy_na0_default_enabled(manager):
        return "O1"
    return next(iter(allowed)) if len(allowed) == 1 else None


def _legacy_na0_default_enabled(manager):
    """Explicit composition rules and single-phase boundaries take precedence."""
    phases = (getattr(manager, "boundary", None) or {}).get("P")
    if not phases or isinstance(phases, dict):
        return False
    from phase_agent.science.structures.boundary_utils import allowed_phases

    allowed = allowed_phases(phases)
    mobile = (
        ((manager.data.get("system_config") or {}).get("species") or {}).get("mobile") or ["Na"]
    )[0]
    return mobile == "Na" and "O1" in allowed and len(allowed) > 1
