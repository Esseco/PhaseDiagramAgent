"""Build one fixed-budget, fixed-model MLIP+MC target for a branch."""

from experiments.branch_surrogate.read_structure_metadata import read_structure_metadata


def build_mc_target(branch: dict, structures: list[dict], *, config: dict, hull_reference: dict) -> dict:
    budget = config.get("target_mc_budget")
    model = config.get("mlip_version")
    hull_version = config.get("hull_reference_version")
    missing = [name for name, value in (("target_mc_budget", budget), ("mlip_version", model), ("hull_reference_version", hull_version)) if value is None]
    group = _composition_group(branch)
    reference = (hull_reference.get("groups") or {}).get(group)
    if hull_reference.get("version") != hull_version:
        missing.append("matching_hull_reference_version")
    if reference is None:
        missing.append(f"hull_reference.groups[{group}]")
    candidates = []
    rejected = []
    for structure in structures:
        attempts = ((structure.get("metadata") or {}).get("calculation_attempts") or [])
        attempts_by_task = {item.get("task_id"): item for item in attempts if item.get("task_id")}
        results = (structure.get("stage_history") or {}).get("deep_search", [])
        for result in results:
            record = _normalize_result(result, structure, attempts, attempts_by_task, budget, model)
            (candidates if record["eligible"] else rejected).append(record)
    valid = [item for item in candidates if item.get("energy_per_atom") is not None]
    best = min(valid, key=lambda item: item["energy_per_atom"]) if valid else None
    if not valid:
        missing.append("eligible_fixed_budget_mc_energy_per_atom")
    distance = None if best is None or reference is None else best["energy_per_atom"] - float(reference)
    return {
        "status": "completed" if not missing else "unknown",
        "task": "mlip_mc_fixed_budget_hull_distance",
        "composition_group": group,
        "target_mc_budget": budget,
        "mlip_version": model,
        "hull_reference_version": hull_version,
        "reference_energy_per_atom": reference,
        "minimum_energy_per_atom": None if best is None else best["energy_per_atom"],
        "distance_to_fixed_hull": distance,
        "best_result_id": None if best is None else best["result_id"],
        "raw_energy_records": candidates,
        "rejected_records": rejected,
        "actual_cost": sum(item["actual_cost"] for item in candidates if isinstance(item.get("actual_cost"), (int, float))),
        "missing": sorted(set(missing)),
    }


def _normalize_result(result, structure, attempt_list, attempts, budget, model):
    metadata = result.get("metadata") or {}
    attempt = attempts.get(metadata.get("task_id")) or {}
    if not attempt:
        compatible = [item for item in attempt_list if item.get("stage") == "deep_search" and item.get("status") == "completed" and item.get("model_version") == (result.get("mlip_version") or model)]
        if len(compatible) == 1:
            attempt = compatible[0]
    actual_budget = metadata.get("mc_budget", metadata.get("budget", attempt.get("budget")))
    if isinstance(actual_budget, dict):
        actual_budget = actual_budget.get("mc_budget", actual_budget.get("resource"))
    actual_model = result.get("mlip_version") or attempt.get("model_version")
    energy = result.get("mlip_energy")
    unit = result.get("energy_unit")
    atom_count = metadata.get("atom_count")
    if atom_count is None:
        atom_count = read_structure_metadata(structure.get("source_path")).get("atom_count")
    energy_per_atom = metadata.get("energy_per_atom")
    if energy_per_atom is None and energy is not None and unit == "eV" and atom_count:
        energy_per_atom = float(energy) / int(atom_count)
    reasons = []
    if result.get("converged") is not True:
        reasons.append("not_converged")
    if actual_budget != budget:
        reasons.append("budget_mismatch_or_unknown")
    if actual_model != model:
        reasons.append("model_version_mismatch_or_unknown")
    if unit != "eV":
        reasons.append("energy_unit_mismatch_or_unknown")
    return {
        "result_id": result.get("result_id"), "structure_id": structure.get("structure_id"),
        "energy": energy, "energy_unit": unit, "atom_count": atom_count,
        "energy_per_atom": energy_per_atom, "mc_budget": actual_budget,
        "mlip_version": actual_model, "actual_cost": metadata.get("actual_cost", attempt.get("actual_cost")),
        "eligible": not reasons and energy_per_atom is not None, "reasons": reasons,
    }


def _composition_group(branch):
    value = branch.get("composition")
    if value is None:
        value = {"x": branch.get("x")}
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
