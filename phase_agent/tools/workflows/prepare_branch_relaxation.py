"""Prepare version-specific, budgeted Relax screening before MC allocation."""

from copy import deepcopy
import hashlib
import json


from phase_agent.science.structures.initialize_branch_structures import initialize_branch_structures
from phase_agent.persistence.ledger.register_candidate_batch import register_candidate_batch
from phase_agent.tools.budget.reserve_budget import reserve_budget
from phase_agent.analysis.phase.branch_relax_hull import (
    ENERGY_BASIS,
    build_relax_hull,
    rank_relaxed_branches,
)
from phase_agent.persistence.ledger.branch_energy_pool_ledger import (
    load_branch_energy_pools,
    save_branch_energy_pool,
)


def prepare_branch_relaxation(candidates, state, context):
    from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost

    config = context["effective_config"]
    manager = context["manager"]
    settings = config["bohb"]
    count = min(3, int(settings.get("relax_structures_per_branch", 3)))
    if count <= 0:
        raise ValueError("relax_structures_per_branch must be positive and is capped at 3")
    mlip = config.get("mlip") or {}
    version = mlip.get("version") or mlip.get("name")
    if not version and mlip.get("model_path"):
        from pathlib import Path

        version = Path(mlip["model_path"]).stem
    if not version:
        raise ValueError("confirmed MLIP version required for Relax screening")
    current = deepcopy(state)
    pending = []
    unavailable = []
    system_id = (config.get("system_config") or {}).get("system_id")
    ledger_path = config.get("branch_energy_pool_ledger_path")
    if ledger_path:
        restored = load_branch_energy_pools(
            ledger_path, system_id=system_id, mlip_version=version, energy_basis=ENERGY_BASIS
        )
        for saved_pool in restored:
            current.setdefault("branch_hull_batches", {}).setdefault(
                saved_pool["version"], saved_pool
            )
    records = []
    from phase_agent.decisions.agent.choose_debug_next_action import _verified_migrated_relax_ids

    migrated_ids = _verified_migrated_relax_ids(current, mlip, version)
    for branch in candidates:
        existing_ids = branch.get("structure_ids") or []
        screened_ids = [
            sid
            for sid in existing_ids
            if manager.data["structures"][sid].get("metadata", {}).get("initialization_method")
            == "electrostatic_top10_random3_layer_occupied"
        ]
        if not screened_ids:
            initialized = initialize_branch_structures(
                [branch],
                manager.boundary,
                context["phase_references"],
                initial_states_per_branch=count,
                seed=config.get("seed", 42),
            )
            register_candidate_batch(
                manager,
                initialized,
                structure_directory=config["structure_directory"],
                ledger_path=config.get("ledger_path"),
            )
        ids = _screen_structure_ids(
            manager, branch["branch_id"], count=count, seed=int(config.get("seed", 42))
        )
        relax_settings = deepcopy((config.get("mlip") or {}).get("relax_parameters") or {})
        settings_id = hashlib.sha256(
            json.dumps(relax_settings, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        for sid in ids:
            key = f"relax-screen:{version}:{settings_id}:{sid}"
            matches = [t for t in current.get("tasks", []) if t.get("task_key") == key]
            if not matches and sid in migrated_ids:
                matches = [
                    t
                    for t in current.get("tasks", [])
                    if t.get("structure_id") == sid
                    and t.get("model_version") == version
                    and t.get("stage") == "relax_and_feature"
                    and t.get("status") == "completed"
                ]
            if matches:
                task = matches[-1]
                out = task.get("outputs") or {}
                normal_stop = out.get("relax_stopped_normally", task.get("converged") is True)
                if task.get("status") == "completed" and normal_stop:
                    composition = out.get("composition")
                    if composition and out.get("structure_path"):
                        records.append(
                            {
                                "branch_id": branch["branch_id"],
                                "structure_id": sid,
                                "structure_path": out["structure_path"],
                                "composition": composition,
                                "energy": out.get("energy"),
                                "energy_unit": out.get("energy_unit"),
                                "relax_stopped_normally": True,
                                "converged": True,
                                "model_version": version,
                            }
                        )
                    else:
                        unavailable.append(sid)
                elif task.get("status") in {"pending", "running"}:
                    pending.append(task)
                else:
                    unavailable.append(sid)
                continue
            record = manager.data["structures"][sid]
            from pymatgen.core import Structure

            atoms = len(Structure.from_file(record["source_path"]))
            cost = estimate_stage_cost(
                "relax_and_feature", atom_count=atoms, budgets=config["budgets"]
            )["value"]
            reservation = reserve_budget(
                current,
                task_key=key,
                stage="relax_and_feature",
                amount=cost,
                limits=config["budgets"],
                config_version=context["config_version"],
                model_version=version,
            )
            if reservation["status"] != "reserved":
                unavailable.append(sid)
                continue
            current = reservation["state"]
            task = {
                "task_id": "RELAX-" + hashlib.sha256(key.encode()).hexdigest()[:12],
                "task_key": key,
                "structure_id": sid,
                "object_id": sid,
                "branch_id": branch["branch_id"],
                "stage": "relax_and_feature",
                "status": "pending",
                "model_version": version,
                "config_version": context["config_version"],
                "planned_relative_cost": cost,
                "parameters": relax_settings,
                "screening_basis": "electrostatic_top10_random3_layer_occupied",
            }
            task["generation_cycle"] = len(current.get("generation_history") or [])
            task["parent_decision_id"] = context.get("approval_record_id")
            current.setdefault("tasks", []).append(task)
            current.setdefault("pending_tasks", []).append(task)
            pending.append(task)
    if pending:
        return {
            "state": current,
            "status": "screening_pending" if pending else "screening_incomplete",
            "candidates": [],
            "unavailable": unavailable,
        }
    if not records:
        return {
            "state": current,
            "status": "screening_incomplete",
            "candidates": [],
            "unavailable": unavailable,
            "reason": "no_successful_relax_results",
        }
    frozen_version = context.get("mc_hull_reference_version")
    pool = (
        (current.get("branch_hull_batches") or {}).get(frozen_version) if frozen_version else None
    )
    if frozen_version and (not pool or pool.get("model_version") != version):
        return {
            "state": current,
            "status": "screening_incomplete",
            "candidates": [],
            "unavailable": unavailable,
            "reason": "approved_hull_reference_unavailable",
        }
    pool = pool or build_relax_hull(records, model_version=version, system_id=system_id)
    diagram = (current.get("phase_diagrams") or {}).get("mlip") or {}
    if (
        diagram.get("status") != "completed"
        or diagram.get("model_version") != version
        or not diagram.get("version")
    ):
        return {
            "state": current,
            "status": "screening_incomplete",
            "candidates": [],
            "unavailable": unavailable,
            "reason": "current_mlip_phase_diagram_unavailable",
        }
    approved_diagram = context.get("mc_phase_diagram_version")
    if approved_diagram and diagram["version"] != approved_diagram:
        return {
            "state": current,
            "status": "screening_incomplete",
            "candidates": [],
            "unavailable": unavailable,
            "reason": "approved_phase_diagram_changed",
        }
    ranked, missing = rank_relaxed_branches(
        candidates,
        pool,
        uncertainty_weight=float(settings.get("uncertainty_weight", 1.0)),
        phase_diagram=diagram,
    )
    if missing:
        return {
            "state": current,
            "status": "screening_incomplete",
            "candidates": [],
            "unavailable": unavailable,
            "unavailable_branches": sorted(set(missing)),
            "reason": "relax_structure_missing_from_current_phase_diagram",
        }
    current.setdefault("branch_hull_batches", {})[pool["version"]] = pool
    current["current_branch_hull_version"] = pool["version"]
    if ledger_path:
        save_branch_energy_pool(pool, ledger_path)
        current["branch_energy_pool_ledger_path"] = str(ledger_path)
    unavailable_structures = sorted(set(unavailable))
    unavailable_branches = sorted(set(missing))
    status = (
        "ready"
        if ranked and not (unavailable_structures or unavailable_branches)
        else ("ready_partial" if ranked else "screening_incomplete")
    )
    return {
        "state": current,
        "status": status,
        "candidates": ranked,
        "pool": pool,
        "unavailable": unavailable_structures,
        "unavailable_branches": unavailable_branches,
        "partial": bool(unavailable_structures or unavailable_branches),
    }


def _screen_structure_ids(manager, branch_id, *, count, seed):
    """Reuse saved random picks from the electrostatic top-ten pool."""
    ids = []
    for structure_id in manager.data["branches"][branch_id].get("structure_ids") or []:
        record = manager.data["structures"][structure_id]
        metadata = record.get("metadata") or {}
        if metadata.get("legal") is False or metadata.get("legality") in {"illegal", "rejected"}:
            continue
        if metadata.get("initialization_method") == "electrostatic_top10_random3_layer_occupied":
            ids.append(structure_id)
    if not ids:
        raise RuntimeError(
            f"{branch_id}: no saved random picks from the legal electrostatic top-ten pool"
        )
    first_seed = (manager.data["structures"][ids[0]].get("metadata") or {}).get(
        "initialization_seed"
    )
    same_batch = [
        sid
        for sid in ids
        if (manager.data["structures"][sid].get("metadata") or {}).get("initialization_seed")
        == first_seed
    ]
    return same_batch[:count]
