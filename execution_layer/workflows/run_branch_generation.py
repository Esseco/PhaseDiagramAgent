"""候选 branch 生成、选择、构型初始化和登记的简洁入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scientific_layer.structures.propose_branches import propose_branches
from scientific_layer.structures.deduplicate_candidates import deduplicate_candidates
from execution_layer.budget.estimate_candidate_cost import estimate_candidate_cost
from scientific_layer.structures.initialize_branch_structures import initialize_branch_structures
from data_layer.ledger.register_candidate_batch import register_candidate_batch
from scientific_layer.structures.select_candidates import select_candidates


def run_branch_generation(
    manager: Any,
    phase_references: dict[str, Any],
    *,
    structure_directory: str | Path,
    quotas: dict[str, int],
    batch_size: int,
    initial_states_per_branch: int,
    seed: int,
    cost_config: dict[str, float] | None = None,
    cost_budget: float | None = None,
    selection_config: dict[str, Any] | None = None,
    parent_branch_ids: list[str] | None = None,
    site_mappings: dict[tuple[str, str], list[int]] | None = None,
    symmetry_permutations: dict[str, list[list[int]]] | None = None,
    ledger_path: str | Path | None = None,
    generation_registry: Any | None = None,
    system_config: dict[str, Any] | None = None,
    strategy_options: dict[str, Any] | None = None,
    framework_enumerator: Any | None = None,
) -> dict[str, Any]:
    """调度完整流程；不运行 MLIP 或 DFT。"""
    proposed = propose_branches(
        manager,
        phase_references,
        quotas=quotas,
        seed=seed,
        parent_branch_ids=parent_branch_ids,
        site_mappings=site_mappings,
        register=False,
        registry=generation_registry,
        system_config=system_config,
        strategy_options=strategy_options,
        framework_enumerator=framework_enumerator,
    )
    branch_dedup = deduplicate_candidates(
        proposed,
        manager=manager,
        phase_references=phase_references,
        symmetry_permutations=symmetry_permutations,
    )
    branches = []
    for candidate in branch_dedup["unique_candidates"]:
        candidate["initial_state_count"] = initial_states_per_branch
        branches.append(candidate)
    costed = estimate_candidate_cost(branches, config=cost_config)
    options = dict(selection_config or {})
    selected = select_candidates(
        costed,
        manager=manager,
        batch_size=batch_size,
        cost_budget=cost_budget,
        seed=seed,
        **options,
    )
    initialized = initialize_branch_structures(
        selected["selected_candidates"],
        manager.boundary,
        phase_references,
        initial_states_per_branch=initial_states_per_branch,
        seed=seed + 50_000,
    )
    structure_dedup = deduplicate_candidates(
        initialized,
        manager=manager,
        phase_references=phase_references,
        symmetry_permutations=symmetry_permutations,
    )
    registered = register_candidate_batch(
        manager,
        structure_dedup["unique_candidates"],
        structure_directory=structure_directory,
        ledger_path=ledger_path,
    )
    return {
        "proposed_branches": proposed,
        "branch_deduplication": branch_dedup,
        "selection": selected,
        "initialized_structures": initialized,
        "structure_deduplication": structure_dedup,
        "registered": registered,
        "coverage": _coverage(manager),
    }


def _coverage(manager):
    groups = {}
    for branch in manager.data["branches"].values():
        key = (branch["P"], str(branch["H"]), branch["x"])
        bucket = groups.setdefault(key, {"branches": 0, "structures": 0})
        bucket["branches"] += 1
        bucket["structures"] += len(branch.get("structure_ids", []))
    return [
        {"P": key[0], "H": key[1], "x": key[2], **value}
        for key, value in sorted(groups.items())
    ]
