"""识别历史台账和当前批次中的重复候选，不修改原始数据。"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

from monty.json import MontyEncoder
from pymatgen.analysis.structure_matcher import SiteOrderedIStructure, StructureMatcher
from pymatgen.core import Structure

from phase_agent.science.structures.boundary_utils import (
    compact_json,
    load_structure,
    normalize_H,
    normalize_fraction,
    allowed_phases_at_x,
)
from phase_agent.science.structures.identify_branch import identify_branch_parameters


class _BatchStructureMatcher(StructureMatcher):
    """Reuse exact reduced structures only within one deduplication call."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._reduced_cache = OrderedDict()

    def _uncached_reduction(self, structure, primitive_cell, niggli):
        # Bypass pymatgen's composition-hash LRU, whose collisions make its
        # approximate site comparisons dominate large same-composition batches.
        reducer = getattr(StructureMatcher._get_reduced_istructure, "__wrapped__", None)
        if reducer is None:
            return super()._get_reduced_structure(structure, primitive_cell, niggli)
        reduced = reducer(SiteOrderedIStructure.from_sites(structure), primitive_cell, niggli)
        return Structure.from_sites(reduced)

    def _get_reduced_structure(self, structure, primitive_cell=True, niggli=True):
        try:
            encoded = json.dumps(
                structure.as_dict(), cls=MontyEncoder, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            return super()._get_reduced_structure(structure, primitive_cell, niggli)
        key = (encoded, primitive_cell, niggli)
        if key not in self._reduced_cache:
            self._reduced_cache[key] = self._uncached_reduction(structure, primitive_cell, niggli)
            if len(self._reduced_cache) > 256:
                self._reduced_cache.popitem(last=False)
        else:
            self._reduced_cache.move_to_end(key)
        return self._reduced_cache[key].copy()


def deduplicate_candidates(
    candidates: list[dict[str, Any]],
    *,
    manager: Any | None = None,
    phase_references: dict[str, Any] | None = None,
    symmetry_permutations: dict[str, list[list[int]]] | None = None,
    matcher: StructureMatcher | None = None,
    branch_only: bool = False,
) -> dict[str, Any]:
    """标记重复框架、branch 和构型，返回保留项及完整判重记录。

    ``symmetry_permutations`` 的键为 ``compact_json([P, H])``。结构匹配仅在
    相同 P、H、x、T 内执行，因此不同周期矩阵不会被合并。
    """
    permutations = symmetry_permutations or {}
    matcher = matcher or _BatchStructureMatcher(primitive_cell=False, scale=False)
    historical_frameworks, historical_branches, historical_configs = _history(manager, permutations)
    seen_frameworks = set(historical_frameworks)
    seen_branches = set(historical_branches)
    branch_representatives = {_branch_key(row, permutations): branch_id for branch_id, row in (manager.data.get("branches", {}) if manager is not None else {}).items()}
    seen_configs = defaultdict(list)
    for item in historical_configs:
        seen_configs[item["branch_key"]].append(item)
    unique: list[dict[str, Any]] = []
    records = []

    for index, original in enumerate(candidates):
        candidate = copy.copy(original)
        candidate.setdefault("candidate_id", f"batch:{index}")
        parameters = _parameters(candidate, manager, phase_references)
        candidate.update({key: parameters[key] for key in ("P", "H", "x", "T")})
        framework_key = _framework_key(candidate)
        branch_key = _branch_key(candidate, permutations)
        duplicate_of = branch_representatives.get(branch_key) if branch_only else _find_configuration_duplicate(
            candidate, branch_key, seen_configs[branch_key], matcher, permutations
        )
        status = {
            "framework_seen": framework_key in seen_frameworks,
            "branch_seen": branch_key in seen_branches,
            "configuration_duplicate": duplicate_of is not None,
            "duplicate_of": duplicate_of,
        }
        candidate["deduplication"] = status
        candidate_id = candidate.get("candidate_id", f"batch:{index}")
        records.append({"candidate_id": candidate_id, **status})
        seen_frameworks.add(framework_key)
        seen_branches.add(branch_key)
        branch_representatives.setdefault(branch_key, candidate_id)

        if duplicate_of is None:
            candidate.setdefault("generation_sources", []).append(_source(candidate))
            unique.append(candidate)
            if not branch_only:
                seen_configs[branch_key].append(
                    _configuration_record(candidate_id, candidate, branch_key)
                )
        else:
            representative = next(
                (item for item in unique if item.get("candidate_id") == duplicate_of),
                None,
            )
            if representative is not None:
                sources = representative.setdefault("generation_sources", [])
                source = _source(candidate)
                if source not in sources:
                    sources.append(source)

    return {
        "unique_candidates": unique,
        "duplicate_records": records,
        "statistics": {
            "input": len(candidates),
            "unique": len(unique),
            "configuration_duplicates": sum(item["configuration_duplicate"] for item in records),
            "frameworks_already_seen": sum(item["framework_seen"] for item in records),
            "branches_already_seen": sum(item["branch_seen"] for item in records),
        },
    }


def _parameters(candidate: dict[str, Any], manager: Any, references: Any) -> dict[str, Any]:
    if candidate.get("structure") is not None and manager is not None and references:
        if candidate.get("initialization_method") == "near_hull_parent_adjacent_na":
            return _inherited_parameters(candidate, manager, references)
        return identify_branch_parameters(
            candidate["structure"],
            manager.boundary,
            phase_references=references,
            phase_hint=candidate.get("P"),
        )
    missing = {"P", "H", "x", "T"} - candidate.keys()
    if missing:
        raise ValueError(f"候选缺少字段：{sorted(missing)}")
    return {
        "P": str(candidate["P"]).upper(),
        "H": normalize_H(candidate["H"]),
        "x": normalize_fraction(candidate["x"]),
        "T": candidate["T"],
    }


def _inherited_parameters(candidate, manager, references):
    """Validate the stored parent identity without re-identifying its relaxed lattice."""
    from fractions import Fraction
    import numpy as np
    from phase_agent.science.structures.identify_branch import calculate_x, extract_T
    from phase_agent.science.structures.inherit_near_hull_occupancy import (
        verify_parent_matches_branch,
    )

    source = candidate.get("near_hull_parent_evidence") or {}
    parent_id = candidate.get("parent_branch_id")
    parent_branch = manager.data.get("branches", {}).get(parent_id)
    if not parent_branch or source.get("branch_id") != parent_id:
        raise ValueError("near_hull_parent_identity_mismatch")
    if (
        Fraction(str(source.get("branch_x"))) != Fraction(str(parent_branch["x"]))
        or candidate["P"] != parent_branch["P"]
        or normalize_H(candidate["H"]) != normalize_H(parent_branch["H"])
        or list(candidate["T"]) != list(parent_branch["T"])
        or candidate["P"] not in allowed_phases_at_x(manager.boundary["P"], candidate["x"])
    ):
        raise ValueError("near_hull_child_branch_mismatch")
    parent = verify_parent_matches_branch(source, parent_branch, manager.boundary, references)
    child = load_structure(candidate["structure"])
    if not np.allclose(child.lattice.matrix, parent.lattice.matrix, atol=1e-7, rtol=0):
        raise ValueError("near_hull_child_lattice_mismatch")

    def sites(structure, sodium):
        return sorted(
            (site.specie.symbol, *(round(float(v) % 1, 7) for v in site.frac_coords))
            for site in structure
            if (site.specie.symbol == "Na") == sodium
        )

    if sites(parent, False) != sites(child, False):
        raise ValueError("near_hull_child_framework_mismatch")
    old_na, new_na = set(sites(parent, True)), set(sites(child, True))
    if not ((len(old_na - new_na), len(new_na - old_na)) in {(1, 0), (0, 1)}):
        raise ValueError("near_hull_child_na_change_mismatch")
    actual_t, _ = extract_T(child, manager.boundary["TM_ratio"])
    if actual_t != list(candidate["T"]) or Fraction(calculate_x(child)) != Fraction(
        str(candidate["x"])
    ):
        raise ValueError("near_hull_child_composition_mismatch")
    return {key: candidate[key] for key in ("P", "H", "x", "T")}


def _history(
    manager: Any | None, permutations: dict[str, list[list[int]]]
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    if manager is None:
        return set(), set(), []
    frameworks, branches, configs = set(), set(), []
    for branch_id, branch in manager.data.get("branches", {}).items():
        framework = _framework_key(branch)
        key = _branch_key(branch, permutations)
        frameworks.add(framework)
        branches.add(key)
        for structure_id in branch.get("structure_ids", []):
            record = manager.data.get("structures", {}).get(structure_id, {})
            configs.append(
                {
                    "id": structure_id,
                    "branch_key": key,
                    "arrangement": record.get("arrangement"),
                    "historical_record": record,
                }
            )
    return frameworks, branches, configs


def _find_configuration_duplicate(candidate, branch_key, records, matcher, permutations):
    arrangement = candidate.get("arrangement")
    if arrangement is None and "V" in candidate:
        arrangement = {"V": candidate["V"]}
    structure = candidate.get("structure")
    for record in records:
        if record["branch_key"] != branch_key:
            continue
        if arrangement is not None and record.get("arrangement") is not None:
            if _arrangement_key(arrangement, candidate, permutations) == _arrangement_key(
                record["arrangement"], candidate, permutations
            ):
                return record["id"]
        if structure is not None:
            if "historical_record" in record and "structure" not in record:
                record["structure"] = _load_historical_structure(record["historical_record"])
            if record.get("structure") is not None and matcher.fit(
                load_structure(structure), record["structure"]
            ):
                return record["id"]
    return None


def _configuration_record(identifier, candidate, branch_key):
    arrangement = candidate.get("arrangement")
    if arrangement is None and "V" in candidate:
        arrangement = {"V": candidate["V"]}
    return {
        "id": identifier,
        "branch_key": branch_key,
        "arrangement": arrangement,
        "structure": load_structure(candidate["structure"])
        if candidate.get("structure") is not None
        else None,
    }


def _framework_key(item):
    return compact_json([str(item["P"]).upper(), normalize_H(item["H"])])


def _branch_key(item, permutations):
    framework = _framework_key(item)
    T = list(item["T"])
    variants = [T]
    for permutation in permutations.get(framework, []):
        if sorted(permutation) != list(range(len(T))):
            raise ValueError(f"框架 {framework} 的 T 对称置换无效")
        variants.append([T[index] for index in permutation])
    return compact_json(
        [framework, normalize_fraction(item["x"]), min(map(compact_json, variants))]
    )


def _arrangement_key(arrangement, candidate, permutations):
    if not isinstance(arrangement, dict) or "V" not in arrangement:
        return compact_json(arrangement)
    values = list(arrangement["V"])
    variants = [values]
    for permutation in permutations.get(_framework_key(candidate), []):
        if len(permutation) == len(values):
            variants.append([values[index] for index in permutation])
    return min(map(compact_json, variants))


def _load_historical_structure(record):
    path = record.get("source_path")
    if path and Path(path).is_file():
        return load_structure(path)
    metadata = record.get("metadata") or {}
    path = metadata.get("structure_path") if isinstance(metadata, dict) else None
    return load_structure(path) if path and Path(path).is_file() else None


def _source(candidate):
    return {
        "candidate_id": candidate.get("candidate_id"),
        "strategy": candidate.get("strategy"),
        "parent_branch_id": candidate.get("parent_branch_id"),
        "near_hull_parent_evidence": candidate.get("near_hull_parent_evidence"),
        "seed": candidate.get("seed"),
        "source_path": candidate.get("source_path"),
    }
