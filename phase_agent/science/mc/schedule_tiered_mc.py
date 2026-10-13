"""Default Relax/Hull-screened tiered MC policy (not BOHB)."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import random


def schedule_tiered_mc(
    candidates,
    state,
    *,
    policy,
    total_budget,
    seed,
    model_version,
    hull_reference_version,
    phase_diagram_version=None,
    scope_by_model=False,
):
    current = deepcopy(state or {"segments": [], "processed_task_keys": []})
    current.setdefault("segments", [])
    current.setdefault("processed_task_keys", [])
    tiers = list(policy.get("tiers") or [])
    if not tiers:
        raise ValueError("mc_policy.tiers cannot be empty")
    _validate_tiers(tiers)
    max_segments = int(policy.get("max_segments_per_branch", 3))
    raw_cost = policy.get("max_cumulative_cost_per_branch")
    max_cost = float(raw_cost) if raw_cost is not None else float("inf")
    cost_reference = max_cost
    from phase_agent.tools.budget.check_budget import budget_is_hard

    hard_budget = budget_is_hard({"enforcement": policy.get("budget_enforcement", "advisory")})
    if not hard_budget:
        max_cost = float("inf")
        total_budget = float("inf")
    rng = random.Random(seed)
    scoped = [
        row
        for row in current["segments"]
        if not scope_by_model or row.get("model_version") == model_version
    ]
    active = [row for row in scoped if row.get("status") in {"pending", "running"}]
    if active:
        return {"status": "tasks_in_progress", "state": current, "actions": []}
    by_branch = {}
    for row in scoped:
        by_branch.setdefault(row["branch_id"], []).append(row)
    refreshed_ground_branches = set()
    if policy.get("successive_halving_enabled"):
        for candidate in candidates:
            previous = by_branch.get(candidate["branch_id"], [])
            gap = candidate.get("relaxed_ehull")
            if (
                candidate.get("ehull_source") == "phase_diagram"
                and candidate.get("relaxed_ehull_unit") == "eV/atom"
                and gap is not None
                and math.isclose(float(gap), 0.0, abs_tol=1e-8)
                and any(row.get("model_version") not in (None, model_version) for row in previous)
            ):
                by_branch[candidate["branch_id"]] = [
                    row for row in previous if row.get("model_version") in (None, model_version)
                ]
                refreshed_ground_branches.add(candidate["branch_id"])
    eligible_candidates, excluded_candidates = [], []
    for candidate in candidates:
        reason = _mc_candidate_exclusion_reason(candidate)
        if reason:
            excluded_candidates.append({"branch_id": candidate.get("branch_id"), "reason": reason})
        else:
            eligible_candidates.append(candidate)
    high = float(policy.get("high_ehull_defer_threshold", 0.30))
    exploration = float(policy.get("random_exploration_fraction", 0.10))
    eligible, deferred = [], []
    for candidate in eligible_candidates:
        ehull = candidate.get("relaxed_ehull")
        (deferred if ehull is not None and float(ehull) > high else eligible).append(candidate)
    rng.shuffle(deferred)
    keep = max(0, round(len(eligible_candidates) * exploration))
    # Keep lower-hull bands first, while rotating phase/x inside a band so an
    # equal-hull cluster cannot consume the whole first allocation.
    eligible = _coverage_balanced_order(eligible, policy)
    pool = (
        _coverage_balanced_order(eligible_candidates, policy)
        if policy.get("_fill_only")
        else eligible + deferred[:keep]
    )
    hb = policy.get("successive_halving_enabled", False)
    if hb and not policy.get("second_segment_enabled", True):
        eligible_candidates = [
            row for row in eligible_candidates if not by_branch.get(row["branch_id"])
        ]
    hb_selection = {}
    if hb:
        pool, hb_selection = _hb_candidate_pool(eligible_candidates, by_branch, policy, seed=seed)
        if hb_selection.get("below_minimum"):
            return {
                "status": "continuation_pool_below_minimum",
                "state": deepcopy(state),
                "actions": [],
                "required_tasks": int(policy.get("minimum_continuation_tasks", 10)),
                "eligible_task_count": len(pool),
                "selection": hb_selection,
            }
    actions, reserved = [], 0.0
    selected_branches = set()
    for candidate in pool:
        branch_id = candidate["branch_id"]
        if branch_id in selected_branches:
            continue
        previous = by_branch.get(branch_id, [])
        # A refreshed hull ground state gets a current-model search budget.
        # Near-hull non-ground branches retain the existing continuation rules.
        refreshed_ground = (
            candidate.get("ehull_source") == "phase_diagram"
            and candidate.get("relaxed_ehull_unit") == "eV/atom"
            and candidate.get("relaxed_ehull") is not None
            and math.isclose(float(candidate["relaxed_ehull"]), 0.0, abs_tol=1e-8)
            and any(row.get("model_version") not in (None, model_version) for row in previous)
        )
        if refreshed_ground:
            previous = [
                row for row in previous if row.get("model_version") in (None, model_version)
            ]
        if len(previous) >= max_segments:
            continue
        spent = sum(
            float(
                row.get("actual_gpu_core_hours")
                if row.get("actual_gpu_core_hours") is not None
                else row.get("actual_cost")
                if row.get("actual_cost") is not None
                else row.get("accounted_cost", row.get("estimated_cost", 0)) or 0
            )
            for row in previous
        )
        if spent >= max_cost:
            continue
        if hb:
            tier_index = len(previous)
            reason = (
                "protected_hull_candidate"
                if _hb_protected(candidate, policy)
                else ("hb_composite_priority" if previous else "hb_initial_coverage")
            )
        elif policy.get("_fill_only"):
            if not previous or any(row.get("status") != "completed" for row in previous):
                continue
            tier_index, reason = 0, "continuation_batch_fill_exploration"
        else:
            tier_index, reason = _next_tier(previous, candidate, tiers, policy)
        if tier_index is None:
            continue
        if (refreshed_ground or branch_id in refreshed_ground_branches) and not previous:
            reason = "refreshed_hull_ground_state"
        band = _initial_band(candidate, policy) if not previous else None
        tier = tiers[min(tier_index, len(tiers) - 1)] if hb else tiers[tier_index]
        if band:
            tier = {**tier, **band}
        if hb:
            base = _initial_band(candidate, policy)
            base_steps = int(base["max_mc_steps"])
            base_patience = int(base["patience_steps"])
            resource_tier = (
                _hb_resource_tier(candidate, previous, policy) if previous else "initial"
            )
            tier = {
                "name": resource_tier,
                "min_improvement": 0.001,
                **(policy["continuation_resource_tiers"][resource_tier] if previous else base),
            }
        requested = int(tier["max_mc_steps"])
        if spent + requested > max_cost:
            continue
        if reserved + requested > total_budget:
            continue
        segment_index = len(previous)
        identity = f"{model_version}:{hull_reference_version}:{branch_id}:{segment_index}:{seed}"
        if phase_diagram_version is not None:
            identity += f":{phase_diagram_version}"
        task_key = "tiered-mc:" + hashlib.sha256(identity.encode()).hexdigest()[:16]
        action = {
            "task_key": task_key,
            "branch_id": branch_id,
            "segment_index": segment_index,
            "stage": "deep_search",
            "status": "pending",
            "budget": requested,
            "incremental_budget": requested,
            "planned_relative_cost": requested,
            "max_mc_steps": requested,
            "requested_max_mc_steps": requested,
            "patience_steps": int(tier["patience_steps"]),
            "min_improvement": float(tier["min_improvement"]),
            "seed": seed + segment_index + len(actions),
            "tier": tier.get("name", tier_index),
            "tier_index": tier_index,
            "selection_source": "relax_hull_tiered_mc",
            "segment_reason": reason,
            "budget_warnings": ["branch_cumulative_cost_reference_exceeded"]
            if not hard_budget and spent >= cost_reference
            else [],
            "restart_mode": "new_segment_from_structure",
            "model_version": model_version,
            "hull_reference_version": hull_reference_version,
            "phase_diagram_version": phase_diagram_version,
            "ehull_source": candidate.get("ehull_source"),
            "relaxed_ehull_unit": "eV/atom" if candidate.get("relaxed_ehull") is not None else None,
        }
        if hb:
            action.update(
                hb_initial_steps=base_steps,
                hb_initial_patience=base_patience,
                stopping_threshold_source="library_default",
                min_improvement=None,
                selection_source="ehull_improvement_feedback_mc",
                improvement_class=_hb_improvement_class(previous, policy),
                composite_priority=_hb_priority(candidate, previous, policy),
                protected_candidate=_hb_protected(candidate, policy),
                resource_tier=resource_tier,
                selection_group=candidate.get("_mc_selection_group", "initial"),
                selection_seed=seed,
            )
        if not previous:
            action["input_energy_per_atom"] = candidate.get("relaxed_energy_per_atom")
        else:
            last = [row for row in previous if row.get("status") == "completed"][-1]
            action["structure_path"] = last.get("result_path") or last.get("structure_path")
            action["input_energy_per_atom"] = (last.get("outputs") or {}).get("energy_per_atom")
            if not action["structure_path"]:
                continue
        for key in (
            "structure_id",
            "relaxed_ehull",
            "branch_energy_std_per_atom",
            "hull_reference_energy_per_atom",
            "first_round_energy_improvement_ev_per_atom",
            "first_round_actual_mc_steps",
            "first_round_stop_reason",
        ):
            if key in candidate:
                action[key] = candidate[key]
        if not previous and candidate.get("structure_path"):
            action["structure_path"] = candidate["structure_path"]
        actions.append(action)
        selected_branches.add(branch_id)
        reserved += requested
        current["segments"].append(deepcopy(action))
    minimum = int(policy.get("minimum_continuation_tasks", 10))
    continuation = any(row.get("segment_index", 0) > 0 for row in actions)
    if minimum and continuation and len(actions) < minimum and not policy.get("_fill_only"):
        improved = any(
            row.get("segment_reason") == "sustained_improvement_promotion" for row in actions
        )
        if improved and not hb:
            used_branches = {row["branch_id"] for row in actions}
            fill = schedule_tiered_mc(
                [row for row in eligible_candidates if row["branch_id"] not in used_branches],
                state,
                policy={**policy, "_fill_only": True, "minimum_continuation_tasks": 0},
                total_budget=total_budget - reserved,
                seed=seed,
                model_version=model_version,
                hull_reference_version=hull_reference_version,
                phase_diagram_version=phase_diagram_version,
                scope_by_model=scope_by_model,
            )
            extra = fill["actions"][: minimum - len(actions)]
            actions.extend(extra)
            current["segments"].extend(deepcopy(extra))
            reserved += sum(row["max_mc_steps"] for row in extra)
        if len(actions) < minimum:
            return {
                "status": "continuation_pool_below_minimum",
                "state": deepcopy(state),
                "actions": [],
                "required_tasks": minimum,
                "eligible_task_count": len(actions),
                "reason": "Expand and Relax more valid branches before MC; no duplicate tasks were padded.",
            }
    status = (
        "budget_exhausted"
        if not actions and total_budget <= 0
        else "scheduled"
        if actions
        else "evidence_insufficient"
    )
    return {
        "status": status,
        "state": current,
        "actions": actions,
        "method": "relax_hull_tiered_mc",
        "requested_budget": reserved,
        "excluded_candidates": excluded_candidates,
        "selection": hb_selection,
    }


def _hb_protected(candidate, policy):
    gap = candidate.get("relaxed_ehull")
    return bool(candidate.get("is_ground_state") or candidate.get("near_degenerate")) or (
        gap is not None and float(gap) <= float(policy.get("protected_ehull_ev_per_atom", 0.003))
    )


def _hb_improvement_class(previous, policy):
    value = previous[-1].get("energy_improvement") if previous else None
    if value is None:
        return "unknown"
    if float(value) > float(policy["improvement_strong_ev_per_atom"]):
        return "strong"
    if float(value) >= float(policy["improvement_moderate_ev_per_atom"]):
        return "moderate"
    if float(value) >= float(policy["improvement_small_ev_per_atom"]):
        return "small"
    return "weak"


def _hb_resource_tier(candidate, previous, policy):
    gap = candidate.get("relaxed_ehull")
    band = (
        3
        if gap is None
        else next((i for i, limit in enumerate((0.010, 0.020, 0.040)) if float(gap) <= limit), 3)
    )
    column = {"strong": 0, "moderate": 1, "small": 2, "weak": 3, "unknown": 3}[
        _hb_improvement_class(previous, policy)
    ]
    matrix = (
        ("high", "high", "medium", "low"),
        ("high", "medium", "low", "exploration"),
        ("medium", "low", "exploration", None),
        ("low", "exploration", None, None),
    )
    tier = matrix[band][column]
    if _hb_protected(candidate, policy) and tier != "high":
        return "medium"
    return tier


def _hb_priority(candidate, previous, policy):
    return {"high": 1, "medium": 2, "low": 3, "exploration": 4, None: 5}[
        _hb_resource_tier(candidate, previous, policy)
    ]


def _hb_candidate_pool(candidates, by_branch, policy, *, seed=0):
    """Frozen wave population determines halving; caller target subsets cannot change it."""
    latest = max((len(rows) - 1 for rows in by_branch.values()), default=-1)
    candidates = list({row["branch_id"]: row for row in candidates}.values())
    new = [row for row in candidates if not by_branch.get(row["branch_id"])]
    if new:
        candidates = new
        protected = [row for row in candidates if _hb_protected(row, policy)]
        ordinary = [row for row in candidates if not _hb_protected(row, policy)]
        low = [
            row
            for row in ordinary
            if row.get("relaxed_ehull") is not None and float(row["relaxed_ehull"]) <= 0.040
        ]
        high = [row for row in ordinary if row not in low]
        fraction = float(policy.get("random_exploration_fraction", 0.10))
        requested_count = policy.get("agent_branch_count")
        total = max(len(protected), int(requested_count)) if requested_count else len(candidates)
        explore = min(len(high), math.ceil(total * fraction))
        if high and not protected and not low:
            explore = max(1, math.ceil(len(high) * fraction))
        low_count = max(0, total - len(protected) - explore)
        pool = (
            _coverage_balanced_order(protected, policy)
            + _coverage_balanced_order(low, {**policy, "initial_ehull_bands": []})[:low_count]
        )
        pool += _coverage_balanced_order(high, policy)[:explore]
        return pool, {"round_kind": "initial", "exploration_count": min(explore, len(high))}
    wave_ids = {key for key, rows in by_branch.items() if len(rows) - 1 == latest}
    available = [
        row
        for row in candidates
        if row["branch_id"] in wave_ids
        and len(by_branch[row["branch_id"]]) < 3
        and all(item.get("status") == "completed" for item in by_branch[row["branch_id"]])
    ]
    protected = [row for row in available if _hb_protected(row, policy)]
    ordinary = [row for row in available if not _hb_protected(row, policy)]
    main, explore = [], []
    for row in ordinary:
        tier = _hb_resource_tier(row, by_branch[row["branch_id"]], policy)
        if tier in {"high", "medium"}:
            main.append(row)
        elif tier in {"low", "exploration"}:
            explore.append(row)
    # Weak feedback alone cannot force another batch; exploration may supplement useful main work.
    if (
        not protected
        and not main
        and not any(
            _hb_improvement_class(by_branch[row["branch_id"]], policy) not in {"weak", "unknown"}
            or row.get("coverage_gap")
            for row in explore
        )
    ):
        explore = []
    ranked = []
    for resource in ("high", "medium"):
        group = [
            row
            for row in main
            if _hb_resource_tier(row, by_branch[row["branch_id"]], policy) == resource
        ]
        group = [
            {
                key: value
                for key, value in row.items()
                if key not in {"second_round_priority", "allocation_score"}
            }
            for row in group
        ]
        for row in group:
            row["_hb_rank_improvement"] = by_branch[row["branch_id"]][-1].get("energy_improvement")
        ranked.extend(_coverage_balanced_order(group, {**policy, "initial_ehull_bands": []}))
    minimum = int(policy.get("minimum_continuation_tasks", 10))
    reference = max(minimum, math.ceil(len(wave_ids) / 2))
    fraction = float(policy.get("continuation_exploration_fraction", 0.10))
    high = [
        row
        for row in ranked
        if _hb_resource_tier(row, by_branch[row["branch_id"]], policy) == "high"
    ]
    medium = [
        row
        for row in ranked
        if _hb_resource_tier(row, by_branch[row["branch_id"]], policy) == "medium"
    ]
    mandatory_count = len(protected) + len(high)  # ranked excludes protected: union, no duplicates.
    count = max(reference, mandatory_count, int(policy.get("agent_branch_count") or 0))
    if mandatory_count and explore:
        count = max(count, math.ceil(mandatory_count / (1 - fraction)))
    # Avoid rounding exploration against an unattainable nominal population.
    count = min(count, len(protected) + len(main) + len(explore))
    exploration_target = min(len(explore), math.ceil(count * fraction))
    pool = [
        {**row, "_mc_selection_group": "protected"}
        for row in _coverage_balanced_order(protected, policy)
    ]
    pool += [
        {**row, "_mc_selection_group": "main"}
        for row in high + medium[: max(0, count - mandatory_count - exploration_target)]
    ]
    sampled = _stratified_exploration_order(explore, by_branch, policy, seed)
    needed = max(0, count - len(pool))
    pool += [
        {
            **row,
            "_mc_selection_group": "exploration" if i < exploration_target else "exploration_fill",
        }
        for i, row in enumerate(sampled[:needed])
    ]
    return pool, {
        "round_kind": "continuation",
        "previous_wave_count": len(wave_ids),
        "reference_count": reference,
        "protected_count": len(protected),
        "high_required_count": sum(
            _hb_resource_tier(row, by_branch[row["branch_id"]], policy) == "high"
            for row in protected
        )
        + len(high),
        "mandatory_count": mandatory_count,
        "main_eligible_count": len(main),
        "exploration_eligible_count": len(explore),
        "exploration_target": exploration_target,
        "actual_exploration_count": sum(
            row["_mc_selection_group"].startswith("exploration") for row in pool
        ),
        "exploration_fraction_reference": fraction,
        "selection_seed": seed,
        "below_minimum": bool(pool) and len(pool) < minimum,
        "exploration_fill_count": sum(
            row["_mc_selection_group"] == "exploration_fill" for row in pool
        ),
        "recommendation": "Consider Branch/Relax or DFT when useful continuation candidates are insufficient.",
    }


def _stratified_exploration_order(candidates, by_branch, policy, seed):
    """Seeded sampling rotates resource tier, phase, composition and structure family."""
    rng = random.Random(seed)
    tiers = {}
    for row in sorted(candidates, key=lambda item: item["branch_id"]):
        tier = _hb_resource_tier(row, by_branch[row["branch_id"]], policy)
        key = json.dumps(
            {key: row.get(key) for key in ("P", "x", "H", "T")}, sort_keys=True, default=str
        )
        tiers.setdefault(tier, {}).setdefault(key, []).append(row)
    queues = {}
    for tier in sorted(tiers):
        groups = tiers[tier]
        keys = sorted(groups)
        rng.shuffle(keys)
        for group in groups.values():
            rng.shuffle(group)
        queues[tier] = []
        while any(groups.values()):
            for key in keys:
                if groups[key]:
                    queues[tier].append(groups[key].pop())
    tier_order = sorted(queues)
    rng.shuffle(tier_order)
    result = []
    while any(queues.values()):
        for tier in tier_order:
            if queues[tier]:
                result.append(queues[tier].pop(0))
    return result


def _mc_candidate_exclusion_reason(candidate):
    """Na/V Monte Carlo is undefined at the empty-ion endpoint (x=0)."""
    raw_x = candidate.get("x")
    if raw_x is None:
        return None  # Preserve compatibility for callers with externally scoped candidates.
    try:
        x = Fraction(str(raw_x))
    except (ValueError, ZeroDivisionError):
        return "invalid_branch_composition"
    if x <= 0:
        return "no_mobile_ion_sites"
    if x > 1:
        return "branch_composition_out_of_range"
    return None


def _coverage_balanced_order(candidates, policy):
    bands = list(policy.get("initial_ehull_bands") or [])
    by_band = {}
    for candidate in candidates:
        gap = candidate.get("relaxed_ehull")
        band_index = len(bands)
        if gap is not None:
            for index, band in enumerate(bands):
                if float(gap) <= float(band["max_ehull_ev_per_atom"]):
                    band_index = index
                    break
        by_band.setdefault(band_index, []).append(candidate)

    ordered = []
    for band_index in sorted(by_band):
        by_phase = {}
        for candidate in by_band[band_index]:
            phase = str(candidate.get("P") or "unknown").upper()
            x_key = _composition_key(candidate.get("x"))
            if policy.get("coverage_structure"):
                x_key += "|" + json.dumps(
                    {key: candidate.get(key) for key in ("H", "T")}, sort_keys=True, default=str
                )
            by_phase.setdefault(phase, {}).setdefault(x_key, []).append(candidate)

        phase_queues = {}
        phase_priority = {}
        for phase, by_x in by_phase.items():
            for rows in by_x.values():
                rows.sort(key=_relax_candidate_key)
            x_order = sorted(by_x, key=lambda key: _relax_candidate_key(by_x[key][0]))
            queue = []
            while any(by_x[key] for key in x_order):
                for key in x_order:
                    if by_x[key]:
                        queue.append(by_x[key].pop(0))
            phase_queues[phase] = queue
            phase_priority[phase] = (
                _relax_candidate_key(queue[0]) if queue else (True, float("inf"), 0, phase)
            )

        phase_order = sorted(phase_queues, key=lambda phase: (phase_priority[phase], phase))
        while any(phase_queues.values()):
            for phase in phase_order:
                if phase_queues[phase]:
                    ordered.append(phase_queues[phase].pop(0))
    return ordered


def _relax_candidate_key(candidate):
    if candidate.get("second_round_priority") is not None:
        return (
            False,
            -float(candidate["second_round_priority"]),
            float(candidate.get("relaxed_ehull") or 0),
            str(candidate.get("branch_id") or ""),
        )
    gap = candidate.get("relaxed_ehull")
    allocation = candidate.get("allocation_score")
    score = (
        float(allocation)
        if allocation is not None
        else (float(gap) if gap is not None else float("inf"))
    )
    return (
        gap is None,
        score,
        float(gap) if gap is not None else float("inf"),
        -float(candidate.get("_hb_rank_improvement") or 0),
        -float(candidate.get("branch_energy_std_per_atom") or 0),
        str(candidate.get("branch_id") or ""),
    )


def _composition_key(value):
    if value is None:
        return "unknown_x"
    try:
        return str(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return str(value)


def _next_tier(previous, candidate, tiers, policy):
    completed = [row for row in previous if row.get("status") == "completed"]
    if not completed:
        return 0, "initial_small_tier"
    if not policy.get("second_segment_enabled", True):
        return None, "second_segment_waits_for_confirmed_policy"
    last = completed[-1]
    improvement = last.get("energy_improvement")
    near = candidate.get("relaxed_ehull") is not None and float(
        candidate["relaxed_ehull"]
    ) <= float(policy.get("near_hull_retry_threshold", 0.10))
    if last.get("stop_reason") == "patience" and near:
        return 0, "near_hull_patience_retry_new_seed"
    if improvement is None or float(improvement) < float(
        last.get("min_improvement", tiers[0]["min_improvement"])
    ):
        return None, "no_sustained_improvement"
    return min(
        int(last.get("tier_index", 0)) + 1, len(tiers) - 1
    ), "sustained_improvement_promotion"


def _initial_band(candidate, policy):
    bands = policy.get("initial_ehull_bands")
    if not bands:
        return None
    gap = candidate.get("relaxed_ehull")
    if gap is None:
        return {
            "name": "unknown_reference_exploration",
            "max_mc_steps": policy["exploration_max_mc_steps"],
            "patience_steps": policy["exploration_patience_steps"],
            "min_improvement": 0.001,
        }
    for index, band in enumerate(bands):
        if float(gap) <= float(band["max_ehull_ev_per_atom"]):
            return {**band, "name": f"ehull_band_{index}"}
    return {
        "name": "high_ehull_exploration",
        "max_mc_steps": policy["exploration_max_mc_steps"],
        "patience_steps": policy["exploration_patience_steps"],
        "min_improvement": 0.001,
    }


def _validate_tiers(tiers):
    for tier in tiers:
        for key in ("max_mc_steps", "patience_steps", "min_improvement"):
            if tier.get(key) is None:
                raise ValueError(f"MC tier missing {key}")
        steps = int(tier["max_mc_steps"])
        if steps <= 0 or int(tier["patience_steps"]) <= 0:
            raise ValueError("MC tiers must have positive max_mc_steps and patience")
