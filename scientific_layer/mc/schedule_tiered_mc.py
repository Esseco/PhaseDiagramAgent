"""Default Relax/Hull-screened tiered MC policy (not BOHB)."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import random


def schedule_tiered_mc(candidates, state, *, policy, total_budget, seed, model_version,
                       hull_reference_version):
    current = deepcopy(state or {"segments": [], "processed_task_keys": []})
    current.setdefault("segments", []); current.setdefault("processed_task_keys", [])
    tiers = list(policy.get("tiers") or [])
    if not tiers:
        raise ValueError("mc_policy.tiers cannot be empty")
    _validate_tiers(tiers)
    max_segments = int(policy.get("max_segments_per_branch", 3))
    max_cost = float(policy.get("max_cumulative_cost_per_branch", float("inf")))
    rng = random.Random(seed)
    active = [row for row in current["segments"] if row.get("status") in {"pending", "running"}]
    if active:
        return {"status": "tasks_in_progress", "state": current, "actions": []}
    by_branch = {}
    for row in current["segments"]:
        by_branch.setdefault(row["branch_id"], []).append(row)
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
    pool = eligible + deferred[:keep]
    actions, reserved = [], 0.0
    for candidate in pool:
        branch_id = candidate["branch_id"]; previous = by_branch.get(branch_id, [])
        if len(previous) >= max_segments:
            continue
        spent = sum(float(row.get("actual_gpu_core_hours") if row.get("actual_gpu_core_hours") is not None
                          else row.get("actual_cost") if row.get("actual_cost") is not None
                          else row.get("accounted_cost", row.get("estimated_cost", 0)) or 0)
                    for row in previous)
        if spent >= max_cost:
            continue
        tier_index, reason = _next_tier(previous, candidate, tiers, policy)
        if tier_index is None:
            continue
        band = _initial_band(candidate, policy) if not previous else None
        tier = {**tiers[tier_index], **band} if band else tiers[tier_index]
        requested = int(tier["max_mc_steps"])
        if spent + requested > max_cost:
            continue
        if reserved + requested > total_budget:
            continue
        segment_index = len(previous)
        task_key = "tiered-mc:" + hashlib.sha256(
            f"{model_version}:{hull_reference_version}:{branch_id}:{segment_index}:{seed}".encode()
        ).hexdigest()[:16]
        action = {"task_key": task_key, "branch_id": branch_id, "stage": "deep_search",
                  "status": "pending", "budget": requested, "incremental_budget": requested,
                  "planned_relative_cost": requested, "max_mc_steps": requested,
                  "requested_max_mc_steps": requested,
                  "patience_steps": int(tier["patience_steps"]),
                  "min_improvement": float(tier["min_improvement"]),
                  "seed": seed + segment_index + len(actions), "tier": tier.get("name", tier_index),
                  "tier_index": tier_index, "selection_source": "relax_hull_tiered_mc",
                  "segment_reason": reason, "restart_mode": "new_segment_from_structure",
                  "model_version": model_version, "hull_reference_version": hull_reference_version}
        if not previous:
            action["input_energy_per_atom"] = candidate.get("relaxed_energy_per_atom")
        else:
            last = [row for row in previous if row.get("status") == "completed"][-1]
            action["structure_path"] = last.get("result_path") or last.get("structure_path")
            action["input_energy_per_atom"] = (last.get("outputs") or {}).get("energy_per_atom")
            if not action["structure_path"]:
                continue
        for key in ("structure_id", "relaxed_ehull",
                    "branch_energy_std_per_atom", "hull_reference_energy_per_atom"):
            if key in candidate: action[key] = candidate[key]
        if not previous and candidate.get("structure_path"):
            action["structure_path"] = candidate["structure_path"]
        actions.append(action); reserved += requested
        current["segments"].append(deepcopy(action))
    status = "budget_exhausted" if not actions and total_budget <= 0 else "scheduled" if actions else "evidence_insufficient"
    return {"status": status, "state": current, "actions": actions,
            "method": "relax_hull_tiered_mc", "requested_budget": reserved,
            "excluded_candidates": excluded_candidates}


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
                if float(gap) < float(band["max_ehull_ev_per_atom"]):
                    band_index = index
                    break
        by_band.setdefault(band_index, []).append(candidate)

    ordered = []
    for band_index in sorted(by_band):
        by_phase = {}
        for candidate in by_band[band_index]:
            phase = str(candidate.get("P") or "unknown").upper()
            x_key = _composition_key(candidate.get("x"))
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
            phase_priority[phase] = _relax_candidate_key(queue[0]) if queue else (True, float("inf"), 0, phase)

        phase_order = sorted(phase_queues, key=lambda phase: (phase_priority[phase], phase))
        while any(phase_queues.values()):
            for phase in phase_order:
                if phase_queues[phase]:
                    ordered.append(phase_queues[phase].pop(0))
    return ordered


def _relax_candidate_key(candidate):
    gap = candidate.get("relaxed_ehull")
    allocation = candidate.get("allocation_score")
    score = float(allocation) if allocation is not None else (
        float(gap) if gap is not None else float("inf"))
    return (gap is None, score, float(gap) if gap is not None else float("inf"),
            -float(candidate.get("branch_energy_std_per_atom") or 0),
            str(candidate.get("branch_id") or ""))


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
    near = candidate.get("relaxed_ehull") is not None and float(candidate["relaxed_ehull"]) <= float(
        policy.get("near_hull_retry_threshold", 0.10))
    if last.get("stop_reason") == "patience" and near:
        return 0, "near_hull_patience_retry_new_seed"
    if improvement is None or float(improvement) < float(last.get("min_improvement", tiers[0]["min_improvement"])):
        return None, "no_sustained_improvement"
    return min(int(last.get("tier_index", 0)) + 1, len(tiers) - 1), "sustained_improvement_promotion"


def _initial_band(candidate, policy):
    bands = policy.get("initial_ehull_bands")
    if not bands:
        return None
    gap = candidate.get("relaxed_ehull")
    if gap is None:
        return {"name": "unknown_reference_exploration",
                "max_mc_steps": policy["exploration_max_mc_steps"],
                "patience_steps": policy["exploration_patience_steps"],
                "min_improvement": 0.001}
    for index, band in enumerate(bands):
        if float(gap) < float(band["max_ehull_ev_per_atom"]):
            return {**band, "name": f"ehull_band_{index}"}
    return {"name": "high_ehull_exploration",
            "max_mc_steps": policy["exploration_max_mc_steps"],
            "patience_steps": policy["exploration_patience_steps"],
            "min_improvement": 0.001}


def _validate_tiers(tiers):
    previous = 0
    for tier in tiers:
        for key in ("max_mc_steps", "patience_steps", "min_improvement"):
            if tier.get(key) is None:
                raise ValueError(f"MC tier missing {key}")
        steps = int(tier["max_mc_steps"])
        if steps <= previous or int(tier["patience_steps"]) <= 0:
            raise ValueError("MC tiers must have increasing max_mc_steps and positive patience")
        previous = steps
