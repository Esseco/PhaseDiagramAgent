"""用有限池密度比和随机探索选择新 branch。"""

import math
import random


def select_bohb_candidates(candidates: list[dict], observations: list[dict], *, count: int, seed: int, config: dict) -> dict:
    evaluated = {item["branch_id"] for item in observations if item.get("status") == "completed"}
    pool = [item for item in candidates if item.get("branch_id") not in evaluated]
    rng = random.Random(seed)
    completed = [item for item in observations if item.get("status") == "completed" and item.get("loss") is not None]
    search_scope = config.get("agent_search_scope") or {}
    focus_regions = set(search_scope.get("focus_regions") or [])
    recommended = [item for item in pool if _in_focus(item, focus_regions)]
    outside = [item for item in pool if item not in recommended]
    exploration_fraction = float(search_scope.get("exploration_fraction", 0.0))
    exploration_count = min(len(outside), math.ceil(count * exploration_fraction)) if focus_regions and count else 0
    rng.shuffle(outside)
    selected_global = outside[:exploration_count]
    primary_count = min(len(recommended), count - len(selected_global))
    selected_primary, mode, scored = _select_by_bohb(
        recommended, completed, count=primary_count, rng=rng, config=config
    )
    selected = selected_primary + selected_global
    if len(selected) < count and not focus_regions:
        selected_ids = {item["branch_id"] for item in selected}
        fill_pool = [item for item in pool if item["branch_id"] not in selected_ids]
        fill, fill_mode, fill_scores = _select_by_bohb(
            fill_pool, completed, count=count - len(selected), rng=rng, config=config
        )
        selected.extend(fill)
        scored.extend(fill_scores)
        if mode == "empty":
            mode = fill_mode
    global_ids = {item["branch_id"] for item in selected_global}
    return {
        "selected": selected,
        "selection_sources": {
            item["branch_id"]: "global_exploration" if item["branch_id"] in global_ids else "agent_recommended_scope"
            for item in selected
        },
        "global_exploration_ids": sorted(global_ids),
        "focus_regions": sorted(focus_regions),
        "model_mode": mode,
        "scores": {item[1]["branch_id"]: item[0] for item in scored},
    }


def _select_by_bohb(pool, completed, *, count, rng, config):
    if not count or not pool:
        return [], "empty", []
    pool = list(pool)
    rng.shuffle(pool)
    random_count = min(len(pool), max(1, round(count * float(config.get("random_fraction", 0.25)))))
    selected_random = pool[:random_count]
    remaining = pool[random_count:]
    scored = []
    if config.get('selection_policy') == 'relax_hull_uncertainty':
        scored = [(float(c['allocation_score']), c) for c in remaining
                  if c.get('allocation_score') is not None]
        scored.sort(key=lambda row: (row[0], row[1]['branch_id']))
        chosen = selected_random + [row[1] for row in scored[:max(0, count-len(selected_random))]]
        if len(chosen) < count:
            chosen_ids = {item['branch_id'] for item in chosen}
            unknown = [item for item in remaining if item['branch_id'] not in chosen_ids
                       and item.get('allocation_score') is None]
            rng.shuffle(unknown)
            chosen.extend(unknown[:count-len(chosen)])
        return chosen, 'relax_hull_uncertainty', scored
    if len(completed) >= int(config.get("minimum_model_observations", 6)):
        ordered = sorted(completed, key=lambda item: item["loss"])
        split = max(1, round(len(ordered) * float(config.get("good_fraction", 0.25))))
        good, bad = ordered[:split], ordered[split:]
        bandwidth = float(config.get("kernel_bandwidth", 1.0))
        for candidate in remaining:
            good_density = _density(candidate, good, bandwidth)
            bad_density = _density(candidate, bad, bandwidth)
            scored.append((good_density / max(bad_density, 1e-12), candidate))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("branch_id"))))
        selected_model = [item[1] for item in scored[: max(0, count - len(selected_random))]]
        mode = "finite_pool_kde_ratio"
    else:
        selected_model = remaining[: max(0, count - len(selected_random))]
        mode = "random_warmup"
    return selected_random + selected_model, mode, scored


def _in_focus(candidate, focus_regions):
    if not focus_regions:
        return True
    regions = candidate.get("region_ids")
    if regions is None:
        region = candidate.get("region_id")
        regions = [] if region is None else [region]
    return bool(set(regions) & focus_regions)


def _density(candidate, observations, bandwidth):
    if not observations:
        return 1e-12
    vector = candidate.get("bohb_features") or []
    values = []
    for item in observations:
        other = item.get("bohb_features") or []
        if vector and len(vector) == len(other):
            distance = sum((float(a) - float(b)) ** 2 for a, b in zip(vector, other))
        else:
            distance = sum(candidate.get(key) != item.get(key) for key in ("P", "x", "T"))
        values.append(math.exp(-distance / max(2 * bandwidth**2, 1e-12)))
    return sum(values) / len(values)
