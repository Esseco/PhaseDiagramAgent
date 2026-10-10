"""Rank legal branches for low-cost Relax using phase×SOC gaps and cost."""

import random


def rank_branch_relax_prescreen(branches, *, count=None, seed=0, random_fraction=0.1):
    legal, rejected = [], []
    for branch in branches:
        if branch.get("legal") is False or branch.get("legality") in {"illegal", "rejected"}:
            rejected.append({"branch_id": branch.get("branch_id"), "reason": "illegal"})
            continue
        phase_soc = branch.get("phase_soc_region") or f"{branch.get('P')}:x={branch.get('x')}"
        gap = branch.get("phase_soc_coverage_gap", branch.get("coverage_gap_score"))
        cost = branch.get("estimated_relax_cost")
        known = gap is not None and cost is not None
        score = (float(gap) / max(float(cost), 1e-12)) if known else None
        legal.append(
            {
                **branch,
                "phase_soc_region": phase_soc,
                "relax_prescreen_score": score,
                "relax_prescreen_evidence": "complete" if known else "partial_unknown",
            }
        )
    limit = len(legal) if count is None else min(len(legal), max(0, int(count)))
    rng = random.Random(seed)
    shuffled = list(legal)
    rng.shuffle(shuffled)
    random_count = min(limit, round(limit * float(random_fraction)))
    exploration = shuffled[:random_count]
    used = {row["branch_id"] for row in exploration}
    remaining = [row for row in legal if row["branch_id"] not in used]
    remaining.sort(
        key=lambda row: (
            row["relax_prescreen_score"] is None,
            -(row["relax_prescreen_score"] or 0),
            row["branch_id"],
        )
    )
    return {
        "selected": exploration + remaining[: limit - random_count],
        "rejected": rejected,
        "method": "legal_phase_soc_gap_per_estimated_cost_with_random_exploration",
    }
