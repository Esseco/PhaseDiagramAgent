"""Fair ordering and audit summary when an MC budget is binding."""

from collections import defaultdict, deque


def mc_stratum(action, branch_phase):
    return (str(action.get("tier") or "unknown"),
            str(branch_phase.get(action.get("branch_id")) or "unknown"))


def interleave_mc_strata(actions, branch_phase):
    """Give each tier/phase one chance before taking a second from any stratum."""
    groups = defaultdict(deque)
    for action in actions:
        groups[mc_stratum(action, branch_phase)].append(action)
    ordered = []
    keys = sorted(groups)
    while keys:
        following = []
        for key in keys:
            ordered.append(groups[key].popleft())
            if groups[key]:
                following.append(key)
        keys = following
    return ordered


def summarize_mc_interception(accepted, rejected, branch_phase, *, full_plan_approved):
    summary = {"approved_full_plan": full_plan_approved,
               "accepted_count": len(accepted), "rejected_count": len(rejected),
               "strata": {}, "reasons": sorted({reason for row in rejected
                                                 for reason in row["reasons"]})}
    for action, field in ([(row, "accepted") for row in accepted]
                          + [(row["action"], "rejected") for row in rejected]):
        tier, phase = mc_stratum(action, branch_phase)
        row = summary["strata"].setdefault(f"{tier}/{phase}",
                                           {"tier": tier, "phase": phase,
                                            "accepted": 0, "rejected": 0})
        row[field] += 1
    return summary
