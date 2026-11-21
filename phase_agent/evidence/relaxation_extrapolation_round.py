"""Aggregate complete, per-structure relaxation monitoring for DFT policy."""

def assess_round(rows, *, scope_complete):
    """Count distinct actually relaxed structures; unknown evidence cannot clear a round.

    Rows must be produced by verified round-specific recovery, not inferred from
    a subset of successful results. Conflicting duplicate observations are unknown.
    """
    if type(scope_complete) is not bool:
        raise ValueError("scope_complete must be boolean")
    observations = {}
    for row in rows:
        identity = row.get("structure_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("missing_structure_id")
        monitored = row.get("monitor_complete") is True
        failure = row.get("extrapolation_failure")
        value = failure if monitored and type(failure) is bool else None
        if identity not in observations:
            observations[identity] = value
        elif observations[identity] != value:
            observations[identity] = None
    if not observations:
        return {"status": "unknown", "structure_count": 0, "failure_count": None,
                "failure_fraction": None, "direct_sampling_required": None}
    complete = scope_complete and all(value is not None for value in observations.values())
    failures = [key for key, value in observations.items() if value is True]
    count, total = len(failures), len(observations)
    return {"status": "complete" if complete else "unknown",
            "structure_count": total,
            "failure_count": count,
            "failure_fraction": count / total if complete else None,
            "direct_sampling_required": count * 10 >= total if complete else None}
