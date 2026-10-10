"""Public calibration for measured relative-cost samples, not elapsed time."""

import math


def calibrate_relative_cost(state, stage, *, backend=None):
    ratios = []
    for row in state.get("cost_history", []):
        if (
            row.get("stage") != stage
            or row.get("status") != "completed"
            or row.get("actual_cost_known") is False
        ):
            continue
        if (
            backend is not None
            and (row.get("backend") or (row.get("runtime_observation") or {}).get("backend"))
            != backend
        ):
            continue
        actual, planned = row.get("actual_cost"), row.get("planned_cost")
        if (
            isinstance(actual, bool)
            or isinstance(planned, bool)
            or not isinstance(actual, (int, float))
            or not isinstance(planned, (int, float))
        ):
            continue
        if not math.isfinite(actual) or not math.isfinite(planned) or actual < 0 or planned <= 0:
            continue
        ratios.append(actual / planned)
    if not ratios:
        return 1.0, 0
    observed = sum(ratios[-10:]) / len(ratios[-10:])
    return max(0.25, min(4.0, 0.3 + 0.7 * observed)), len(ratios[-10:])
