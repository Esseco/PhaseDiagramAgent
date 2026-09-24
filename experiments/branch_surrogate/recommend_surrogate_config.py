"""Create a loadable recommendation without changing production policy."""


def recommend_surrogate_config(report: dict, *, config: dict) -> dict:
    requirement = config.get("recommendation") or {}
    candidates = []
    for name, result in report.get("methods", {}).items():
        if name in {"random", "trial_energy"}:
            continue
        test = result.get("test") or {}
        replay = result.get("replay") or {}
        if test.get("status") != "completed" or test.get("row_count", 0) < int(requirement.get("minimum_test_rows", 10)):
            continue
        if requirement.get("require_replay", True) and replay.get("status") != "completed":
            continue
        metric = test.get(requirement.get("primary_metric", "important_miss_rate"))
        if metric is not None:
            candidates.append((float(metric), name))
    if not candidates:
        return {"status": "not_recommended", "production_policy_change": False, "reason": "no method satisfies validation requirements"}
    _, method = min(candidates)
    return {"status": "candidate", "method": method, "production_policy_change": False, "reason": "offline candidate only; requires explicit approval and prospective validation"}
