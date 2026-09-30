"""Order-of-magnitude timing predictions from lightweight completed samples."""
import math
from statistics import median


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def predict_runtime(stage, *, atom_count, state, backend=None, hardware=None,
                    mc_steps=None, patience=None, maximum=None):
    if not _positive(atom_count):
        return {"status": "insufficient_samples", "samples": 0}
    rows = []
    for row in (state or {}).get("cost_history") or []:
        timing = row.get("runtime_observation") or {}
        atoms = row.get("atom_count")
        if row.get("stage") != stage or row.get("status") != "completed":
            continue
        if not _positive(atoms) or not _positive(timing.get("elapsed_seconds")):
            continue
        if not timing.get("backend") or not timing.get("hardware"):
            continue
        if backend is not None and timing["backend"] != backend:
            continue
        if hardware is not None and timing["hardware"] != hardware:
            continue
        if not .5 <= atoms / atom_count <= 2:
            continue
        if stage == "deep_search" and not _positive(timing.get("actual_mc_steps") or row.get("actual_mc_steps")):
            continue
        rows.append((row, timing))
    if not rows:
        return {"status": "insufficient_samples", "samples": 0}
    groups = {}
    for row, timing in rows:
        key = (timing["backend"], timing["hardware"], timing.get("gpu_count"), timing.get("cpu_count"))
        groups.setdefault(key, []).append((row, timing))
    key, cohort = max(groups.items(), key=lambda item: len(item[1]))
    cohort = cohort[-30:]
    values = []
    step_samples = []
    exponent = 3 if stage.startswith("dft") else 1.2
    for row, timing in cohort:
        seconds = timing["elapsed_seconds"] * (atom_count / row["atom_count"]) ** exponent
        if stage == "deep_search":
            steps = timing.get("actual_mc_steps") or row.get("actual_mc_steps")
            seconds *= mc_steps / steps
            if timing.get("patience") == patience and timing.get("max_mc_steps") == maximum:
                step_samples.append(steps)
        values.append(seconds)
    seconds = median(values)
    return {"status": "historical_runtime_estimate", "samples": len(values),
            "elapsed_seconds": seconds, "sample_range_seconds": [min(values), max(values)],
            "backend": key[0], "hardware": key[1], "gpu_count": key[2], "cpu_count": key[3],
            "gpu_hours": seconds * key[2] / 3600 if _positive(key[2]) else None,
            "cpu_core_hours": seconds * key[3] / 3600 if _positive(key[3]) else None,
            "typical_mc_steps": median(step_samples) if step_samples else None,
            "typical_mc_samples": len(step_samples),
            "note": "Rough size-scaled median; observed range is not a confidence interval. Unknown counts stay unknown."}
