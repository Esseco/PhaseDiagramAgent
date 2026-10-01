"""Phase-resolved observed coverage, not a sampling policy or a new hull."""

import math


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize_phase_coverage(entries):
    phases = {}
    for entry in entries:
        phase = entry.get("actual_phase") or entry.get("phase") or "unknown"
        row = phases.setdefault(phase, {"phase": phase, "entry_count": 0,
            "stable_count": 0, "missing_na_count": 0, "missing_ehull_count": 0,
            "observed_na_values": set(), "lowest_ehull_entry": None})
        row["entry_count"] += 1
        row["stable_count"] += bool(entry.get("is_stable"))
        x = entry.get("x_Na_per_O2")
        composition = entry.get("composition") or {}
        if not _number(x) and _number(composition.get("O")) and composition["O"] > 0:
            sodium = composition.get("Na", 0)
            if _number(sodium):
                x = 2 * sodium / composition["O"]
        if _number(x):
            row["observed_na_values"].add(x)
        else:
            row["missing_na_count"] += 1
        ehull = entry.get("ehull")
        if not _number(ehull):
            row["missing_ehull_count"] += 1
        elif row["lowest_ehull_entry"] is None or ehull < row["lowest_ehull_entry"]["ehull"]:
            row["lowest_ehull_entry"] = {"record_id": entry.get("record_id"),
                "structure_id": entry.get("structure_id"), "x_Na_per_O2": x if _number(x) else None,
                "ehull": ehull, "ehull_unit": entry.get("ehull_unit"),
                "is_stable": entry.get("is_stable")}
    output = []
    for phase in sorted(phases):
        row = phases[phase]
        values = sorted(row.pop("observed_na_values"))
        row.update(observed_na_count=len(values), observed_na_range=[values[0], values[-1]] if values else None)
        output.append(row)
    return {"phases": output, "interpretation": "Observed coverage only; no unobserved gap or ground state is inferred. Ehull retains the source snapshot unit."}
