"""Source-bound evidence previews, never a substitute for complete candidate pools."""

from collections import Counter
from copy import deepcopy
import json
import math

_CATEGORIES = {
    "phase",
    "P",
    "status",
    "stage",
    "x",
    "model_version",
    "config_version",
    "strategy",
    "method",
    "action",
}
_HUMAN = {"human_constraints", "human_system_knowledge", "search_rules", "frozen_parameter_advice"}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _rows(rows):
    fields = {}
    for key in sorted({k for row in rows for k in row}):
        values = [row[key] for row in rows if key in row]
        info = {}
        if len(values) != len(rows):
            info["missing_count"] = len(rows) - len(values)
        if any(v is None for v in values):
            info["null_count"] = sum(v is None for v in values)
        if key in _HUMAN:
            info["values"] = deepcopy(values)
        numeric = [v for v in values if type(v) in {int, float} and math.isfinite(v)]
        if numeric:
            info.update(finite_count=len(numeric), finite_min=min(numeric), finite_max=max(numeric))
        if key in _CATEGORIES:
            counts = Counter(
                json.dumps(v, sort_keys=True, default=str)
                for v in values
                if not isinstance(v, (dict, list))
            )
            info["categorical_counts"] = [
                {"value": json.loads(v), "count": count} for v, count in sorted(counts.items())
            ]
        fields[key] = info
    return {
        "count": len(rows),
        "fields": fields,
        "counter_defaults": "Absent missing_count/null_count means zero in this preview only; no row identities or correlations are supplied.",
    }


def summarize(value):
    """All rows contribute; no row sample, scientific threshold or rounded statistic."""
    if isinstance(value, dict):
        if (
            len(value) > 16
            and not _HUMAN.intersection(value)
            and all(isinstance(v, dict) for v in value.values())
        ):
            return {"mapping_entries": len(value), "values_overview": _rows(list(value.values()))}
        return {k: deepcopy(v) if k in _HUMAN else summarize(v) for k, v in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(row, dict) for row in value):
            return _rows(value)
        return {"count": len(value), "types": dict(Counter(type(v).__name__ for v in value))}
    return deepcopy(value)


def split_read(sources, names, view, revision):
    """Small fields stay exact. Large previews are separately labeled as incomplete."""
    full = set()
    overviews = {}
    for key in names:
        value = sources[key]
        if view == "full" or key in _HUMAN or _size(value) <= 1200:
            full.add(key)
            continue
        summary = summarize(value)
        preview = {
            "complete_field": False,
            "source_field": key,
            "evidence_revision": revision,
            "summary": summary,
        }
        if _size(preview) >= _size(value):
            full.add(key)
        else:
            overviews[key] = preview
    return full, overviews


def action_uses_brief_core(action):
    from phase_agent.tools.dispatch.scientific_operations import bind_operation

    bound = bind_operation(action) if action.get("operation") is not None else action
    tool = bound.get("tool") or bound.get("action_type")
    return tool in {"select_dft_candidates", "allocate_mc_bohb"} or (
        tool == "prepare_local_batch_files"
        and (bound.get("parameters") or {}).get("mode")
        in {"relax_mc_inputs", "mc_inputs", "dft_inputs"}
    )


def decision_core(brief):
    """Exact policy/limits/unknowns and all other facts; only large coverage/detail maps defer."""
    deferred = {
        key
        for key in {"coverage", "task_status_by_branch"}
        if key in brief and _size(brief[key]) > 1200
    }
    return {key: deepcopy(value) for key, value in brief.items() if key not in deferred}, deferred
