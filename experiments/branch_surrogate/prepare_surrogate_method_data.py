"""Connect the fixed branch dataset to phase_agent.science.surrogate_models feature outputs."""

from phase_agent.science.features.build_cost_record import build_cost_record
from phase_agent.tools.budget.select_comparable_cost import select_comparable_cost
from phase_agent.tools.budget.check_structure_cost_limits import check_structure_cost_limits


def prepare_surrogate_method_data(dataset: dict, feature_table: dict | None, *, config: dict) -> dict:
    by_branch = {row["branch_id"]: row for row in (feature_table or {}).get("rows", [])}
    rows = []
    for source in dataset.get("rows", []):
        target = (source.get("target") or {}).get(config.get("target_key", "distance_to_fixed_hull"))
        target_record = source.get("target") or {}
        raw = target_record.get("raw_energy_records") or []
        best = next((item for item in raw if item.get("result_id") == target_record.get("best_result_id")), raw[0] if len(raw) == 1 else {})
        label_cost_record = build_cost_record(atom_count=best.get("atom_count"), evaluation_count=target_record.get("target_mc_budget", best.get("mc_budget")), proxy_config=config.get("cost_model") or {}, measured_value=target_record.get("actual_cost"), measured_unit=target_record.get("cost_unit"))
        row = {"branch_id": source["branch_id"], "split": source.get("split"), "split_group": source.get("split_group"), "target": target, "label_cost": select_comparable_cost(label_cost_record, basis=config.get("cost_basis", "proxy_relative"), measured_unit=config.get("measured_cost_unit")), "label_cost_record": label_cost_record, "atom_count": best.get("atom_count"), "det_H": (source.get("branch") or {}).get("det_H"), "composition_group": target_record.get("composition_group"), "important": bool(source.get("important") or (source.get("branch") or {}).get("important")), "trial_energy": (source.get("features") or {}).get(config.get("trial_energy_key", "trial_energy_per_atom")), "manual": None, "embedding": None, "feature_cost": 0.0, "feature_cost_status": "known"}
        records = by_branch.get(source["branch_id"], {}).get("structure_records", [])
        usable = [item for item in records if (item.get("relax") or {}).get("status") == "completed" and (item.get("relax") or {}).get("converged") is True]
        manual = [item["manual"]["values"] for item in usable if (item.get("manual") or {}).get("status") == "completed"]
        embedding = [item["embedding"]["values"] for item in usable if (item.get("embedding") or {}).get("status") == "completed"]
        row["manual"] = _mean_dicts(manual) if manual else None
        row["embedding"] = _mean_vectors(embedding) if embedding else None
        proxy_values = [(((item.get("relax") or {}).get("cost_record") or {}).get("proxy") or {}).get("value") for item in records]
        if config.get("cost_basis", "proxy_relative") == "proxy_relative":
            row["feature_cost"] = sum(proxy_values) if proxy_values and all(value is not None for value in proxy_values) else None
        else:
            measured = [(((item.get("relax") or {}).get("cost_record") or {}).get("measured") or {}).get("value") for item in records]
            row["feature_cost"] = sum(measured) if measured and all(value is not None for value in measured) else None
        row["feature_cost_status"] = "known" if row["feature_cost"] is not None else "unknown"
        row["cost_limit_check"] = check_structure_cost_limits(atom_count=row["atom_count"], det_H=row["det_H"], proxy_cost=(label_cost_record.get("proxy") or {}).get("value"), limits=config.get("structure_limits") or {})
        rows.append(row)
    return {"task": dataset.get("task"), "split_manifest": dataset.get("split_manifest"), "rows": rows}


def _mean_dicts(items):
    keys = sorted(set.intersection(*(set(item) for item in items)))
    return {key: sum(float(item[key]) for item in items) / len(items) for key in keys}


def _mean_vectors(items):
    if not items or len({len(item) for item in items}) != 1:
        return None
    return [sum(float(item[i]) for item in items) / len(items) for i in range(len(items[0]))]
