"""Reserve validated DFT actions idempotently in search state."""

from copy import deepcopy


def reserve_dft_actions(state: dict, decisions: list[dict]) -> dict:
    updated = deepcopy(state); effective = updated.setdefault("effective_decisions", {}); reservations = updated.setdefault("budget_reservations", {})
    for item in decisions:
        if item["action"] not in {"DFT_SINGLE_POINT", "DFT_RELAX"} or item["task_key"] in effective:
            continue
        effective[item["task_key"]] = {"status": "reserved", "action": item["action"], "candidate_id": item["candidate_id"], "config_version": item["config_version"]}
        reservations[item["task_key"]] = {
            "status": "reserved",
            "relative_cost": item["relative_cost"],
            "reserved_cost": item["relative_cost"],
            "stage": "dft_single_point" if item["action"] == "DFT_SINGLE_POINT" else "dft_relax",
            "config_version": item["config_version"],
        }
        updated["reserved_relative_cost"] = float(updated.get("reserved_relative_cost", 0)) + item["relative_cost"]
        updated["reserved_dft_cost"] = float(updated.get("reserved_dft_cost", 0)) + item["relative_cost"]
    return updated
