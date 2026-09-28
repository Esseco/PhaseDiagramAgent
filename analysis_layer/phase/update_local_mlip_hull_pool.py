"""Build a local versioned MLIP Relax energy pool from recovered task facts."""
from copy import deepcopy
from pathlib import Path

from analysis_layer.phase.branch_relax_hull import ENERGY_BASIS, build_relax_hull
from data_layer.ledger.branch_energy_pool_ledger import save_branch_energy_pool


def update_local_mlip_hull_pool(state, *, config, path):
    current = deepcopy(state)
    if not path:
        return current
    system_id = ((config.get("system_config") or config.get("system") or {}).get("system_id"))
    version = ((config.get("mlip") or {}).get("version")
               or (config.get("mlip") or {}).get("name"))
    if not system_id or not version:
        return current
    rows = []
    for task in current.get("tasks") or []:
        if task.get("stage") != "relax_and_feature" or task.get("status") != "completed":
            continue
        if task.get("model_version") != version:
            continue
        output = task.get("outputs") or {}
        path_value = output.get("structure_path")
        if (task.get("converged") is not True or not path_value
                or not Path(path_value).is_file() or not output.get("composition")):
            continue
        rows.append({"branch_id": task.get("branch_id"),
                     "structure_id": task.get("structure_id"),
                     "structure_path": str(Path(path_value).resolve()),
                     "composition": output["composition"], "energy": output.get("energy"),
                     "energy_unit": output.get("energy_unit"),
                     "converged": True, "model_version": version,
                     "source_task_id": task.get("task_id")})
    if not rows:
        return current
    rows.sort(key=lambda row: (row["source_task_id"] or "", row["structure_id"] or ""))
    pool = build_relax_hull(rows, model_version=version, system_id=system_id)
    if not pool["records"]:
        return current
    save_branch_energy_pool(pool, path)
    current.setdefault("branch_hull_batches", {})[pool["version"]] = pool
    current["current_branch_hull_version"] = pool["version"]
    current["branch_energy_pool_ledger_path"] = str(path)
    return current
