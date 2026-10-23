"""Update independent model-version energy pools from recovered Relax and MC."""

from pathlib import Path

from phase_agent.analysis.phase.branch_relax_hull import ENERGY_BASIS, build_relax_hull
from phase_agent.persistence.ledger.branch_energy_pool_ledger import save_branch_energy_pool


def update_local_mlip_hull_pool(state, *, config, path, _owned=False):
    from phase_agent.analysis.convergence.mlip_relaxation import reconcile_relaxation_quality

    current = reconcile_relaxation_quality(state, _owned=_owned)
    if not path:
        return current
    system_id = (config.get("system_config") or config.get("system") or {}).get("system_id")
    version = (config.get("mlip") or {}).get("version") or (config.get("mlip") or {}).get("name")
    if not system_id or not version:
        return current
    from phase_agent.analysis.phase.phase_boundary import (
        phase_boundary_rule,
        outside_phase_boundary,
    )

    rule = phase_boundary_rule(current)
    excluded = [
        r
        for r in current.get("phase_records") or []
        if str(r.get("status", "")).startswith("excluded_")
    ]
    by_version = {}
    for task in current.get("tasks") or []:
        if (
            task.get("stage") not in {"relax_and_feature", "deep_search"}
            or task.get("status") != "completed"
            or task.get("checks_passed", True) is not True
        ):
            continue
        task_version = task.get("model_version")
        if not task_version:
            continue
        output = task.get("outputs") or {}
        if output.get("mlip_version") not in (None, task_version):
            continue
        if (
            not output.get("actual_phase")
            or (output.get("phase_identification") or {}).get("status") != "identified"
        ):
            continue
        path_value = output.get("structure_path")
        evaluated_refresh = (
            task.get("model_refresh_id")
            and output.get("single_point_completed") is True
            and (task.get("parameters") or {}).get("model_refresh_operation") == "predict"
        )
        if (
            (
                task.get("stage") != "deep_search"
                and task.get("converged") is not True
                and not evaluated_refresh
            )
            or not path_value
            or not Path(path_value).is_file()
            or not output.get("composition")
        ):
            continue
        digest = output.get("structure_checksum") or (output.get("phase_identification") or {}).get(
            "structure_sha256"
        )
        if outside_phase_boundary(
            {"composition": output["composition"], "phase": output["actual_phase"]}, rule
        ):
            continue
        if any(
            (r.get("model_version") or r.get("source_version")) == task_version
            and r.get("source_task_id") == task.get("task_id")
            and (
                (digest and digest == r.get("structure_sha256"))
                or path_value == r.get("structure_path")
            )
            for r in excluded
        ):
            continue
        by_version.setdefault(task_version, []).append(
            {
                "branch_id": task.get("branch_id"),
                "structure_id": task.get("structure_id"),
                "structure_path": str(Path(path_value).resolve()),
                "composition": output["composition"],
                "energy": output.get("energy"),
                "energy_unit": output.get("energy_unit"),
                "converged": task.get("converged"),
                "single_point_completed": bool(evaluated_refresh),
                "search_completed": task.get("stage") == "deep_search",
                "model_version": task_version,
                "stage": task.get("stage"),
                "source_task_id": task.get("task_id"),
                "source_phase": output.get("source_phase"),
                "actual_phase": output.get("actual_phase"),
                "structure_sha256": (output.get("phase_identification") or {}).get(
                    "structure_sha256"
                ),
                "phase_identification": output.get("phase_identification"),
            }
        )
    for task_version, rows in sorted(by_version.items()):
        rows.sort(key=lambda row: (row["source_task_id"] or "", row["structure_id"] or ""))
        pool = build_relax_hull(rows, model_version=task_version, system_id=system_id)
        if not pool["records"]:
            continue
        if (
            pool["version"] not in (current.get("branch_hull_batches") or {})
            or not Path(path).is_file()
        ):
            save_branch_energy_pool(pool, path)
        current.setdefault("branch_hull_batches", {})[pool["version"]] = pool
        current.setdefault("current_branch_hull_by_model", {})[task_version] = pool["version"]
    current["current_branch_hull_version"] = (
        current.get("current_branch_hull_by_model") or {}
    ).get(version)
    current["branch_energy_pool_ledger_path"] = str(path)
    return current
