"""Replace an uncomputed initial pool inside the approved generation tool."""

from copy import deepcopy
from pathlib import Path

from phase_agent.policies.region_sampling import preview


def replace_initial_pool(*, action, context, generate):
    state = deepcopy(context.get("event_state") or {})
    manager = context["manager"]
    config = deepcopy(context["effective_config"])
    if state.get("tasks") or state.get("phase_records"):
        return {
            "status": "rejected",
            "state": state,
            "reason": "initial_pool_has_calculation_tasks_or_results",
        }
    for row in manager.data.get("structures", {}).values():
        if any(
            row.get(key) for key in ("calculations", "results", "energies", "calculation_attempts")
        ) or any((row.get("stage_history") or {}).values()):
            return {
                "status": "rejected",
                "state": state,
                "reason": "initial_pool_has_structure_results",
            }
    if not config.get("ledger_path"):
        return {
            "status": "rejected",
            "state": state,
            "reason": "initial_pool_replacement_requires_ledger",
        }
    root = Path(config["structure_directory"]).resolve()
    # Only exact files owned by the old ledger are eligible; never remove a tree.
    old_paths = set()
    for row in list(manager.data["branches"].values()) + list(manager.data["structures"].values()):
        for key in ("source_path", "full_na_structure_path"):
            if row.get(key):
                path = Path(row[key]).resolve()
                if not path.is_relative_to(root) or path == root:
                    return {
                        "status": "rejected",
                        "state": state,
                        "reason": "unsafe_initial_pool_cleanup_path",
                    }
                old_paths.add(path)
    shadow = deepcopy(manager)
    shadow.data["branches"] = {}
    shadow.data["structures"] = {}
    retired = state.setdefault("branch_region_review", {}).setdefault("retired_batches", [])
    for history in state.get("generation_history") or []:
        retired.append(
            {
                "task_key": history.get("task_key"),
                "branch_ids": history.get("new_branch_ids") or [],
                "reason": "user_requested_initial_pool_replacement",
            }
        )
    state["generation_history"] = []
    state["branch_candidates"] = []
    state["coverage"] = []
    state["dedup_gate"] = {"status": "ready", "valid_structure_ids": []}
    replacement = deepcopy(action)
    params = replacement["parameters"]
    params["replace_initial_pool"] = False
    params["region_preview"] = preview(params, shadow, context["phase_references"], config, state)
    ledger = config.pop("ledger_path")
    config["ledger_path"] = None
    result = generate(
        action=replacement,
        context={**context, "effective_config": config, "event_state": state, "manager": shadow},
    )
    phases = {row["P"] for row in shadow.data["branches"].values()}
    expected = {row["phase"] for row in params["regions"]}
    if result.get("status") != "completed" or not phases or not expected.issubset(phases):
        return {
            "status": "rejected",
            "state": context.get("event_state") or {},
            "reason": "replacement_failed_or_missing_phase",
            "generation_summary": result.get("summary"),
        }
    if len(shadow.data["branches"]) > params["max_new_branches"]:
        raise ValueError("initial_pool_replacement_exceeds_limit")
    old_count = len(manager.data["branches"])
    shadow.save(ledger)
    manager.data = shadow.data
    current_paths = {
        Path(row[key]).resolve()
        for row in list(shadow.data["branches"].values()) + list(shadow.data["structures"].values())
        for key in ("source_path", "full_na_structure_path")
        if row.get(key)
    }
    cleanup_errors = []
    for path in sorted(old_paths - current_paths):
        try:
            path.unlink(missing_ok=True)
        except OSError as error:
            cleanup_errors.append({"path": str(path), "error": str(error)})
    result["summary"]["replaced_branch_count"] = old_count
    result["summary"]["cleanup_errors"] = cleanup_errors
    result["state"]["branch_candidates"] = list(manager.data["branches"].values())
    result["state"]["workflow_stage"] = "branch_generation_completed"
    return result
