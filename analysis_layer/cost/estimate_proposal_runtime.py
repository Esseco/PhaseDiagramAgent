"""Approximate serial workload duration, never a queue or completion ETA."""
from analysis_layer.cost.predict_runtime import predict_runtime


def _atoms(row):
    return row.get("atom_count") or sum((row.get("composition") or {}).values()) or None


def estimate_proposal_runtime(action, state):
    params = action.get("parameters") or {}
    tool = action.get("tool")
    if params.get("mode") == "model_refresh_inputs":
        return (params.get("refresh_preview") or {}).get("runtime_budget") or {"status": "unavailable"}
    specs = []
    if tool == "select_dft_candidates":
        candidates = {r.get("candidate_id") or r.get("structure_id"): r
                      for r in state.get("qbc_candidates") or []}
        stages = {"DFT_SINGLE_POINT": "dft_single_point", "DFT_RELAX": "dft_relax"}
        for row in params.get("decisions") or []:
            if isinstance(row, dict) and row.get("action") in stages:
                specs.append({"stage": stages[row["action"]],
                              "atom_count": _atoms(candidates.get(row.get("candidate_id"), {}))})
    elif tool == "allocate_mc_bohb":
        branches = {r.get("branch_id"): r for r in state.get("branch_candidates") or []}
        for row in (params.get("budget_preview") or {}).get("allocations") or []:
            specs.append({"stage": "deep_search", "atom_count": _atoms(row) or _atoms(branches.get(row.get("branch_id"), {})),
                          "mc_steps": row.get("max_mc_steps"),
                          "patience": row.get("patience_steps"), "maximum": row.get("max_mc_steps")})
    elif tool in {"prepare_local_batch_files", "run_calculation_stage"}:
        targets = set(action.get("target_ids") or [])
        for row in state.get("tasks") or []:
            if row.get("status") == "pending" and (row.get("structure_id") in targets or row.get("branch_id") in targets):
                parameters = row.get("parameters") or {}
                specs.append({"stage": row.get("stage"), "atom_count": _atoms(row),
                              "mc_steps": parameters.get("max_mc_steps"),
                              "patience": parameters.get("patience_steps"),
                              "maximum": parameters.get("max_mc_steps")})
    reports = [predict_runtime(state=state, **spec) for spec in specs]
    known = [r for r in reports if r.get("elapsed_seconds") is not None]
    complete = bool(reports) and len(known) == len(reports)
    return {"status": "estimated" if complete else "unavailable",
            "task_count": len(reports), "predicted_tasks": len(known),
            "serial_seconds": sum(r["elapsed_seconds"] for r in known) if complete else None,
            "sample_range_seconds": [sum(r["sample_range_seconds"][i] for r in known)
                                     for i in (0, 1)] if complete else None,
            "note": "串行任务总耗时粗估；不含排队，不是并行完成时间；观测范围不是置信区间。"}
