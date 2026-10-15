"""One JSON per generation, with proposed bounds and actual branch inventory."""

import hashlib
import json
from pathlib import Path
from phase_agent.tools.relax_mc_batch.files import write, read
from phase_agent.tools.workflows.branch_review_view import readable_view, raw_record
from phase_agent.tools.policy.adaptive_branch_count import configured_selection_guards


def artifact_path(state, task_key, params, directory=None):
    config = state.get("confirmed_config") or {}
    root = (state.get("workspace_layout") or {}).get("workspace_root") or (
        config.get("storage") or (config.get("config") or {}).get("storage") or {}
    ).get("workspace_root")
    if root:
        identity = (
            str(task_key or json.dumps(params, sort_keys=True))
            + ":"
            + str(state.get("active_model_version"))
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        return Path(root) / "workflow_state/branch_plans" / f"branch_plan_{digest}.json"
    return Path(directory) / "branch_inventory.json" if directory else None


def proposal_artifacts(proposal, state):
    from phase_agent.tools.policy.region_sampling import is_region
    from phase_agent.tools.workflows.region_review import write_proposal

    if is_region(proposal.get("raw_action") or {}):
        return write_proposal(proposal, state)
    action = proposal.get("raw_action") or {}
    if (action.get("tool") or proposal.get("recommended_action")) != "generate_branches":
        return {}
    params = action.get("parameters") or proposal.get("action_parameters") or {}
    path = artifact_path(state, action.get("task_key"), params)
    if path is None:
        return {}
    if path.exists():
        saved = read(path)
        original = raw_record(saved)
        if original.get("status") == "completed":
            if original is saved:
                write(path, readable_view(original), sort_keys=False)
            return {"json": str(path)}
    config = state.get("confirmed_config") or {}
    config = config.get("config") or config
    boundary = (
        config.get("boundary")
        or (config.get("system") or {}).get("boundary")
        or (config.get("generation") or {}).get("boundary")
        or {}
    )
    scope, cost = (
        params.get("module_execution_scope") or {},
        params.get("generation_cost_preview") or {},
    )
    payload = {
        "schema_version": 1,
        "status": "awaiting_approval",
        "task_key": action.get("task_key"),
        "model_version": state.get("active_model_version"),
        "config_version": state.get("confirmed_config_version"),
        "quantity_reason": (action.get("decision_review") or {}).get("rationale")
        or (action.get("scientific_decision") or {}).get("rationale")
        or action.get("reason"),
        "plan": {
            "phases": boundary.get("P"),
            "boundary": boundary,
            "max_det_H": params.get("max_det_H"),
            "supercells": [
                {"phase": phase, "H": matrix, "det_H": determinant(matrix)}
                for phase, matrices in (boundary.get("H") or {}).items()
                if isinstance(matrices, list)
                for matrix in matrices
                if determinant(matrix) is not None
                and (params.get("max_det_H") is None or determinant(matrix) <= params["max_det_H"])
            ],
            "Na_fractions": boundary.get("x") or "生成时按整数占位规则确定；端点相规则见 phases",
            "branch_count_policy": (
                action.get("decision_context") or proposal.get("decision_context") or {}
            ).get("branch_count_policy"),
            "selection_guards": configured_selection_guards(config),
            "candidate_quota": params.get("total_quota"),
            "selected_branch_upper": params.get("batch_size"),
            "initial_states_per_branch": params.get("initial_states_per_branch"),
            "seed": params.get("seed"),
            "allocations": [
                {
                    k: row[k]
                    for k in (
                        "phase",
                        "strategy",
                        "quota",
                        "min_det_H",
                        "max_det_H",
                        "na_min",
                        "na_max",
                        "reason",
                        "x",
                    )
                    if k in row
                }
                for row in params.get("generation_plan") or []
            ],
            "quotas": params.get("quotas"),
            "note": "配额不等于去重后新增数量；各相/H矩阵/Na含量的实际分布在生成后登记。",
        },
        "execution_scope": "generate_and_prepare_relax_mc_inputs" if scope else "generate_only",
        "submit_authorized": False,
        "budget_preview": {
            k: cost[k] for k in ("status", "relax_mc_cost_upper", "remaining_budget") if k in cost
        },
        "actual": None,
    }
    write(path, readable_view(payload), sort_keys=False)
    return {"json": str(path)}


def determinant(matrix):
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(not isinstance(r, list) or len(r) != 3 for r in matrix)
    ):
        return None
    a, b, c = matrix
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def result_artifacts(directory, history, manager, registered, state, action):
    from phase_agent.tools.policy.region_sampling import is_region

    if is_region(action):
        from phase_agent.tools.workflows.region_review import write_result

        return write_result(history, manager, registered, state, action)
    path = artifact_path(state, action.get("task_key"), action.get("parameters") or {}, directory)
    payload = (
        raw_record(read(path))
        if path.exists()
        else {"schema_version": 1, "task_key": action.get("task_key")}
    )
    new_ids, rows = set(history.get("new_branch_ids") or []), []
    for bid in sorted({r["branch_id"] for r in registered}):
        branch = manager.data["branches"][bid]
        structures = []
        for item in registered:
            if item["branch_id"] != bid:
                continue
            structure = (manager.data.get("structures") or {}).get(item["structure_id"]) or {}
            structures.append(
                {
                    "structure_id": item["structure_id"],
                    "path": structure.get("source_path"),
                    "atom_count": sum((structure.get("composition") or {}).values()) or None,
                }
            )
        rows.append(
            {
                "branch_id": bid,
                "is_new": bid in new_ids,
                "phase": branch.get("P"),
                "H": branch.get("H"),
                "supercell_det": determinant(branch.get("H")),
                "Na_fraction": str(branch["x"]) if branch.get("x") is not None else None,
                "composition": branch.get("composition"),
                "structure_count": len(structures),
                "structures": structures,
            }
        )
    payload.update(
        status="completed",
        actual={
            "new_branch_count": len(new_ids),
            "registered_branch_count": len(rows),
            "registered_structure_count": len(registered),
            "branches": rows,
        },
    )
    write(path, readable_view(payload), sort_keys=False)
    return {"json": str(path)}
