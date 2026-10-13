"""Compose existing recovery, phase feedback and MC scheduling for one frozen batch."""

from copy import deepcopy
from pathlib import Path

from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback
from phase_agent.analysis.phase.branch_relax_hull import build_relax_hull, rank_relaxed_branches
from phase_agent.science.mc.schedule_tiered_mc import schedule_tiered_mc
from phase_agent.science.mc.second_round_state import (
    reconciled_mc_state,
    second_round_candidates,
    first_round_source,
)
from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost
from phase_agent.tools.budget.reserve_budget import reserve_budget
from phase_agent.tools.relax_mc_batch.files import write, rebase
from phase_agent.tools.remote.integrity import payload_checksum


def feedback(state, results, manager, references, root, model_version):
    state = reconcile_task_results(state, results)["state"]
    return apply_scientific_feedback(
        state,
        results,
        manager=manager,
        active_model_version=model_version,
        phase_references=references,
        phase_diagram_directory=Path(root) / "analysis",
        phase_identification_cache_path=Path(root) / "phase_identification.json",
    )["state"]


def next_mc(state, manager, manifest, root, index):
    config, version = manifest["config"], manifest["model_version"]
    task_map = {row["task_id"]: row for row in state["tasks"]}
    rows = []
    for record in state.get("phase_records") or []:
        task = task_map.get(record.get("source_task_id")) or {}
        if task.get("stage") == "relax_and_feature" and record.get("status") == "completed":
            rows.append(
                {
                    **record,
                    "branch_id": task["branch_id"],
                    "converged": task.get("converged"),
                    "stage": "relax_and_feature",
                }
            )
    pool = build_relax_hull(rows, model_version=version)
    state.setdefault("branch_hull_batches", {})[pool["version"]] = pool
    state["current_branch_hull_version"] = pool["version"]
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if diagram.get("status") != "completed" or diagram.get("model_version") != version:
        return {
            "status": "phase_diagram_unavailable",
            "actions": [],
            "state": state,
            "reason": diagram.get("reason", diagram.get("error"))
            or "current_model_hull_required_before_mc",
        }
    candidates, missing = rank_relaxed_branches(
        list(manager.data["branches"].values()), pool, phase_diagram=diagram
    )
    if index and state.get("tiered_mc_state"):
        source = first_round_source(state, version)
        candidates, missing = second_round_candidates(candidates, state, diagram, version)
    policy = deepcopy(config["mc_policy"])
    scheduled = schedule_tiered_mc(
        candidates,
        reconciled_mc_state(state),
        policy=policy,
        total_budget=float("inf"),
        seed=manifest["seed"] + index,
        model_version=version,
        hull_reference_version=pool["version"],
        phase_diagram_version=diagram.get("version"),
        scope_by_model=True,
    )
    state["tiered_mc_state"] = scheduled["state"]
    atoms = {row["branch_id"]: row["atom_count"] for row in candidates}
    for child in scheduled["actions"]:
        cost = estimate_stage_cost(
            "deep_search",
            atom_count=atoms[child["branch_id"]],
            mc_steps=child["max_mc_steps"],
            budgets=config["budgets"],
        )["value"]
        reserved = reserve_budget(
            state,
            task_key=child["task_key"],
            stage="deep_search",
            amount=cost,
            limits=config["budgets"],
            config_version=manifest["config_version"],
            model_version=version,
        )
        if reserved["status"] == "rejected":
            raise ValueError("approved_batch_budget_reservation_rejected")
        state = reserved["state"]
        task = {
            **child,
            "task_id": "MC-" + child["task_key"].split(":")[-1],
            "config_version": manifest["config_version"],
            "atom_count": atoms[child["branch_id"]],
            "planned_relative_cost": cost,
            "relax_mc_bundle_id": manifest["bundle_id"],
            "parameters": {
                "max_mc_steps": child["max_mc_steps"],
                "patience_steps": child["patience_steps"],
                "seed": child["seed"],
            },
        }
        state.setdefault("tasks", []).append(task)
    state["pending_tasks"] = [
        r for r in state["tasks"] if r.get("status") in {"pending", "running"}
    ]
    if index and scheduled["actions"]:
        record = {
            "model_version": version,
            "source_checksum": source["checksum"],
            "allocation_checksum": payload_checksum(scheduled["actions"]),
            "task_ids": ["MC-" + r["task_key"].split(":")[-1] for r in scheduled["actions"]],
        }
        state.setdefault("mc_second_round_allocations", []).append(record)
        state["mc_second_round_allocation"] = record
    scheduled["state"] = state
    scheduled["missing_branch_ids"] = missing
    write(
        Path(root) / "analysis" / f"mc_plan_{index}.json",
        rebase({k: v for k, v in scheduled.items() if k != "state"}, root, portable=True),
    )
    return scheduled


def candidate_pool(state):
    by_path = {
        r.get("structure_path"): r
        for r in ((state.get("phase_diagrams") or {}).get("mlip") or {}).get("entries") or []
    }
    latest = {}
    all_pool = []
    for task in state.get("tasks") or []:
        if task.get("status") != "completed":
            continue
        output = task.get("outputs") or {}
        path = output.get("structure_path")
        entry = by_path.get(path) or {}
        if path:
            latest[task["branch_id"]] = {
                "candidate_id": task["structure_id"],
                "structure_id": task["structure_id"],
                "branch_id": task["branch_id"],
                "structure_path": path,
                "composition": output.get("composition"),
                "energy": output.get("energy"),
                "ehull": entry.get("ehull"),
                "ehull_unit": entry.get("ehull_unit"),
                "qbc": deepcopy(output.get("qbc") or {}),
                "model_version": task["model_version"],
                "actual_phase": task.get("actual_phase"),
                "phase_identification_status": task.get("phase_identification_status"),
            }
        for index, row in enumerate(output.get("pool_summary") or []):
            all_pool.append(
                {
                    "source_task_id": task["task_id"],
                    "branch_id": task["branch_id"],
                    "pool_index": index,
                    **deepcopy(row),
                }
            )
    from phase_agent.science.qbc.post_mc_candidates import post_mc_candidates

    return {
        "branch_candidates": post_mc_candidates(state, state["active_model_version"])
        or list(latest.values()),
        "mc_pool": all_pool,
    }
