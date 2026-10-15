"""Region-only approved workflow. No downstream input preparation or submission."""

from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.tools.policy.region_sampling import CONTROLS, preview
from phase_agent.tools.policy.adaptive_branch_count import configured_selection_guards
from phase_agent.tools.workflows.region_controller import sample_regions
from phase_agent.tools.workflows.run_branch_generation import run_branch_generation
from phase_agent.tools.workflows.region_review import write_result


def run_region_generation(*, action, context):
    config = context["effective_config"]
    state = deepcopy(context.get("event_state") or {})
    params = action["parameters"]
    manager, refs = context["manager"], context["phase_references"]
    try:
        fresh = preview(params, manager, refs, config, state)
        if params.get("sampling_controls") != CONTROLS or params.get("region_preview") != fresh:
            raise ValueError("区域方案所依据的账本、模型、配置或保护规则已变化，需重新审阅")
    except (ValueError, KeyError) as error:
        return {"status": "rejected", "state": state, "reason": str(error)}
    if not params["regions"]:
        _finish_review(state, action, "completed")
        return {
            "status": "completed",
            "state": state,
            "registered": [],
            "summary": {"new_branch_count": 0, "defer_reason": params["defer_reason"]},
        }
    candidates, report = sample_regions(
        manager,
        refs,
        params,
        system_config=config.get("system_config"),
        strategy_options=config.get("generation_options"),
    )
    before = set(manager.data["branches"])
    digest = hashlib.sha256(str(action["task_key"]).encode()).hexdigest()[:16]
    directory = Path(config["structure_directory"]) / f"RegionSampling-{digest}"
    if directory.exists():
        return {
            "status": "rejected",
            "state": state,
            "reason": "生成目录已存在，须核对上次执行，不能覆盖重跑",
        }
    guards = {
        k: v if v is not None else max(1, len(candidates))
        for k, v in configured_selection_guards(config).items()
    }
    result = run_branch_generation(
        manager,
        refs,
        structure_directory=directory,
        quotas={},
        batch_size=max(1, len(candidates)),
        initial_states_per_branch=params.get("initial_states_per_branch", 3),
        seed=params.get("seed", 0),
        selection_config=guards,
        ledger_path=config.get("ledger_path"),
        system_config=config.get("system_config"),
        prepared_candidates=candidates,
        sampling_report=report,
    )
    history = {
        "task_key": action["task_key"],
        "sampling_mode": params["sampling_mode"],
        "regions": deepcopy(params["regions"]),
        "model_version": state.get("active_model_version"),
        "refresh_plan_checksum": ((state.get("model_refresh") or {}).get("plan") or {}).get(
            "checksum"
        ),
        "new_branch_ids": sorted({r["branch_id"] for r in result["registered"]} - before),
        "registered_ids": [r["structure_id"] for r in result["registered"]],
        "coverage": result["coverage"],
        "summary": result["summary"],
        "structure_directory": str(directory),
    }
    state.setdefault("generation_history", []).append(history)
    _finish_review(state, action, report.get("status", "completed"))
    state["coverage"] = result["coverage"]
    artifacts = write_result(history, manager, result["registered"], state, action)
    history["artifacts"] = artifacts
    gate = state.get("dedup_gate") or {}
    if gate.get("status") in {None, "ready"}:
        state["dedup_gate"] = {
            "status": "ready",
            "source": "local_generation_dedup",
            "valid_structure_ids": sorted(
                set(gate.get("valid_structure_ids") or []) | set(history["registered_ids"])
            ),
        }
    return {
        "status": "completed",
        "state": state,
        "registered": result["registered"],
        "coverage": result["coverage"],
        "summary": result["summary"],
        "generation_artifacts": artifacts,
    }


def _finish_review(state, action, status):
    review = state.get("branch_region_review") or {}
    if review.get("pending_task_key") == action.get("task_key"):
        review["status"] = (
            "awaiting_sampling_continuation_review"
            if status == "paused_for_review"
            else "completed"
        )
        state["branch_region_review"] = review
        state["workflow_stage"] = (
            "branch_generation_paused"
            if status == "paused_for_review"
            else "branch_generation_completed"
        )
