"""把正式 Agent tools 接到现有生成、BOHB 和计算实现。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from decision_layer.strategy.choose_generation_strategy import choose_generation_strategy
from execution_layer.dispatch.dispatch_calculation_stage import execute_registered_stage
from execution_layer.workflows.run_branch_generation import run_branch_generation
from execution_layer.workflows.run_round_scheduler import run_round_scheduler
from execution_layer.budget.reserve_budget import reserve_budget
from scientific_layer.bohb.build_bohb_candidate_pool import build_bohb_candidate_pool
from scientific_layer.bohb.collect_bohb_results import collect_bohb_results
from execution_layer.workflows.prepare_branch_relaxation import prepare_branch_relaxation
from execution_layer.state.branch_batch_state import (ACTIVE_BRANCH_BATCH_STATES,
    create_branch_batch, transition_branch_batch)
from scientific_layer.mc.schedule_tiered_mc import schedule_tiered_mc
from decision_layer.scoring.rank_branch_relax_prescreen import rank_branch_relax_prescreen
from execution_layer.workflows.prepare_dedup_batch import prepare_dedup_batch


def create_active_learning_handlers(stage_registry):
    return {
        "generate_branches": _generate_branches,
        "prepare_dedup_batch": _prepare_dedup,
        "allocate_mc_bohb": _allocate_mc_bohb,
        "run_calculation_stage": lambda *, action, context: _run_calculation_stage(action, context, stage_registry),
    }


def _prepare_dedup(*, action, context):
    state = deepcopy(context.get("event_state") or {})
    requested = set(action.get("target_ids") or [])
    candidates = []
    for structure_id, record in context["manager"].data.get("structures", {}).items():
        if requested and structure_id not in requested: continue
        metadata = record.get("metadata") or {}
        candidates.append({"structure_id": structure_id,
                           "periodic_search_space_id": metadata.get("periodic_search_space_id")})
    settings = context["effective_config"].get("dedup") or {}
    return prepare_dedup_batch(
        candidates, state, config_version=context["config_version"],
        model_version=state.get("active_model_version") or
                      (context["effective_config"].get("mlip") or {}).get("version"),
        approved=True, budget_limit=settings.get("budget_limit"),
        resource_limit=settings.get("resource_limit"),
        budget_limits=context["effective_config"].get("budgets"),
    )


def _generate_branches(*, action, context):
    config = context["effective_config"]; params = deepcopy(action.get("parameters") or {})
    quotas = params.get("quotas")
    if quotas is None:
        quotas = choose_generation_strategy(
            context.get("event_state", {}).get("generation_metrics") or config.get("generation_metrics") or {},
            total_quota=int(params.get("total_quota", config.get("total_quota", 0))),
        )["quotas"]
    options = deepcopy(config.get("generation_options") or {})
    result = run_branch_generation(
        context["manager"], context["phase_references"],
        structure_directory=config["structure_directory"], quotas=quotas,
        batch_size=int(params.get("batch_size", config["batch_size"])),
        initial_states_per_branch=int(params.get("initial_states_per_branch", config["initial_states_per_branch"])),
        seed=int(params.get("seed", config["seed"])), ledger_path=config.get("ledger_path"),
        system_config=config.get("system_config"), **options,
    )
    state = deepcopy(context.get("event_state") or {})
    state.setdefault("generation_history", []).append({
        "task_key": action.get("task_key"), "quotas": deepcopy(quotas),
        "registered_ids": [item.get("structure_id") for item in result["registered"]],
        "coverage": deepcopy(result["coverage"]),
    })
    state["coverage"] = deepcopy(result["coverage"])
    return {"status": "completed", "state": state, "registered": result["registered"], "coverage": result["coverage"]}


def _allocate_mc_bohb(*, action, context):
    config = context["effective_config"]; params = deepcopy(action.get("parameters") or {})
    state = deepcopy(context.get("event_state") or {})
    region_builder = context.get("region_builder") or _default_region
    candidates = build_bohb_candidate_pool(
        context["manager"], feature_builder=context.get("bohb_feature_builder"),
        group_builder=context.get("bohb_group_builder"), region_builder=region_builder,
    )
    requested = list(dict.fromkeys(action.get("target_ids") or []))
    lifecycle = state.get("branch_batch") or {}
    frozen_batch = lifecycle if lifecycle.get("status") in ACTIVE_BRANCH_BATCH_STATES else (state.get("pending_branch_screening") or {})
    frozen_ids = list(frozen_batch.get("branch_ids") or [])
    if frozen_ids:
        if requested and set(requested) != set(frozen_ids):
            return {"status": "not_configured", "state": state,
                    "error": "agent_branch_batch_changed_during_screening",
                    "required_branch_ids": frozen_ids}
        requested = frozen_ids
    known = {item["branch_id"] for item in candidates}
    unknown = sorted(set(requested) - known)
    if unknown:
        return {"status": "not_configured", "state": state,
                "error": "unknown_agent_branch_ids", "unknown_branch_ids": unknown}
    if requested:
        selected = set(requested)
        candidates = [item for item in candidates if item["branch_id"] in selected]
    else:
        focus = set(params.get("focus_regions") or [])
        if focus:
            candidates = [item for item in candidates if set(item.get("region_ids") or [item.get("region_id")]) & focus]
    if not candidates:
        return {"status": "not_configured", "state": state, "error": "agent_branch_batch_empty"}
    prescreen = rank_branch_relax_prescreen(
        candidates, count=params.get("max_relax_branches"),
        seed=int(params.get("seed", config.get("seed", 0))),
        random_fraction=float(params.get("exploration_fraction",
            (config.get("mc_policy") or {}).get("random_exploration_fraction", .1))),
    )
    candidates = prescreen["selected"]
    if not candidates:
        return {"status": "not_configured", "state": state,
                "error": "no_legal_branch_after_relax_prescreen", "prescreen": prescreen}
    uses_relax_screening = (config.get("bohb") or {}).get("selection_policy") == "relax_hull_uncertainty"
    if uses_relax_screening and not frozen_ids:
        state["branch_batch"] = create_branch_batch(
            [item["branch_id"] for item in candidates],
            selected_by=action.get("decision_source", "agent_tool_action"),
            reason=action.get("reason"),
        )
        state["pending_branch_screening"] = {
            "branch_ids": sorted(item["branch_id"] for item in candidates),
            "selected_by": action.get("decision_source", "agent_tool_action"),
            "reason": action.get("reason"),
        }
    for item in candidates:
        item.setdefault("atom_count", _branch_atom_count(context["manager"], item))
    bohb = deepcopy(config.get("bohb") or {})
    bohb["new_candidates_per_iteration"] = len(candidates)
    if bohb.get('selection_policy') == 'relax_hull_uncertainty':
        screening = prepare_branch_relaxation(candidates, state, context)
        state = screening['state']
        if screening['status'] != 'ready':
            target_status = 'relax_pending' if screening['status'] == 'screening_pending' else 'failed'
            batch = state.get('branch_batch')
            if batch and batch.get('status') != target_status:
                state['branch_batch'] = transition_branch_batch(batch, target_status,
                    unavailable=deepcopy(screening.get('unavailable') or []))
            return {'status': 'completed' if screening['status'] == 'screening_pending' else 'not_configured',
                    'state': state, 'screening': screening['status'], 'unavailable': screening['unavailable']}
        candidates = screening['candidates']
        batch = state.get('branch_batch')
        if batch.get('status') == 'selected':
            batch = transition_branch_batch(batch, 'relax_pending')
        if batch.get('status') == 'relax_pending':
            batch = transition_branch_batch(batch, 'relax_completed')
        if batch.get('status') == 'relax_completed':
            batch = transition_branch_batch(batch, 'hull_ready',
                                             hull_version=screening['pool']['version'])
        state['branch_batch'] = batch
        state.pop("pending_branch_screening", None)
        bohb.setdefault('scope', {})['hull_reference_version'] = screening['pool']['version']
        bohb['objective'] = {**bohb['objective'], 'name': 'relaxed_batch_hull_gap'}
    bohb.setdefault("budget_limits", config["budgets"])
    bohb.setdefault("scope", {})
    bohb["scope"] = {
        "mlip_version": bohb["scope"].get("mlip_version") or (config.get("mlip") or {}).get("version") or (config.get("mlip") or {}).get("name") or "unconfigured",
        "hull_reference_version": bohb["scope"].get("hull_reference_version") or _hull_version(state),
        "candidate_set_version": bohb["scope"].get("candidate_set_version") or _candidate_version(candidates),
    }
    if not bohb.get("bohb_selection_interface_enabled", False):
        tier_state = deepcopy(state.get("tiered_mc_state") or {})
        segments = tier_state.get("segments") or []
        task_rows = {row.get("task_key"): row for row in state.get("tasks", [])}
        tier_state["segments"] = [{**row, **deepcopy(task_rows.get(row.get("task_key")) or {})}
                                   for row in segments]
        total_mc_budget = int(params.get("mc_budget", action.get("budget", 0)))
        mc_policy = config.get("mc_policy") or {
            "tiers": [{"name": str(value), "max_mc_steps": value,
                       "patience_steps": max(1, value // 3), "min_improvement": 0.001}
                      for value in bohb.get("budget_levels", [10, 30, 90])],
            "max_segments_per_branch": 3, "max_cumulative_cost_per_branch": 90,
            "random_exploration_fraction": bohb.get("random_fraction", .1),
        }
        scheduled = schedule_tiered_mc(
            candidates, tier_state, policy=mc_policy,
            total_budget=total_mc_budget, seed=int(params.get("seed", config.get("seed", 0))),
            model_version=bohb["scope"]["mlip_version"],
            hull_reference_version=bohb["scope"]["hull_reference_version"],
        )
        state["tiered_mc_state"] = scheduled["state"]
        accepted, rejected = [], []
        for child in scheduled["actions"]:
            reservation = reserve_budget(
                state, task_key=child["task_key"], stage="deep_search",
                amount=float(child["planned_relative_cost"]), limits=config["budgets"],
                config_version=context["config_version"], model_version=child["model_version"],
            )
            if reservation["status"] != "reserved":
                rejected.append({"action": child, "reasons": reservation["reasons"]}); continue
            state = reservation["state"]
            task = {**deepcopy(child),
                    "task_id": f"MC-{hashlib.sha256(child['task_key'].encode()).hexdigest()[:12]}",
                    "object_id": child["branch_id"], "status": "pending",
                    "parameters": {"max_mc_steps": child["max_mc_steps"],
                                   "patience_steps": child["patience_steps"],
                                   "min_improvement": child["min_improvement"],
                                   "seed": child["seed"]}}
            state.setdefault("tasks", []).append(task); state.setdefault("pending_tasks", []).append(task)
            accepted.append(task)
        return {"status": "completed", "state": state, "method": scheduled["method"],
                "mc_status": scheduled["status"], "actions": accepted,
                "rejected_actions": rejected}
    scheduler = deepcopy(state.get("round_scheduler") or {})
    active = scheduler.get("active_round") or {}
    bohb_state = active.get("bohb_state")
    if bohb_state and bohb_state.get("pending_tasks"):
        pending_keys = {item["task_key"] for item in bohb_state["pending_tasks"]}
        recovered = [item for item in state.get("tasks", []) if item.get("task_key") in pending_keys and item.get("status") not in {"pending", "running"}]
        if recovered:
            active["bohb_state"] = collect_bohb_results(bohb_state, candidates, recovered, config=active["bohb_config"])
            scheduler["active_round"] = active
    total_mc_budget = int(params.get("mc_budget", action.get("budget", 0)))
    strategy = {
        "focus_regions": list(params.get("focus_regions") or []),
        "generation_quotas": deepcopy(params.get("generation_quotas") or {}),
        "mc_budget": total_mc_budget,
        "dft_budget": float(params.get("dft_budget", 0)),
        "exploration_fraction": float(params.get("exploration_fraction", config.get("round_strategy", {}).get("minimum_exploration_fraction", .1))),
        "reason": action.get("reason") or "Agent-selected branch batch",
        "source": action.get("decision_source", "agent_tool_action"),
    }
    summary = {**state, "known_region_ids": sorted({item.get("region_id") for item in candidates if item.get("region_id") is not None}), "seed": int(params.get("seed", config.get("seed", 0)))}
    output = run_round_scheduler(
        scheduler, candidates, summary=summary, bohb_config=bohb,
        strategy_config=config.get("round_strategy") or {},
        invocation_id=action.get("task_key") or f"bohb-{state.get('event_index', 0)}",
        evaluator=context.get("bohb_evaluator"), proposed_strategy=strategy,
    )
    scheduler = output["state"]; state["round_scheduler"] = scheduler
    state["active_round"] = deepcopy(scheduler.get("active_round"))
    if state.get('branch_batch', {}).get('status') == 'hull_ready':
        terminal = 'completed' if output.get('status') == 'budget_exhausted' else 'hb_active'
        state['branch_batch'] = transition_branch_batch(state['branch_batch'], terminal,
                                                        strategy_version=output.get('strategy_version'))
    elif state.get('branch_batch', {}).get('status') == 'hb_active' and output.get('status') == 'budget_exhausted':
        state['branch_batch'] = transition_branch_batch(state['branch_batch'], 'completed')
    pending_keys = {item["task_key"] for item in ((scheduler.get("active_round") or {}).get("bohb_state") or {}).get("pending_tasks", [])}
    accepted, rejected = [], []
    for child in output.get("actions", []):
        if child["task_key"] not in pending_keys:
            continue
        reservation = reserve_budget(state, task_key=child["task_key"], stage="deep_search", amount=float(child["planned_relative_cost"]), limits=config["budgets"])
        if reservation["status"] != "reserved":
            rejected.append({"action": child, "reasons": reservation["reasons"]})
            _drop_bohb_pending(state, child["task_key"])
            continue
        state = reservation["state"]
        task = {**deepcopy(child), "task_id": f"BOHB-{hashlib.sha256(child['task_key'].encode()).hexdigest()[:12]}", "object_id": child["branch_id"], "status": "pending"}
        state.setdefault("tasks", []).append(task); state.setdefault("pending_tasks", []).append(task)
        accepted.append(task)
    return {"status": "completed", "state": state, "bohb_status": output["status"], "actions": accepted, "rejected_actions": rejected, "strategy_version": output.get("strategy_version")}


def _run_calculation_stage(action, context, stage_registry):
    params = deepcopy(action.get("parameters") or {}); target = (action.get("target_ids") or [params.get("structure_id")])[0]
    task = {
        "task_id": params.get("task_id") or f"TASK-{hashlib.sha256(action['task_key'].encode()).hexdigest()[:12]}",
        "task_key": action["task_key"], "object_id": target, "structure_id": target,
        "stage": action.get("stage") or params.get("stage"), "parameters": params.get("calculation_parameters") or {},
        "budget": params.get("stage_budget") or {}, "checkpoint": params.get("checkpoint"),
        "config_version": context.get("config_version"),
    }
    dispatcher = context.get("dispatcher")
    if dispatcher is not None:
        return dispatcher(task)
    factory = context.get("stage_context_factory")
    if factory is None:
        return {**task, "status": "not_configured", "error": "dispatcher or stage_context_factory required"}
    return execute_registered_stage(task, factory(task), stage_registry)


def _default_region(branch):
    return branch.get("region_id") or f"{branch.get('P')}:x={branch.get('x')}"


def _branch_atom_count(manager, branch):
    values = []
    for structure_id in branch.get("structure_ids") or []:
        record = manager.data.get("structures", {}).get(structure_id, {})
        value = (record.get("metadata") or {}).get("atom_count") or record.get("atom_count")
        if value: values.append(int(value))
    if values: return max(values)
    composition = branch.get("composition") or {}
    total = sum(float(value) for value in composition.values())
    return max(1, round(total)) if total else 40


def _candidate_version(candidates):
    payload = [(item["branch_id"], item.get("region_id"), item.get("atom_count")) for item in candidates]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _hull_version(state):
    diagrams = state.get("phase_diagrams") or {}
    for method in ("dft", "mlip"):
        if (diagrams.get(method) or {}).get("version"):
            return diagrams[method]["version"]
    return "uninitialized"


def _drop_bohb_pending(state, task_key):
    active = (state.get("round_scheduler") or {}).get("active_round") or {}
    bohb_state = active.get("bohb_state") or {}
    bohb_state["pending_tasks"] = [item for item in bohb_state.get("pending_tasks", []) if item.get("task_key") != task_key]
