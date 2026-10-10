"""把正式 Agent tools 接到现有生成、BOHB 和计算实现。"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from phase_agent.decisions.strategy.choose_generation_strategy import choose_generation_strategy
from phase_agent.tools.dispatch.dispatch_calculation_stage import execute_registered_stage
from phase_agent.tools.workflows.run_branch_generation import run_branch_generation
from phase_agent.tools.workflows.run_round_scheduler import run_round_scheduler
from phase_agent.tools.budget.reserve_budget import reserve_budget
from phase_agent.science.bohb.build_bohb_candidate_pool import build_bohb_candidate_pool
from phase_agent.science.bohb.collect_bohb_results import collect_bohb_results
from phase_agent.tools.workflows.prepare_branch_relaxation import prepare_branch_relaxation
from phase_agent.tools.state.branch_batch_state import (
    ACTIVE_BRANCH_BATCH_STATES,
    create_branch_batch,
    transition_branch_batch,
)
from phase_agent.science.mc.schedule_tiered_mc import schedule_tiered_mc
from phase_agent.decisions.scoring.rank_branch_relax_prescreen import rank_branch_relax_prescreen
from phase_agent.tools.workflows.prepare_dedup_batch import prepare_dedup_batch


def create_active_learning_handlers(stage_registry):
    return {
        "generate_branches": _generate_branches,
        "prepare_dedup_batch": _prepare_dedup,
        "allocate_mc_bohb": _allocate_mc_bohb,
        "run_calculation_stage": lambda *, action, context: _run_calculation_stage(
            action, context, stage_registry
        ),
    }


def _prepare_dedup(*, action, context):
    state = deepcopy(context.get("event_state") or {})
    requested = set(action.get("target_ids") or [])
    candidates = []
    for structure_id, record in context["manager"].data.get("structures", {}).items():
        if requested and structure_id not in requested:
            continue
        metadata = record.get("metadata") or {}
        candidates.append(
            {
                "structure_id": structure_id,
                "periodic_search_space_id": metadata.get("periodic_search_space_id"),
            }
        )
    settings = context["effective_config"].get("dedup") or {}
    return prepare_dedup_batch(
        candidates,
        state,
        config_version=context["config_version"],
        model_version=state.get("active_model_version")
        or (context["effective_config"].get("mlip") or {}).get("version"),
        approved=True,
        budget_limit=settings.get("budget_limit"),
        resource_limit=settings.get("resource_limit"),
        budget_limits=context["effective_config"].get("budgets"),
    )


def _generate_branches(*, action, context):
    config = context["effective_config"]
    params = deepcopy(action.get("parameters") or {})
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment

    if post_dft_assessment(context.get("event_state") or {}, config) and not params.get(
        "generation_plan"
    ):
        return {
            "status": "not_configured",
            "state": context.get("event_state") or {},
            "reason": "DFT 后生成必须明确策略及目标相/Na分配；未生成结构。",
        }
    if params.get("generation_plan"):
        from phase_agent.decisions.agent.generation_plan import (
            derive_generation_totals,
            validate_generation_plan,
        )

        params = derive_generation_totals({"tool": "generate_branches", "parameters": params})[
            "parameters"
        ]
        validate_generation_plan(params)
    quotas = params.get("quotas")
    if params.get("generation_plan"):
        from phase_agent.analysis.cost.generation_preflight import attach_generation_preflight

        try:
            attach_generation_preflight(action, context.get("event_state") or {}, context, config)
        except ValueError as error:
            return {
                "status": "not_configured",
                "state": context.get("event_state") or {},
                "reason": str(error),
            }
    existing = context["manager"].data.get("branches") or {}
    from phase_agent.science.structures.boundary_utils import allowed_phases

    required_phases = allowed_phases(context["manager"].boundary["P"])
    missing_phases = required_phases - {row["P"] for row in existing.values()}
    if not existing and not params.get("generation_plan"):
        # Parent-based strategies cannot produce candidates in an empty ledger.
        total = (
            sum(quotas.values())
            if quotas is not None
            else int(params.get("total_quota", config.get("total_quota", 0)))
        )
        quotas = {"coverage": total}
    elif quotas is None and missing_phases:
        quotas = {"coverage": int(params.get("total_quota", config.get("total_quota", 0)))}
    elif quotas is None:
        quotas = choose_generation_strategy(
            context.get("event_state", {}).get("generation_metrics")
            or config.get("generation_metrics")
            or {},
            total_quota=int(params.get("total_quota", config.get("total_quota", 0))),
        )["quotas"]
    options = deepcopy(config.get("generation_options") or {})
    if "max_det_H" in params:
        options["max_det_H"] = params["max_det_H"]
    first_round_cap = ((config.get("system_config") or {}).get("H_generation") or {}).get(
        "first_round_max_det_H"
    )
    if not existing and first_round_cap is not None:
        options["max_det_H"] = first_round_cap
    generation_directory = config["structure_directory"]
    if config.get("upload_batches_directory"):
        from phase_agent.tools.remote.model_upload_directory import model_upload_directory

        layout_state = deepcopy(context.get("event_state") or {})
        version = (
            layout_state.get("active_model_version")
            or (config.get("mlip") or {}).get("version")
            or (config.get("mlip") or {}).get("name")
        )
        group_index = len(layout_state.get("generation_history") or []) + 1
        generation_directory = str(
            model_upload_directory(config["upload_batches_directory"], layout_state, version)
            / f"Search-group-{group_index:04d}"
            / "Branch-0001"
        )
    result = run_branch_generation(
        context["manager"],
        context["phase_references"],
        structure_directory=generation_directory,
        quotas=quotas,
        batch_size=int(params.get("batch_size", config["batch_size"])),
        initial_states_per_branch=int(
            params.get("initial_states_per_branch", config["initial_states_per_branch"])
        ),
        seed=int(params.get("seed", config["seed"]))
        + 100_000 * len((context.get("event_state") or {}).get("generation_history") or []),
        ledger_path=config.get("ledger_path"),
        system_config=config.get("system_config"),
        **options,
        generation_plan=params.get("generation_plan"),
    )
    state = deepcopy(context.get("event_state") or {})
    state.setdefault("generation_history", []).append(
        {
            "task_key": action.get("task_key"),
            "quotas": deepcopy(quotas),
            "generation_plan": deepcopy(params.get("generation_plan")),
            "model_version": state.get("active_model_version")
            or (config.get("mlip") or {}).get("version")
            or (config.get("mlip") or {}).get("name"),
            "search_group_index": len(state.get("generation_history") or []) + 1,
            "registered_ids": [item.get("structure_id") for item in result["registered"]],
            "coverage": deepcopy(result["coverage"]),
            "summary": deepcopy(result["summary"]),
            "structure_directory": generation_directory,
        }
    )
    state["coverage"] = deepcopy(result["coverage"])
    gate = state.get("dedup_gate") or {}
    if gate.get("status") in {None, "ready"}:
        ids = set(gate.get("valid_structure_ids") or [])
        ids.update(item["structure_id"] for item in result["registered"])
        state["dedup_gate"] = {
            "status": "ready",
            "source": "local_generation_dedup",
            "valid_structure_ids": sorted(ids),
        }
    return {
        "status": "completed",
        "state": state,
        "registered": result["registered"],
        "coverage": result["coverage"],
        "summary": result["summary"],
    }


def _allocate_mc_bohb(*, action, context):
    from phase_agent.decisions.agent.mc_contracts import mc_contract_errors

    errors = mc_contract_errors(
        action.get("parameters", {}), fallback_budget=action.get("budget", 0)
    )
    if errors:
        return {
            "status": "rejected",
            "state": deepcopy(context.get("event_state") or {}),
            "reason": "invalid_mc_allocation_parameters",
            "errors": errors,
        }
    config = context["effective_config"]
    params = deepcopy(action.get("parameters") or {})
    state = deepcopy(context.get("event_state") or {})
    from phase_agent.science.mc.second_round_state import (
        first_round_source,
        reconciled_mc_state,
        second_round_candidates,
        second_round_already_allocated,
    )

    second_round = params.get("round_kind") == "second"
    has_existing_mc = any(row.get("stage") == "deep_search" for row in state.get("tasks") or [])
    if has_existing_mc and not second_round:
        return {
            "status": "not_configured",
            "state": state,
            "reason": "existing_mc_results_require_verified_second_round_proposal",
        }
    if second_round:
        model_version = (config.get("mlip") or {}).get("version") or (config.get("mlip") or {}).get(
            "name"
        )
        if second_round_already_allocated(
            state, model_version, params.get("first_round_source_checksum")
        ):
            return {
                "status": "not_configured",
                "state": state,
                "reason": "second_mc_round_already_allocated",
            }
        source = first_round_source(state, model_version)
        if not (config.get("mc_policy") or {}).get("second_segment_enabled", True):
            return {
                "status": "not_configured",
                "state": state,
                "reason": "second_mc_segment_disabled_in_confirmed_config",
            }
        if source is None or source["checksum"] != params.get("first_round_source_checksum"):
            return {
                "status": "awaiting_approval",
                "state": state,
                "reason": "first_round_mc_results_changed_since_approval",
            }
    preview = params.get("budget_preview") or {}
    if (
        preview.get("compression_requires_user_choice")
        and params.get("compression_choice") != "reduce_branch_count"
    ):
        return {
            "status": "awaiting_approval",
            "state": state,
            "reason": "choose_full_plan_or_compression",
            "budget_preview": preview,
        }
    region_builder = context.get("region_builder") or _default_region
    candidates = build_bohb_candidate_pool(
        context["manager"],
        feature_builder=context.get("bohb_feature_builder"),
        group_builder=context.get("bohb_group_builder"),
        region_builder=region_builder,
    )
    requested = list(dict.fromkeys(action.get("target_ids") or []))
    lifecycle = state.get("branch_batch") or {}
    frozen_batch = (
        lifecycle
        if lifecycle.get("status") in ACTIVE_BRANCH_BATCH_STATES
        else (state.get("pending_branch_screening") or {})
    )
    frozen_ids = list(frozen_batch.get("branch_ids") or [])
    if frozen_ids and not second_round:
        if requested and set(requested) != set(frozen_ids):
            return {
                "status": "not_configured",
                "state": state,
                "error": "agent_branch_batch_changed_during_screening",
                "required_branch_ids": frozen_ids,
            }
        requested = frozen_ids
    if second_round and (not requested or not set(requested).issubset(source["branch_ids"])):
        return {
            "status": "not_configured",
            "state": state,
            "reason": "second_mc_round_requires_completed_first_round_branches",
        }
    known = {item["branch_id"] for item in candidates}
    unknown = sorted(set(requested) - known)
    if unknown:
        return {
            "status": "not_configured",
            "state": state,
            "error": "unknown_agent_branch_ids",
            "unknown_branch_ids": unknown,
        }
    if requested:
        selected = set(requested)
        candidates = [item for item in candidates if item["branch_id"] in selected]
    else:
        focus = set(params.get("focus_regions") or [])
        if focus:
            candidates = [
                item
                for item in candidates
                if set(item.get("region_ids") or [item.get("region_id")]) & focus
            ]
    if not candidates:
        return {"status": "not_configured", "state": state, "error": "agent_branch_batch_empty"}
    prescreen = rank_branch_relax_prescreen(
        candidates,
        count=params.get("max_relax_branches"),
        seed=int(params.get("seed", config.get("seed", 0))),
        random_fraction=float(
            params.get(
                "exploration_fraction",
                (config.get("mc_policy") or {}).get("random_exploration_fraction", 0.1),
            )
        ),
    )
    candidates = prescreen["selected"]
    if not candidates:
        return {
            "status": "not_configured",
            "state": state,
            "error": "no_legal_branch_after_relax_prescreen",
            "prescreen": prescreen,
        }
    uses_relax_screening = (config.get("bohb") or {}).get(
        "selection_policy"
    ) == "relax_hull_uncertainty"
    if uses_relax_screening and not frozen_ids and not second_round:
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
    if second_round:
        diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
        approved_diagram_version = params.get("phase_diagram_version") or (
            params.get("budget_preview") or {}
        ).get("phase_diagram_version")
        if (
            diagram.get("status") != "completed"
            or diagram.get("model_version") != model_version
            or not approved_diagram_version
            or diagram.get("version") != approved_diagram_version
        ):
            return {
                "status": "awaiting_approval",
                "state": context.get("event_state") or {},
                "reason": "mc_phase_diagram_changed_since_approval",
            }
        approved_hull_version = params.get("hull_reference_version")
        pool = (state.get("branch_hull_batches") or {}).get(approved_hull_version)
        if (
            not approved_hull_version
            or not pool
            or pool.get("version") != approved_hull_version
            or pool.get("model_version") != model_version
        ):
            return {
                "status": "awaiting_approval",
                "state": context.get("event_state") or {},
                "reason": "mc_hull_reference_changed_since_approval",
            }
        candidates, missing_mc = second_round_candidates(candidates, state, diagram, model_version)
        if missing_mc or not candidates:
            return {
                "status": "not_configured",
                "state": state,
                "reason": "second_mc_round_requires_current_identified_mc_hull_entries",
                "missing_branch_ids": missing_mc,
            }
        candidates = [{**row, "phase_diagram_version": diagram["version"]} for row in candidates]
        screening = {
            "state": state,
            "status": "ready",
            "candidates": candidates,
            "pool": pool,
            "unavailable": [],
            "unavailable_branches": [],
        }
        bohb.setdefault("scope", {})["hull_reference_version"] = approved_hull_version
        bohb["scope"]["mlip_version"] = model_version
        bohb["objective"] = {**bohb.get("objective", {}), "name": "relaxed_batch_hull_gap"}
    elif bohb.get("selection_policy") == "relax_hull_uncertainty":
        screening = prepare_branch_relaxation(
            candidates,
            state,
            {
                **context,
                "mc_hull_reference_version": params.get("hull_reference_version"),
                "mc_phase_diagram_version": params.get("phase_diagram_version"),
            },
        )
        state = screening["state"]
        if screening["status"] not in {"ready", "ready_partial"}:
            target_status = (
                "relax_pending" if screening["status"] == "screening_pending" else "failed"
            )
            batch = state.get("branch_batch")
            if batch and batch.get("status") != target_status:
                state["branch_batch"] = transition_branch_batch(
                    batch, target_status, unavailable=deepcopy(screening.get("unavailable") or [])
                )
            return {
                "status": "completed"
                if screening["status"] == "screening_pending"
                else "not_configured",
                "state": state,
                "screening": screening["status"],
                "reason": screening.get("reason"),
                "unavailable": screening["unavailable"],
                "unavailable_branches": screening.get("unavailable_branches", []),
            }
        candidates = screening["candidates"]
        batch = state.get("branch_batch")
        if batch and batch.get("status") == "selected":
            batch = transition_branch_batch(batch, "relax_pending")
        if batch and batch.get("status") == "relax_pending":
            batch = transition_branch_batch(batch, "relax_completed")
        if batch and batch.get("status") == "relax_completed":
            batch = transition_branch_batch(
                batch,
                "hull_ready",
                hull_version=screening["pool"]["version"],
                screening_status=screening["status"],
                unavailable_structures=deepcopy(screening.get("unavailable") or []),
                unavailable_branches=deepcopy(screening.get("unavailable_branches") or []),
            )
        if batch:
            state["branch_batch"] = batch
        state.pop("pending_branch_screening", None)
        bohb.setdefault("scope", {})["hull_reference_version"] = screening["pool"]["version"]
        bohb["scope"]["mlip_version"] = screening["pool"]["model_version"]
        bohb["objective"] = {**bohb["objective"], "name": "relaxed_batch_hull_gap"}
    bohb.setdefault("budget_limits", config["budgets"])
    bohb.setdefault("scope", {})
    bohb["scope"] = {
        "mlip_version": bohb["scope"].get("mlip_version")
        or (config.get("mlip") or {}).get("version")
        or (config.get("mlip") or {}).get("name")
        or "unconfigured",
        "hull_reference_version": bohb["scope"].get("hull_reference_version")
        or _hull_version(state),
        "candidate_set_version": bohb["scope"].get("candidate_set_version")
        or _candidate_version(candidates),
        "phase_diagram_version": ((state.get("phase_diagrams") or {}).get("mlip") or {}).get(
            "version"
        ),
    }
    current_diagram_version = bohb["scope"].get("phase_diagram_version")
    if (
        params.get("budget_preview")
        and params["budget_preview"].get("phase_diagram_version") != current_diagram_version
    ):
        return {
            "status": "awaiting_approval",
            "state": context.get("event_state") or {},
            "reason": "mc_phase_diagram_changed_since_approval",
        }
    if any(
        row.get("ehull_source") != "phase_diagram"
        or row.get("phase_diagram_version") != current_diagram_version
        or row.get("relaxed_ehull") is None
        or row.get("relaxed_ehull_unit") != "eV/atom"
        for row in candidates
    ):
        return {
            "status": "not_configured",
            "state": state,
            "reason": "mc_requires_current_phase_diagram_ehull_per_atom",
        }
    if not bohb.get("bohb_selection_interface_enabled", False):
        tier_state = reconciled_mc_state(state)
        total_mc_budget = int(params.get("mc_budget", action.get("budget", 0)))
        mc_policy = config.get("mc_policy") or {
            "tiers": [
                {
                    "name": str(value),
                    "max_mc_steps": value,
                    "patience_steps": max(1, value // 3),
                    "min_improvement": 0.001,
                }
                for value in bohb.get("budget_levels", [10, 30, 90])
            ],
            "max_segments_per_branch": 3,
            "max_cumulative_cost_per_branch": 90,
            "random_exploration_fraction": bohb.get("random_fraction", 0.1),
        }
        scheduled = schedule_tiered_mc(
            candidates,
            tier_state,
            policy=mc_policy,
            total_budget=total_mc_budget,
            seed=int(params.get("seed", config.get("seed", 0))),
            model_version=bohb["scope"]["mlip_version"],
            hull_reference_version=bohb["scope"]["hull_reference_version"],
            phase_diagram_version=(state.get("phase_diagrams") or {})
            .get("mlip", {})
            .get("version"),
        )
        tier_state = scheduled["state"]
        from phase_agent.tools.budget.estimate_stage_cost import estimate_stage_cost

        candidate_atoms = {row["branch_id"]: row.get("atom_count") for row in candidates}
        costs = {
            child["task_key"]: estimate_stage_cost(
                "deep_search",
                atom_count=candidate_atoms[child["branch_id"]],
                mc_steps=child["max_mc_steps"],
                budgets=config["budgets"],
            )["value"]
            for child in scheduled["actions"]
        }
        preview = params.get("budget_preview")
        if preview:
            from phase_agent.decisions.strategy.estimate_branch_mc_budget import (
                estimate_branch_mc_budget,
            )

            checked = estimate_branch_mc_budget(
                candidates,
                screening["pool"],
                state,
                config,
                step_limit=total_mc_budget,
                seed=int(params.get("seed", config.get("seed", 0))),
            )
            if checked["allocation_checksum"] != preview.get("allocation_checksum"):
                return {
                    "status": "awaiting_approval",
                    "state": context.get("event_state") or {},
                    "reason": "mc_allocation_changed_since_approval",
                    "budget_preview": checked,
                }
            costs = {
                row["task_key"]: row["planned_relative_cost"] for row in checked["allocations"]
            }
        if second_round and (
            not preview
            or preview.get("round_kind") != "second"
            or preview.get("first_round_source_checksum") != source["checksum"]
            or any(child.get("segment_index") != 1 for child in scheduled["actions"])
        ):
            return {
                "status": "awaiting_approval",
                "state": context.get("event_state") or {},
                "reason": "second_mc_round_allocation_changed_since_approval",
            }
        for child in scheduled["actions"]:
            child["planned_relative_cost"] = costs[child["task_key"]]
        for child in tier_state["segments"]:
            if child["task_key"] in costs:
                child["planned_relative_cost"] = costs[child["task_key"]]
        full_plan_approved = (
            bool(context.get("human_approved_mc_full_plan"))
            and bool(preview)
            and len(scheduled["actions"]) == preview.get("selected_branch_count")
            and preview.get("selected_branch_count") == preview.get("full_plan_branch_count")
            and preview.get("requested_steps") == preview.get("full_plan_steps")
        )
        reservation_limits = deepcopy(config["budgets"])
        if full_plan_approved:
            active_reservations = [
                row
                for row in (state.get("budget_reservations") or {}).values()
                if row.get("status") in {"reserved", "submitted", "running"}
            ]
            existing_total = sum(float(row.get("reserved_cost", 0)) for row in active_reservations)
            existing_mc = [row for row in active_reservations if row.get("stage") == "deep_search"]
            approved_cost = sum(float(row["planned_relative_cost"]) for row in scheduled["actions"])
            stage_limit = reservation_limits.setdefault("stage_limits", {}).setdefault(
                "deep_search", {}
            )
            usage = state.get("budget_usage") or {}
            stage_usage = (usage.get("stages") or {}).get("deep_search") or {}
            reservation_limits["total_relative_cost"] = max(
                float(reservation_limits["total_relative_cost"]),
                float(usage.get("total_relative_cost", 0)) + existing_total + approved_cost,
            )
            stage_limit["max_cost"] = max(
                float(stage_limit.get("max_cost") or 0),
                float(stage_usage.get("cost", 0))
                + sum(float(row.get("reserved_cost", 0)) for row in existing_mc)
                + approved_cost,
            )
            stage_limit["max_tasks"] = max(
                int(stage_limit.get("max_tasks") or 0),
                int(stage_usage.get("tasks", 0)) + len(existing_mc) + len(scheduled["actions"]),
            )
            state.setdefault("approved_mc_budget_overrides", []).append(
                {
                    "allocation_checksum": preview["allocation_checksum"],
                    "task_count": len(scheduled["actions"]),
                    "estimated_relative_cost": approved_cost,
                    "configured_limits": deepcopy(config["budgets"]),
                    "reason": "explicit_full_plan_approval",
                }
            )
        from phase_agent.tools.budget.stratify_mc_actions import (
            interleave_mc_strata,
            summarize_mc_interception,
        )

        branch_phase = {row.get("branch_id"): row.get("P") for row in candidates}
        ordered_actions = (
            scheduled["actions"]
            if full_plan_approved
            else interleave_mc_strata(scheduled["actions"], branch_phase)
        )
        accepted, rejected = [], []
        for child in ordered_actions:
            reservation = reserve_budget(
                state,
                task_key=child["task_key"],
                stage="deep_search",
                amount=float(child["planned_relative_cost"]),
                limits=reservation_limits,
                config_version=context["config_version"],
                model_version=child["model_version"],
            )
            if reservation["status"] != "reserved":
                rejected.append({"action": child, "reasons": reservation["reasons"]})
                tier_state["segments"] = [
                    row
                    for row in tier_state.get("segments", [])
                    if row.get("task_key") != child["task_key"]
                ]
                continue
            state = reservation["state"]
            if full_plan_approved:
                state["budget_reservations"][child["task_key"]]["approval_override"] = preview[
                    "allocation_checksum"
                ]
            task = {
                **deepcopy(child),
                "task_id": f"MC-{hashlib.sha256(child['task_key'].encode()).hexdigest()[:12]}",
                "object_id": child["branch_id"],
                "status": "pending",
                "generation_cycle": len(state.get("generation_history") or []),
                "parent_decision_id": context.get("approval_record_id"),
                "upload_operation_id": hashlib.sha256(
                    str(
                        (preview or {}).get("allocation_checksum") or action.get("task_key")
                    ).encode()
                ).hexdigest()[:12],
                "parameters": {
                    "max_mc_steps": child["max_mc_steps"],
                    "patience_steps": child["patience_steps"],
                    "min_improvement": child["min_improvement"],
                    "seed": child["seed"],
                },
            }
            state.setdefault("tasks", []).append(task)
            state.setdefault("pending_tasks", []).append(task)
            accepted.append(task)
        state["tiered_mc_state"] = tier_state
        if second_round and accepted:
            allocation_record = {
                "source_checksum": source["checksum"],
                "allocation_checksum": preview["allocation_checksum"],
                "task_ids": [row["task_id"] for row in accepted],
                "model_version": model_version,
                "phase_diagram_version": current_diagram_version,
            }
            state.setdefault("mc_second_round_allocations", []).append(allocation_record)
            state["mc_second_round_allocation"] = allocation_record
        if rejected:
            state["mc_budget_interception"] = summarize_mc_interception(
                accepted, rejected, branch_phase, full_plan_approved=full_plan_approved
            )
        upload = None
        if accepted and context.get("execution_mode") == "interactive":
            from phase_agent.tools.local.prepare_mc_upload_batches import prepare_mc_upload_batches

            try:
                upload = prepare_mc_upload_batches(
                    action={"parameters": {"mode": "mc_inputs"}},
                    context={**context, "event_state": state},
                )
            except Exception as error:
                upload = {
                    "status": "not_configured",
                    "reason": f"mc_input_preparation_failed:{error}",
                }
            state = upload.get("state", state)
            if upload.get("status") not in {"prepared", "already_prepared"}:
                return {
                    "status": "not_configured",
                    "state": state,
                    "reason": upload.get("reason") or "mc_inputs_not_prepared",
                    "actions": accepted,
                    "mc_status": scheduled["status"],
                    "mc_upload": {key: value for key, value in upload.items() if key != "state"},
                }
        return {
            "status": "completed",
            "state": state,
            "method": scheduled["method"],
            "mc_status": scheduled["status"],
            "actions": accepted,
            "rejected_actions": rejected,
            "mc_upload": {key: value for key, value in (upload or {}).items() if key != "state"},
            "excluded_candidates": scheduled.get("excluded_candidates", []),
        }
    scheduler = deepcopy(state.get("round_scheduler") or {})
    active = scheduler.get("active_round") or {}
    bohb_state = active.get("bohb_state")
    if bohb_state and bohb_state.get("pending_tasks"):
        pending_keys = {item["task_key"] for item in bohb_state["pending_tasks"]}
        recovered = [
            item
            for item in state.get("tasks", [])
            if item.get("task_key") in pending_keys
            and item.get("status") not in {"pending", "running"}
        ]
        if recovered:
            active["bohb_state"] = collect_bohb_results(
                bohb_state, candidates, recovered, config=active["bohb_config"]
            )
            scheduler["active_round"] = active
    total_mc_budget = int(params.get("mc_budget", action.get("budget", 0)))
    strategy = {
        "focus_regions": list(params.get("focus_regions") or []),
        "generation_quotas": deepcopy(params.get("generation_quotas") or {}),
        "mc_budget": total_mc_budget,
        "dft_budget": float(params.get("dft_budget", 0)),
        "exploration_fraction": float(
            params.get(
                "exploration_fraction",
                config.get("round_strategy", {}).get("minimum_exploration_fraction", 0.1),
            )
        ),
        "reason": action.get("reason") or "Agent-selected branch batch",
        "source": action.get("decision_source", "agent_tool_action"),
    }
    summary = {
        **state,
        "known_region_ids": sorted(
            {item.get("region_id") for item in candidates if item.get("region_id") is not None}
        ),
        "seed": int(params.get("seed", config.get("seed", 0))),
    }
    output = run_round_scheduler(
        scheduler,
        candidates,
        summary=summary,
        bohb_config=bohb,
        strategy_config=config.get("round_strategy") or {},
        invocation_id=action.get("task_key") or f"bohb-{state.get('event_index', 0)}",
        evaluator=context.get("bohb_evaluator"),
        proposed_strategy=strategy,
    )
    scheduler = output["state"]
    state["round_scheduler"] = scheduler
    state["active_round"] = deepcopy(scheduler.get("active_round"))
    if not second_round:
        if state.get("branch_batch", {}).get("status") == "hull_ready":
            terminal = "completed" if output.get("status") == "budget_exhausted" else "hb_active"
            state["branch_batch"] = transition_branch_batch(
                state["branch_batch"], terminal, strategy_version=output.get("strategy_version")
            )
        elif (
            state.get("branch_batch", {}).get("status") == "hb_active"
            and output.get("status") == "budget_exhausted"
        ):
            state["branch_batch"] = transition_branch_batch(state["branch_batch"], "completed")
    pending_keys = {
        item["task_key"]
        for item in ((scheduler.get("active_round") or {}).get("bohb_state") or {}).get(
            "pending_tasks", []
        )
    }
    accepted, rejected = [], []
    for child in output.get("actions", []):
        if child["task_key"] not in pending_keys:
            continue
        reservation = reserve_budget(
            state,
            task_key=child["task_key"],
            stage="deep_search",
            amount=float(child["planned_relative_cost"]),
            limits=config["budgets"],
        )
        if reservation["status"] != "reserved":
            rejected.append({"action": child, "reasons": reservation["reasons"]})
            _drop_bohb_pending(state, child["task_key"])
            continue
        state = reservation["state"]
        task = {
            **deepcopy(child),
            "task_id": f"BOHB-{hashlib.sha256(child['task_key'].encode()).hexdigest()[:12]}",
            "object_id": child["branch_id"],
            "status": "pending",
        }
        state.setdefault("tasks", []).append(task)
        state.setdefault("pending_tasks", []).append(task)
        accepted.append(task)
    return {
        "status": "completed",
        "state": state,
        "bohb_status": output["status"],
        "actions": accepted,
        "rejected_actions": rejected,
        "strategy_version": output.get("strategy_version"),
    }


def _run_calculation_stage(action, context, stage_registry):
    params = deepcopy(action.get("parameters") or {})
    target = (action.get("target_ids") or [params.get("structure_id")])[0]
    task = {
        "task_id": params.get("task_id")
        or f"TASK-{hashlib.sha256(action['task_key'].encode()).hexdigest()[:12]}",
        "task_key": action["task_key"],
        "object_id": target,
        "structure_id": target,
        "stage": action.get("stage") or params.get("stage"),
        "parameters": params.get("calculation_parameters") or {},
        "budget": params.get("stage_budget") or {},
        "checkpoint": params.get("checkpoint"),
        "config_version": context.get("config_version"),
    }
    dispatcher = context.get("dispatcher")
    if dispatcher is not None:
        return dispatcher(task)
    factory = context.get("stage_context_factory")
    if factory is None:
        return {
            **task,
            "status": "not_configured",
            "error": "dispatcher or stage_context_factory required",
        }
    return execute_registered_stage(task, factory(task), stage_registry)


def _default_region(branch):
    return branch.get("region_id") or f"{branch.get('P')}:x={branch.get('x')}"


def _branch_atom_count(manager, branch):
    values = []
    for structure_id in branch.get("structure_ids") or []:
        record = manager.data.get("structures", {}).get(structure_id, {})
        value = (record.get("metadata") or {}).get("atom_count") or record.get("atom_count")
        if value:
            values.append(int(value))
    if values:
        return max(values)
    composition = branch.get("composition") or {}
    total = sum(float(value) for value in composition.values())
    return max(1, round(total)) if total else 40


def _candidate_version(candidates):
    payload = [
        (item["branch_id"], item.get("region_id"), item.get("atom_count")) for item in candidates
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :12
    ]


def _hull_version(state):
    diagrams = state.get("phase_diagrams") or {}
    for method in ("dft", "mlip"):
        if (diagrams.get(method) or {}).get("version"):
            return diagrams[method]["version"]
    return "uninitialized"


def _drop_bohb_pending(state, task_key):
    active = (state.get("round_scheduler") or {}).get("active_round") or {}
    bohb_state = active.get("bohb_state") or {}
    bohb_state["pending_tasks"] = [
        item for item in bohb_state.get("pending_tasks", []) if item.get("task_key") != task_key
    ]
