"""Role authority over existing operations, not another Action or Tool registry."""

ADVISORY_OPERATIONS = {
    "exploration": frozenset(
        {
            "generate_branches",
            "allocate_mlip_mc",
            "prepare_mlip_relax",
            "prepare_mlip_mc",
            "prepare_relax_mc_batch",
            "prepare_ordering_search",
        }
    ),
    "reliability": frozenset(
        {
            "select_dft_candidates",
            "sample_direct_extrapolation",
            "prepare_perturbation_recovery",
            "prepare_parent_first_recovery",
            "prepare_dft_inputs",
            "prepare_mlip_training",
            "prepare_training_handoff",
            "prepare_model_refresh",
            "prepare_mc_energy_repair",
        }
    ),
}
RECOMMENDATIONS = {
    "exploration": ("explore", "deepen", "defer", "insufficient_evidence"),
    "reliability": ("dft", "finetune", "validate", "defer", "insufficient_evidence"),
}
ROLES = ("exploration", "reliability", "strategic", "simulation")


def advisory_roles(operation):
    return tuple(
        role for role, operations in ADVISORY_OPERATIONS.items() if operation in operations
    )


def role_permissions(role, *, available_operations=None, registered_tools=()):
    if role not in ROLES:
        raise ValueError("unknown_agent_role")
    if available_operations is None:
        from phase_agent.tools.dispatch.scientific_operations import OPERATIONS

        available_operations = OPERATIONS
    operations = set(available_operations)
    return {
        "role": role,
        "read_scopes": ["scientific_state", "approved_memory"]
        + (["expert_advice", "execution_records"] if role in {"strategic", "simulation"} else []),
        "advisory_operations": sorted(operations & ADVISORY_OPERATIONS.get(role, frozenset())),
        "final_proposal_operations": sorted(operations) if role == "strategic" else [],
        "execution_tools": sorted(registered_tools) if role == "simulation" else [],
        "may_propose_final_action": role == "strategic",
        "may_execute": role == "simulation",
        "authority": "Advisory proposals grant no execution permission. Confirmed policy, frozen approval, validation and receipts govern execution.",
    }


def advisory_option_errors(role, option):
    from phase_agent.tools.dispatch.scientific_operations import OPERATIONS, bind_operation

    if role not in ADVISORY_OPERATIONS:
        return ["role has no specialist proposal permission"]
    tool, operation, params = option.tool, option.operation, option.parameters
    if operation is not None:
        if operation not in ADVISORY_OPERATIONS[role]:
            return ["operation is outside specialist role permission"]
        try:
            bind_operation({"operation": operation, "tool": tool, "parameters": params})
        except (ValueError, TypeError) as error:
            return [str(error)]
        return []
    # Historical pause opinions mean scoped deferral, never global termination.
    if tool == "pause_search":
        return []
    if tool == "adjust_strategy" and role == "exploration":
        patch = params.get("patch") or {}
        sampling_fields = {
            "round_strategy.minimum_exploration_fraction",
            "round_strategy.maximum_mc_budget",
            "round_strategy.generation_quota_total",
            "round_strategy.allowed_generation_strategies",
            "round_strategy.rule_default.focus_regions",
            "round_strategy.rule_default.generation_quotas",
            "round_strategy.rule_default.mc_budget",
            "run.total_quota",
            "run.batch_size",
            "run.initial_states_per_branch",
            "run.seed",
        }
        prefixes = ("generation_actions.", "round_strategy.rule_default.generation_quotas.")
        return (
            []
            if isinstance(patch, dict)
            and patch
            and all(
                isinstance(key, str) and (key in sampling_fields or key.startswith(prefixes))
                for key in patch
            )
            else ["sampling adjustment is outside specialist role scope"]
        )
    for name in ADVISORY_OPERATIONS[role]:
        spec = OPERATIONS[name]
        if spec["tool"] != tool:
            continue
        fixed = spec["parameters"]
        if tool == "prepare_local_batch_files" and params.get("mode") != fixed.get("mode"):
            continue
        if any(key in params and params[key] != value for key, value in fixed.items()):
            continue
        return []
    return ["tool proposal is outside specialist role permission"]


def execution_role_allowed(role):
    return role == "simulation"


def necessary_roles(action, *, state=None, context=None):
    """Conservative floor for new scientific choices, never an action router."""
    from phase_agent.configuration.schema.action_state_schema import normalize_action
    from phase_agent.agents.simulation.program_checks import dft_input_record

    action = normalize_action(action)
    context = context or {}
    tool, params = action["tool"], action.get("parameters") or {}
    both = {
        "exploration": "cross-stage or unknown scientific scope",
        "reliability": "cross-stage or unknown scientific scope",
    }
    point = (action.get("scientific_decision") or {}).get("point")
    if tool == "check_convergence" or action.get("post_dft_review") or point == "round_direction":
        required = dict(both)
    elif tool in {"generate_branches", "allocate_mc_bohb", "select_candidates"}:
        required = {"exploration": "new exploration scope or sampling allocation"}
    elif tool in {"select_dft_candidates", "update_mlip", "reevaluate_candidates"}:
        required = {"reliability": "DFT selection, model update or reliability judgement"}
    elif tool == "prepare_local_batch_files":
        fixed_state = (
            state
            if (state or {}).get("tasks") is not None
            else context.get("registered_dft_inputs") or {}
        )
        fixed = dft_input_record(action, fixed_state)
        if fixed is not None and fixed["status"] == "passed":
            required = {}
        elif params.get("mode") in {
            "parent_first_recovery_inputs",
            "perturbation_recovery_inputs",
            "direct_extrapolation",
            "model_refresh_inputs",
            "mc_energy_repair_inputs",
            "training_handoff_inputs",
            "dft_inputs",
        }:
            required = {"reliability": "recovery/model/input scope not proven fixed"}
        elif params.get("mode") in {
            "ordering_search_inputs",
            "relax_mc_inputs",
            "relax_inputs",
            "mc_inputs",
        }:
            required = {"exploration": "exploration input scope not proven fixed"}
        else:
            required = both
    elif tool == "pause_search":
        required = {}
    else:
        required = both
    domains = {
        role
        for step in (action.get("research_plan") or {}).get("steps") or []
        for role in advisory_roles(step.get("operation"))
    }
    if domains == {"exploration", "reliability"}:
        required.update(both)
    rounds = (context.get("reliability_extrapolation_rounds") or {}).get("rounds") or []
    if any(row.get("direct_sampling_required") is True for row in rounds):
        required["reliability"] = "unresolved extrapolation recovery risk"
    return required


def missing_role_reasons(action, advice, *, snapshot_id, state=None, context=None):
    required = necessary_roles(action, state=state, context=context)
    for opinion in advice:
        for role in opinion.get("handoff_roles") or []:
            required.setdefault(role, "specialist requested cross-domain review")
    answered = {
        row["role"]
        for row in advice
        if snapshot_id
        and row.get("snapshot_id") == snapshot_id
        and row.get("recommendation") != "insufficient_evidence"
    }
    return {role: reason for role, reason in required.items() if role not in answered}
