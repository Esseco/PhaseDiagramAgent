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
            "prepare_dft_inputs",
            "prepare_mlip_training",
            "prepare_training_handoff",
            "prepare_model_refresh",
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
