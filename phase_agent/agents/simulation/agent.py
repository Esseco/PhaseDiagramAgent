"""Plan, inspect and diagnose through the existing approved tool boundary."""

from copy import deepcopy
from phase_agent.agents.contracts import model_content
from phase_agent.agents.simulation.contracts import SimulationPlan, SimulationDiagnosis
from phase_agent.configuration.schema.action_state_schema import normalize_action
from phase_agent.policies.file_approval import proposal_hash

COLOR = "#616161"
PLAN_INSTRUCTION = (
    "You are Simulation Agent. Plan execution of exactly the supplied Strategic Action Proposal. "
    "Reason about prerequisites, registered tool invocation and result checks. Produce one preflight, "
    "exactly one invoke of action.tool, and verification steps. Preserve all approved targets, inputs, "
    "resources and budget; describe tasks but do not emit shell commands or alternative scientific "
    "actions. Execution failure must return to Strategic or human review, never auto-retry. "
    "Checks are proposed checks, not executed checks or measured success. "
    "Program preflight covers only its named validators; unknown inputs remain unknown. "
    "A prepared input, submitted task or tool return is not scientific completion. JSON only."
)


def action_hash(action):
    normalized = normalize_action(action)
    return proposal_hash({k: v for k, v in normalized.items() if not k.startswith("_")})


def validate_plan(value, action, *, expected_binding=None):
    plan = SimulationPlan.model_validate(model_content(value))
    if expected_binding is not None and (
        plan.context_binding is None
        or plan.context_binding.model_dump(exclude_none=True) != expected_binding
    ):
        raise ValueError("simulation_plan_stale_context; replan with current facts")
    if plan.action_hash != action_hash(action):
        raise ValueError("simulation_plan_stale_action; replan and request approval")
    invoke = [step for step in plan.steps if step.kind == "invoke"]
    if len(invoke) != 1 or invoke[0].tool != normalize_action(action)["tool"]:
        raise ValueError("simulation_plan_tool_scope_conflict")
    kinds = [step.kind for step in plan.steps]
    if (
        kinds[0] != "preflight"
        or kinds[-1] != "verify"
        or any(step.tool is not None for step in plan.steps if step.kind != "invoke")
        or any(kind != "preflight" for kind in kinds[: kinds.index("invoke")])
        or any(kind != "verify" for kind in kinds[kinds.index("invoke") + 1 :])
    ):
        raise ValueError("simulation_plan_invalid_task_order")
    result = plan.model_dump()
    if plan.context_binding is None:
        result.pop("context_binding")
    else:
        result["context_binding"] = plan.context_binding.model_dump(exclude_none=True)
    return result


def _contains_intent(actual, requested):
    if isinstance(requested, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_intent(actual[key], value)
            for key, value in requested.items()
            if not key.startswith("_")
        )
    return actual == requested


def validate_frozen_plan(action, *, state=None):
    if action.get("_simulation_program"):
        from phase_agent.agents.simulation.program_checks import validate_program

        intent = action.get("_simulation_intent") or action
        if not _contains_intent(action, intent) or action.get(
            "_simulation_execution_hash"
        ) != action_hash(action):
            raise ValueError("simulation_program_stale_action; request new approval")
        if state is None:
            raise ValueError("simulation_program_requires_current_state")
        return validate_program(action["_simulation_program"], action, state)
    plan = action["_simulation_plan"]
    intent = action.get("_simulation_intent")
    if intent is not None:
        validated = validate_plan(plan, intent)
        if not _contains_intent(action, intent):
            raise ValueError("simulation_intent_changed; request new planning and approval")
        if action.get("_simulation_execution_hash") != action_hash(action):
            raise ValueError("simulation_plan_stale_action; request new approval")
        return validated
    return validate_plan(plan, action)


def planning_request(action, *, facts=None, operations=None):
    from phase_agent.agents.common.action_contracts import action_contract_errors
    from phase_agent.agents.common.proposal_validation import prerequisite_errors
    from phase_agent.tools.dispatch.scientific_operations import operation_readiness

    action = normalize_action(action)
    decision = ((facts or {}).get("facts") or {}).get("decision_context") or {}
    errors = action_contract_errors(action) + prerequisite_errors(action, decision)
    if errors:
        raise ValueError("simulation_preflight_failed: " + "; ".join(errors))
    scoped_operations = [
        row
        for row in operations or []
        if row.get("tool") == action["tool"]
        and (action.get("operation") is None or row.get("operation") == action["operation"])
        and all(
            (action.get("parameters") or {}).get(key) == value
            for key, value in (row.get("parameters") or {}).items()
        )
        and (row.get("stage") is None or row["stage"] == action.get("stage"))
    ]
    consultation = (((facts or {}).get("facts") or {}).get("decision_context") or {}).get(
        "codex_consultation"
    ) or {}
    binding = {"snapshot_id": facts["snapshot_id"]} if (facts or {}).get("snapshot_id") else None
    if binding is not None and consultation.get("team_id"):
        binding.update(team_id=consultation["team_id"], revision=consultation["revision"])
    contract = SimulationPlan.model_json_schema()
    if binding is not None:
        contract["required"].append("context_binding")
    return {
        "mode": "simulation_plan",
        "agent_role": "simulation",
        "allowed_tools": [],
        "action": {k: deepcopy(v) for k, v in action.items() if not k.startswith("_")},
        "action_hash": action_hash(action),
        "context_binding": binding,
        "facts": deepcopy(facts or {}),
        "registered_operations": deepcopy(scoped_operations),
        "operation_readiness": operation_readiness(
            {"decision_context": decision, "scientific_operations": scoped_operations}
        ),
        "program_preflight": {
            "status": "passed",
            "validators": ["action_contract_errors", "prerequisite_errors"],
            "scope": "wire contract and known dependency gates only; not input/HPC verification or permission",
        },
        "execution_checks": {"status": "not_executed"},
        "scientific_completion": "unknown",
        "instruction": PLAN_INSTRUCTION
        + " Echo context_binding exactly when supplied; it binds this plan to these facts and this consultation.",
        "output_contracts": {"simulation_plan": contract},
    }


def client_for(client):
    if (
        client is None
        or getattr(client, "_codex_structured", False)
        or getattr(client, "_decision_source", None) == "codex"
    ):
        return None
    if (
        getattr(client, "_validates_original_facts", False)
        and getattr(client, "expert_client", client) is client
    ):
        return None
    return (
        getattr(client, "expert_client", client)
        if getattr(client, "supports_simulation", False) is True
        else None
    )


def prepare_proposal(proposal, client, *, state, config, registry):
    proposal = deepcopy(proposal)
    action = proposal["raw_action"]
    from phase_agent.policies.training_input_action import is_training_input_action

    if is_training_input_action(action):
        from phase_agent.policies.training_handoff_scope import proposed_scope

        action["_training_handoff_authorization"] = proposed_scope(action, config)
    if (action.get("parameters") or {}).get("mode") == "training_handoff_inputs":
        from phase_agent.tools.local.prepare_training_followup import request_binding

        action["_training_handoff_request"] = request_binding(action, state, config)
    from phase_agent.agents.simulation.program_checks import dft_input_record, validate_program

    program = action.get("_simulation_program")
    if not action.get("_simulation_plan"):
        intent = action.get("_simulation_intent") or action
        if program:
            validate_program(program, action, state)
            if not _contains_intent(action, intent):
                raise ValueError("simulation_program_intent_changed")
        else:
            program = dft_input_record(action, state)
        if program is not None:
            if program["status"] != "passed":
                raise ValueError(
                    "simulation_program_blocked: " + "; ".join(program["unmet_conditions"])
                )
            if action["tool"] not in registry:
                raise ValueError("simulation_program_tool_unavailable")
            action["_simulation_program"] = program
            action["_simulation_execution_hash"] = action_hash(action)
            proposal["simulation_program"] = deepcopy(program)
            return proposal
    if (action.get("parameters") or {}).get("repair_profile") == "dft_recalculation":
        from phase_agent.agents.simulation.dft_recalculation import review, validate_action

        params = action["parameters"]
        native = getattr(client, "_codex_structured", False) or getattr(client, "_decision_source", None) == "codex"
        if native:
            if not params.get("failure_evidence_digest") or not params.get("diagnosis"):
                raise ValueError("dft_recalculation_requires_bound_native_diagnosis")
        else:
            checked = review(state, action.get("target_ids") or [], client)
            _record_usage(state, checked.get("llm_usage"))
            if not checked.get("diagnosis"):
                raise ValueError("dft_recalculation_simulation_diagnosis_unavailable")
            params["failure_evidence_digest"] = checked["review"]["failure_evidence_digest"]
            params["diagnosis"] = checked["diagnosis"]
        validate_action(action, state)
    stored = action.get("_simulation_plan")
    if stored:
        if action.get("_simulation_intent") is not None:
            plan = validate_plan(stored, action["_simulation_intent"])
            if not _contains_intent(action, action["_simulation_intent"]):
                raise ValueError("simulation_intent_changed; request new planning and approval")
            # Native planning precedes program-owned previews/defaults. Freeze the
            # complete derived execution action before its separate human approval.
            action["_simulation_execution_hash"] = action_hash(action)
        else:
            plan = validate_plan(stored, action)
    elif (planner := client_for(client)) is not None:
        from phase_agent.evidence.snapshot import simulation_facts
        from phase_agent.tools.dispatch.scientific_operations import operation_catalog

        try:
            request = planning_request(
                action,
                facts=simulation_facts(action, state=state, config=config),
                operations=operation_catalog([action["tool"]], registry=registry),
            )
            response = planner(request)
        except Exception as error:
            _record_usage(state, getattr(error, "llm_usage", None))
            raise
        _record_usage(state, response.get("_llm_usage") if isinstance(response, dict) else None)
        plan = validate_plan(response, action, expected_binding=request["context_binding"])
    else:
        # Preserve native read-only evaluation/pause exemptions; the actual
        # convergence judgement still requires both scientific specialists.
        if action["tool"] in {"pause_search", "check_convergence"} and getattr(
            client, "_codex_structured", False
        ):
            return proposal
        if getattr(client, "supports_expert_advice", False) or getattr(
            client, "_codex_structured", False
        ):
            raise ValueError("simulation_model_capability_unavailable; no execution plan")
        # Historical callable interface remains available; it has never promised
        # model planning. Its scientific choices still meet the decision role guard.
        return proposal
    action["_simulation_plan"] = plan
    proposal["simulation_plan"] = deepcopy(plan)
    return proposal


def _record_usage(state, usage):
    if usage:
        from phase_agent.tools.budget.record_budget_usage import record_budget_usage

        state.update(record_budget_usage(state, {"llm_usage": usage}))


def inspect_execution(action, execution, client=None):
    """Diagnostics are advisory; no retry, scheduler or task mutation lives here."""
    result = execution.get("result")
    result_status = result.get("status") if isinstance(result, dict) else None
    outcome = execution.get("operation_outcome") or {}
    failed = (
        execution.get("status") != "completed"
        or not isinstance(result, dict)
        or result_status
        in {
            "failed",
            "rejected",
            "blocked",
            "not_configured",
            "reconciliation_required",
            "awaiting_approval",
            "approval_reconciliation_required",
            "execution_reconciliation_required",
            "insufficient_new_data",
            "budget_exhausted",
            "already_prepared",
            "cancelled",
        }
    )
    report = {
        "role": "simulation",
        "handling": "program" if action.get("_simulation_program") else "model_plan",
        "color": COLOR,
        "action_hash": action_hash(action),
        "tool_completed": not failed,
        "scientific_completion_claimed": False,
        "observations": {
            "execution.status": execution.get("status"),
            "execution.error": execution.get("error"),
            "execution.result.status": result_status,
            "execution.operation_outcome": deepcopy(outcome),
            "execution.result.error": deepcopy(result.get("error"))
            if isinstance(result, dict)
            else None,
            "execution.result.reason": deepcopy(result.get("reason"))
            if isinstance(result, dict)
            else None,
            "execution.result.validation_errors": deepcopy(result.get("validation_errors"))
            if isinstance(result, dict)
            else None,
        },
        "planned_checks": {
            "status": "not_evaluated",
            "items": deepcopy((action.get("_simulation_plan") or {}).get("checks", [])),
        },
        "next": "return_to_strategic" if failed else "observe_results",
    }
    # Explicit tool blockers already carry deterministic evidence. Unknown failures
    # (including exceptions) remain eligible for advisory model diagnosis.
    known_blocker = (
        isinstance(result, dict)
        and (
            result_status == "already_prepared"
            or (
                result_status in {
                    "blocked", "not_configured", "rejected", "awaiting_approval",
                    "insufficient_new_data", "budget_exhausted", "cancelled",
                }
                and bool(result.get("reason") or result.get("validation_errors"))
            )
        )
    )
    if known_blocker:
        report["diagnosis_source"] = "program"
        report["unmet_conditions"] = deepcopy(result.get("validation_errors") or [result.get("reason") or result_status])
    if failed and not known_blocker:
        request = {
            "mode": "simulation_diagnosis",
            "agent_role": "simulation",
            "allowed_tools": [],
            "action": {"tool": action["tool"], "task_key": action.get("task_key")},
            "observations": report["observations"],
            "instruction": "Diagnose only the supplied observations; cite their exact keys. Do not claim remote completion or execute/retry anything. Recommend Strategic/human review. JSON only.",
            "output_contracts": {"simulation_diagnosis": SimulationDiagnosis.model_json_schema()},
        }
        report["diagnosis_request"] = request
    if failed and not known_blocker and (diagnostician := client_for(client)) is not None:
        try:
            response = diagnostician(request)
            if isinstance(response, dict) and response.get("_llm_usage"):
                report["llm_usage"] = response["_llm_usage"]
            report["diagnosis"] = SimulationDiagnosis.model_validate(
                model_content(response)
            ).model_dump()
        except Exception as error:
            if getattr(error, "llm_usage", None):
                report["llm_usage"] = error.llm_usage
            report["diagnosis_unavailable"] = type(error).__name__
    return report
