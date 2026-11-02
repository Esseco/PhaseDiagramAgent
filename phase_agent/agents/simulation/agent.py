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
    "A prepared input, submitted task or tool return is not scientific completion. JSON only."
)


def action_hash(action):
    normalized = normalize_action(action)
    return proposal_hash({k: v for k, v in normalized.items() if not k.startswith("_")})


def validate_plan(value, action):
    plan = SimulationPlan.model_validate(model_content(value))
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
    return plan.model_dump()


def _contains_intent(actual, requested):
    if isinstance(requested, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains_intent(actual[key], value)
            for key, value in requested.items()
            if not key.startswith("_")
        )
    return actual == requested


def validate_frozen_plan(action):
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
    action = normalize_action(action)
    return {
        "mode": "simulation_plan",
        "agent_role": "simulation",
        "allowed_tools": [],
        "action": {k: deepcopy(v) for k, v in action.items() if not k.startswith("_")},
        "action_hash": action_hash(action),
        "facts": deepcopy(facts or {}),
        "registered_operations": deepcopy(operations or []),
        "instruction": PLAN_INSTRUCTION,
        "output_contracts": {"simulation_plan": SimulationPlan.model_json_schema()},
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
        from phase_agent.evidence.project_authority import project_identity
        from phase_agent.tools.dispatch.scientific_operations import operation_catalog
        from phase_agent.persistence.memory.decision_context import decision_memory_context

        try:
            response = planner(
                planning_request(
                    action,
                    facts={
                        **project_identity(state),
                        "current_status": state.get("current_status"),
                        "budget_limits": config.get("budgets"),
                        "memory_context": decision_memory_context(state),
                        "human_guidance": (action.get("_decision_team") or {}).get("guidance")
                        or [],
                    },
                    operations=operation_catalog([action["tool"]], registry=registry),
                )
            )
        except Exception as error:
            _record_usage(state, getattr(error, "llm_usage", None))
            raise
        _record_usage(state, response.get("_llm_usage") if isinstance(response, dict) else None)
        plan = validate_plan(response, action)
    else:
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
            "cancelled",
        }
    )
    report = {
        "role": "simulation",
        "color": COLOR,
        "action_hash": action_hash(action),
        "tool_completed": not failed,
        "scientific_completion_claimed": False,
        "observations": {
            "execution.status": execution.get("status"),
            "execution.error": execution.get("error"),
            "execution.result.status": result_status,
            "execution.operation_outcome": deepcopy(outcome),
        },
        "next": "return_to_strategic" if failed else "observe_results",
    }
    if failed:
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
    if failed and (diagnostician := client_for(client)) is not None:
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
