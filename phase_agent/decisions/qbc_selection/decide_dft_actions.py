"""Ask the agent for categorical actions using only a StateSnapshot."""

from phase_agent.configuration.schema.action_state_schema import validate_state_schema


def decide_dft_actions(metrics: list[dict], context: dict, *, agent_client=None) -> dict:
    if agent_client is None:
        return {
            "status": "not_configured",
            "decisions": [],
            "global_action": "CONTINUE_DATA_COLLECTION",
            "source": "none",
            "error": "agent client not configured",
        }
    try:
        schema_errors = validate_state_schema(context)
        if schema_errors:
            raise ValueError("DFT Agent requires StateSnapshot: " + ",".join(schema_errors))
        output = agent_client(
            {
                "mode": "agent_dft_scheduler",
                "state": context,
                "uncertainty_metrics": metrics,
                "decision_context": context.get("decision_context") or {},
                "phase_diagram_state": context.get("current_convex_hull"),
                "training_coverage": context.get("search_coverage"),
                "remaining_dft_budget": context.get("available_budget"),
                "search_history": context.get("search_history"),
                "instruction": "Consult the StateSnapshot and cite evidence in reason. QBC values are observations only. Assign exactly one action per candidate: DFT_SINGLE_POINT, DFT_RELAX, DEFER, or REJECT. Optionally request RETRAIN_MLIP or CONTINUE_DATA_COLLECTION. Do not output or modify numeric scientific results or DFT parameters.",
            }
        )
        if not isinstance(output, dict) or not isinstance(output.get("decisions"), list):
            raise TypeError("agent output requires decisions list")
        return {
            "status": "completed",
            "decisions": output["decisions"],
            "global_action": output.get("global_action", "CONTINUE_DATA_COLLECTION"),
            "reason": output.get("reason"),
            "source": "llm_agent",
            "error": None,
        }
    except Exception as error:
        return {
            "status": "failed",
            "decisions": [],
            "global_action": "CONTINUE_DATA_COLLECTION",
            "source": "llm_agent",
            "error": f"{type(error).__name__}: {error}",
        }
