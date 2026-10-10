"""Ask an LLM for a patch proposal, with no execution capability."""


def propose_config_revision(session: dict, *, agent_client=None) -> dict:
    prompt = session.get("default_parameter_prompt") or {}
    if prompt.get("status") == "awaiting_response":
        return {
            "status": "awaiting_default_parameter_choice",
            "question": prompt.get("question"),
            "default_parameters": prompt.get("default_parameters"),
            "source": "configuration_dialogue",
        }
    from phase_agent.configuration.schema.validate_search_config import validate_search_config

    audit = validate_search_config(session["config"])
    if agent_client is None:
        return {
            "status": "needs_user_input" if not audit["valid"] else "ready_for_confirmation",
            "patch": {},
            "reasons": audit,
            "source": "rule",
        }
    try:
        output = agent_client(
            {
                "mode": "configuration_dialogue",
                "config": session["config"],
                "audit": audit,
                "instruction": "Propose a JSON path patch. Separate hard constraints from adjustable suggestions. Do not run tools or submit calculations.",
            }
        )
        if not isinstance(output, dict) or not isinstance(output.get("patch", {}), dict):
            raise TypeError("invalid configuration proposal")
        return {
            "status": "proposal",
            "patch": output.get("patch", {}),
            "reasons": output.get("reasons", {}),
            "source": "llm_agent",
        }
    except Exception as error:
        return {
            "status": "needs_user_input",
            "patch": {},
            "reasons": audit,
            "source": "rule",
            "fallback_reason": f"{type(error).__name__}: {error}",
        }
