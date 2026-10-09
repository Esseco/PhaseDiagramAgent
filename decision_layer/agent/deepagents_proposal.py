"""Opt-in analysis adapter. No network client construction or business tools."""
import json
from copy import deepcopy


def proposal_material(payload):
    """Explicit transport allowlist: summaries and reviewed references only."""
    context = payload.get("decision_context") or {}
    fields = {"status", "version", "energy_basis_id", "entry_count", "stable_count",
              "record_id", "candidate_id", "branch_id", "batch_id", "task_id", "phase",
              "composition", "ehull", "is_stable", "model_version", "energy_mae", "energy_rmse",
              "force_mae", "force_rmse", "mae", "rmse", "count", "completed_count",
              "actual_cost", "reward", "new_stable_entries", "ehull_improvement",
              "knowledge_id", "scope", "statement", "maturity", "evidence_refs", "items",
              "input_tokens", "output_tokens", "action", "tool", "reason", "target_ids"}
    fields.update({"mlip", "dft", "phase_coverage", "lowest_ehull_entries", "rewards", "actions",
                   "O3", "OP2", "P3", "O1", "total", "completed", "failed", "pending",
                   "energy", "energy_per_atom", "metrics", "retrieval_reason",
                   "matched_terms", "applicability", "min", "max"})
    fields.update({"post_dft_assessment", "comparison", "energy_metrics", "force_metrics",
                   "energy_mae_per_atom", "energy_rmse_per_atom", "force_component_mae",
                   "force_component_rmse", "paired_count", "eligible_count", "recovered_count",
                   "submitted_count", "available_budget", "budget_remaining", "relative_cost",
                   "atom_count", "x_Na_per_O2", "ehull_per_atom", "qbc", "uncertainty",
                   "single_point_max_per_round", "single_point_cost_per_round", "max_relax_fraction",
                   "minimum_new_dft_records", "finetune_enabled", "comparison_status"})

    def summary(value):
        if isinstance(value, dict):
            return {key: summary(item) for key, item in value.items() if key in fields}
        if isinstance(value, list):
            return [summary(item) for item in value]
        return deepcopy(value)

    material = {"instruction": payload.get("instruction", "Return one complete action JSON."),
            "allowed_tools": deepcopy(payload.get("allowed_tools") or []),
            "mode": payload.get("mode"),
            "decision_context": {key: summary(context[key]) for key in
                ("current_phase_diagram", "relevant_approved_knowledge", "long_term_human_advice",
                 "coverage_gaps", "recent_experience", "task_stage_evidence", "post_dft_assessment",
                 "qbc_candidates", "available_branches", "available_budget", "dft_selection_policy") if key in context}}
    # Schemas are program-owned contracts, not scientific/raw filesystem data.
    material["output_contracts"] = deepcopy(payload.get("output_contracts") or {})
    if payload.get("validation_errors"):
        material["validation_errors"] = list(payload["validation_errors"])
        material["repair_instruction"] = "Return the complete corrected action using the supplied contracts and evidence."
    return material


def create_proposal_client(model, *, recursion_limit=16):
    """Caller supplies an approved model; production wiring is deliberately separate."""
    from deepagents import create_deep_agent
    from deepagents.backends import StateBackend
    from deepagents.backends.utils import create_file_data
    from langchain.agents.middleware import wrap_tool_call
    from decision_layer.agent.create_deepseek_client import _parse_json_object
    from decision_layer.agent.prepare_llm_request import prepare_llm_request

    @wrap_tool_call
    def restrict_tools(request, handler):
        if request.tool_call["name"] in {"task", "execute"}:
            raise ValueError("Proposal harness cannot delegate or execute commands")
        return handler(request)

    agent = create_deep_agent(model=model, backend=StateBackend(), subagents=[],
        middleware=[restrict_tools], memory=["/memory/AGENTS.md"], name="scientific_proposal",
        system_prompt=("Analyze supplied scientific facts and approved knowledge. "
            "Files are invocation-local scratch, not business state or durable knowledge. "
            "Do not execute, delegate, approve, submit, train or activate anything. "
            "Return exactly one complete JSON action matching the supplied instruction. "
            "Cite supplied evidence IDs; never invent measurements. "
            "Distinguish human constraints from observations and check memory applicability."))

    def call(payload):
        prepared = proposal_material(prepare_llm_request(payload))
        memory_text = json.dumps(prepared["decision_context"].get("relevant_approved_knowledge", []), ensure_ascii=False)
        result = agent.invoke({"messages": [{"role": "user", "content": json.dumps(prepared, ensure_ascii=False)}],
                               "files": {"/memory/AGENTS.md": create_file_data(
                                   "Approved references, not execution authorization.\n" + memory_text)}},
                              config={"recursion_limit": recursion_limit})
        messages = result.get("messages") or []
        if not messages or messages[-1].type != "ai" or messages[-1].tool_calls:
            raise ValueError("Deep Agents did not return a final proposal")
        action = _parse_json_object(messages[-1].text)
        usage = [message.usage_metadata or {} for message in messages if message.type == "ai"]
        action["_llm_usage"] = {"calls": len(usage), "input_tokens": sum(row.get("input_tokens", 0) for row in usage),
                                "output_tokens": sum(row.get("output_tokens", 0) for row in usage),
                                "cost": None, "model": getattr(model, "model_name", "injected")}
        action["_proposal_harness"] = "deepagents"
        return deepcopy(action)

    call.decision_backend = "langgraph"
    return call


def create_deepseek_proposal_client(*, api_key, model, base_url, timeout, max_tokens=8192):
    """Explicit opt-in provider factory; never chosen by existing default clients."""
    from langchain_openai import ChatOpenAI
    chat_model = ChatOpenAI(api_key=api_key, model=model, base_url=base_url,
        timeout=timeout, max_tokens=max_tokens, temperature=0, max_retries=0,
        extra_body={"thinking": {"type": "disabled"}})
    return create_proposal_client(chat_model)
