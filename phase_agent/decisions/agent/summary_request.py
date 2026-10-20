"""Summary-first, bounded, read-only evidence transport for scientific proposals."""

from copy import deepcopy
import hashlib
import json

TOPICS = {
    "mc": {
        "available_branches",
        "branch_candidates",
        "mc_entry_conditions",
        "relax_mc_entry_conditions",
        "branch_pool_facts",
        "branch_pool_review_contract",
        "branch_pool_review_required",
        "generation_region_contract",
    },
    "dft": {
        "qbc_candidates",
        "post_dft_assessment",
        "training_excluded_data_ids",
        "dft_magnetic_diagnostics",
        "dft_sampling_recommendation",
    },
    "tasks": {
        "task_stage_evidence",
        "task_round_evidence",
        "refresh_result_facts",
        "retryable_tasks",
        "failure_attempt_facts",
        "tasks",
        "short_term_memory",
    },
    "model": {
        "model_refresh",
        "model_round_summary",
        "training_validation_planning",
        "post_refresh_search",
        "training_result_reports",
        "mlip_status",
        "new_model_generation",
        "active_model",
        "active_model_version",
    },
    "budget": {
        "round_budget_evidence",
        "calculation_cost_reference",
        "budget_usage",
        "available_budget",
    },
    "convergence": {
        "latest_convergence_check",
        "coverage",
        "search_coverage",
        "current_convex_hull",
        "current_phase_diagram",
    },
    "configuration": {
        "confirmed_configuration",
        "confirmed_config",
        "convergence_rules",
        "current_project",
        "scientific_decision_brief",
    },
    "plans": {"pending_execution_policies", "pending_plans"},
    "history": set(),
}
ESSENTIAL = {
    "snapshot_id",
    "snapshot_index",
    "config_version",
    "current_status",
    "status",
    "task_counts",
    "task_stage_counts",
    "pool_counts",
    "human_constraints",
    "available_budget",
    "current_project",
    "recent_failures",
    "gpu_usage_summary",
    "batch_return_facts",
}

PROGRESSIVE_READS = 4


def required_topics(action):
    """Read prerequisites before accepting a proposal, never grant authority."""
    from phase_agent.tools.dispatch.scientific_operations import bind_operation

    bound = bind_operation(action) if action.get("operation") is not None else action
    tool = bound.get("tool") or bound.get("action_type")
    if not isinstance(tool, str) or not tool:
        return set()
    common = {"configuration", "budget", "model", "tasks"}
    scientific = {
        "generate_branches": {"mc", "convergence", "history"},
        "select_dft_candidates": {"dft", "convergence"},
        "allocate_mc_bohb": {"mc", "convergence"},
        "prepare_mlip_training": {"dft", "convergence"},
        "activate_mlip_candidate": {"dft", "convergence"},
        "update_mlip": {"dft", "convergence"},
        "reevaluate_candidates": {"dft", "convergence"},
        "adjust_strategy": {"mc", "dft", "convergence", "history"},
        "check_convergence": {"dft", "convergence"},
        "pause_search": {"convergence"},
    }
    mode = str((bound.get("parameters") or {}).get("mode", "")).lower()
    if tool == "prepare_local_batch_files":
        extra = (
            {"dft", "convergence"}
            if "dft" in mode
            else {"model"}
            if "refresh" in mode
            else {"mc", "convergence"}
        )
    else:
        extra = scientific.get(tool, set())
    return common | extra


def expand_topics(request, sources, topics):
    """Preserve full values and distinguish absent evidence from empty pools."""
    found = set()
    for key, value in sources.items():
        topic = next((name for name, keys in TOPICS.items() if key in keys), "history")
        if topic not in topics or key == "decision_context":
            continue
        found.add(topic)
        if key == "current_project":
            request["state"][key] = deepcopy(value)
        elif key not in request["state"]:
            request["decision_context"][key] = deepcopy(value)
    unavailable = set(request["decision_context"].get("unavailable_topics") or []) | (
        set(topics) - found
    )
    request["decision_context"]["unavailable_topics"] = sorted(unavailable)


def project_summary(project):
    if not isinstance(project, dict):
        return deepcopy(project)
    return {key: deepcopy(value) for key, value in project.items() if key != "configuration"}


def _evidence_revision(sources):
    # Encode canonical bytes one field at a time with the fast C encoder.
    digest = hashlib.sha256()
    digest.update(b"{")
    for index, key in enumerate(sorted(sources)):
        if index:
            digest.update(b", ")
        digest.update(json.dumps(key, ensure_ascii=False).encode())
        digest.update(b": ")
        digest.update(
            json.dumps(sources[key], ensure_ascii=False, sort_keys=True, default=str).encode()
        )
    digest.update(b"}")
    return digest.hexdigest()


def summary_payload(payload):
    state = payload.get("state") or {}
    context = {
        **(state.get("decision_context") or {}),
        **(payload.get("decision_context") or {}),
    }
    sources = {**state, **context}
    for key in ("available_branches", "qbc_candidates"):
        if state.get(key) is not None:
            sources[key] = state[key]
    for key in (
        "relax_mc_batches",
        "active_model_version",
        "batch_return_facts",
        "phase_diagrams",
        "tasks",
    ):
        if key in state:
            sources[key] = state[key]
    catalog = {}
    for key, value in sources.items():
        if key == "decision_context":
            continue
        topic = next((name for name, keys in TOPICS.items() if key in keys), "history")
        catalog.setdefault(topic, {})[key] = (
            {"count": len(value)}
            if isinstance(value, (dict, list))
            else {"available": value is not None}
        )
    view = {
        key: deepcopy(value)
        for key, value in state.items()
        if key in ESSENTIAL and key != "current_project"
    }
    if "current_project" in state:
        view["current_project"] = project_summary(state["current_project"])
    memory = state.get("short_term_memory") or {}
    if memory.get("task_status"):
        view["task_counts"] = deepcopy(memory["task_status"])
    from phase_agent.analysis.state.batch_return_facts import batch_return_facts

    receipt_sources = sources
    if "relax_mc_batches" in state:
        receipt_sources = {**sources, "relax_mc_batches": state["relax_mc_batches"]}
    model = state.get("active_model_version") or (state.get("mlip_status") or {}).get(
        "active_version"
    )
    if model:
        receipt_sources = {**receipt_sources, "active_model_version": model}
    projected = state.get("batch_return_facts")
    if (
        "relax_mc_batches" not in state
        and isinstance(projected, dict)
        and (not model or projected.get("model_version") == model)
    ):
        view["batch_return_facts"] = deepcopy(projected)
    else:
        view["batch_return_facts"] = batch_return_facts(receipt_sources)
    view.setdefault(
        "pool_counts",
        {key: len(sources.get(key) or []) for key in ("available_branches", "qbc_candidates")},
    )
    long_term = state.get("long_term_memory") or context.get("long_term_memory") or {}
    view["human_constraints"] = deepcopy(
        state.get("human_constraints") or long_term.get("human_system_knowledge") or []
    )
    for key in ("frozen_parameter_advice", "search_rules"):
        if long_term.get(key):
            view[key] = deepcopy(long_term[key])
    result = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key not in {"state", "decision_context"}
    }
    result.update(state=view, decision_context={"evidence_catalog": catalog, "loaded_topics": []})
    if context.get("proposal_validation_feedback"):
        result["decision_context"]["proposal_validation_feedback"] = deepcopy(
            context["proposal_validation_feedback"]
        )
    references = (context.get("evidence_catalog") or {}).get("references") or {}
    result["decision_context"]["evidence_reference_index"] = {
        "references": {
            key: {"kind": row.get("kind")}
            for key, row in references.items()
            if isinstance(row, dict)
        },
        "instruction": "Use exact keys from this index for evidence_refs; never invent references. Keys identify sources only, not scientific adequacy. Read the relevant evidence topics before making any scientific proposal; request history for full reference records when needed.",
    }
    # Bind the short request to every underlying fact, even those not transmitted.
    result["evidence_revision"] = _evidence_revision(sources)
    from phase_agent.decisions.agent.dialogue_contract import EvidenceRequestOutcome

    result["output_contracts"] = {"evidence_request": EvidenceRequestOutcome.model_json_schema()}
    result["instruction"] = (
        "Select the evidence topics needed to analyze the current request. Return only inspect_evidence with topics; do not perform scientific reasoning or produce a proposed action yet. Request the prerequisites for the routes you need to compare together."
    )
    result["evidence_reader"] = {
        "kind": "inspect_evidence",
        "phase": "select_topics",
        "topics": list(TOPICS),
        "max_reads": PROGRESSIVE_READS + 1,
        "instruction": "Only a summary is loaded. Request needed topics together as {kind: inspect_evidence, topics: [...]}. Missing details are unobserved, not absent. Candidate selection requires the complete relevant pool. No permission to execute or alter approvals. Reuse loaded topics; do not request the same topic again.",
    }
    result["evidence_reader"]["required_before_action"] = {
        tool: sorted(required_topics({"tool": tool})) for tool in payload.get("allowed_tools") or []
    }
    result["evidence_reader"]["fallback"] = (
        "After four progressive reads, remaining evidence is expanded together once. Absent evidence is unavailable, not proof of an empty pool. Do not guess prerequisites. Original full-state validation and approval remain required."
    )
    return result, sources


def merge_usage(total, usage):
    if not usage:
        return total
    result = {**total, **usage}
    for key in ("calls", "input_tokens", "output_tokens"):
        result[key] = int(total.get(key) or 0) + int(usage.get(key) or 0)
    return result


def request_summary_first(client, payload):
    if (
        getattr(client, "_validates_original_facts", False)
        or payload.get("mode") != "autonomous_search"
        or payload.get("semantic_dialogue")
    ):
        return client(payload)
    request, sources = summary_payload(payload)
    loaded = set()
    usage = {}
    for step in range(PROGRESSIVE_READS + 2):
        try:
            action = client(request)
        except Exception as error:
            error.llm_usage = merge_usage(usage, getattr(error, "llm_usage", None))
            raise
        if not isinstance(action, dict):
            raise ValueError("模型未返回有效对象；未执行任务。")
        usage = merge_usage(usage, action.get("_llm_usage"))
        inspecting = action.get("kind") == "inspect_evidence"
        if inspecting:
            from phase_agent.decisions.agent.dialogue_contract import dialogue_errors

            if dialogue_errors(action, True):
                error = ValueError("取证请求不符合只读契约；未执行任务。")
                error.llm_usage = usage
                raise error
        needed = set() if inspecting else required_topics(action)
        topics = action.get("topics") if inspecting else sorted(needed - loaded)
        if not inspecting and not topics:
            from phase_agent.tools.dispatch.scientific_operations import bind_operation

            bound = bind_operation(action) if action.get("operation") is not None else action
            if (
                bound.get("tool") or bound.get("action_type")
            ) == "select_dft_candidates" and not isinstance(
                sources.get("qbc_candidates"), (list, dict)
            ):
                error = ValueError("缺少完整 DFT 候选池证据；无法形成可靠选择，本轮未执行任务。")
                error.llm_usage = usage
                raise error
            missing_evidence = needed & set(
                request["decision_context"].get("unavailable_topics") or []
            )
            if missing_evidence:
                error = ValueError(
                    "关键决策证据缺失（"
                    + "、".join(sorted(missing_evidence))
                    + "）；无法形成可靠方案，本轮未执行任务。"
                )
                error.llm_usage = usage
                raise error
            action = deepcopy(action)
            if usage:
                action["_llm_usage"] = usage
            return action
        if (
            not isinstance(topics, list)
            or not topics
            or any(not isinstance(t, str) or t not in TOPICS for t in topics)
        ):
            error = ValueError("证据主题无效；未执行任务。")
            error.llm_usage = usage
            raise error
        new = set(topics) - loaded
        if not new or step == PROGRESSIVE_READS + 1:
            error = ValueError("按需取证已达上限或重复请求；未执行任务。")
            error.llm_usage = usage
            raise error
        if step == PROGRESSIVE_READS:
            new = set(TOPICS) - loaded
        # Last expansion is comprehensive, but never performs scientific work.
        expand_topics(request, sources, new)
        request["instruction"] = payload.get("instruction", "")
        request["output_contracts"] = {
            **(payload.get("output_contracts") or {}),
            "evidence_request": request["output_contracts"]["evidence_request"],
        }
        request["evidence_reader"]["phase"] = "analyze"
        loaded.update(new)
        request["decision_context"]["loaded_topics"] = sorted(loaded)
        from phase_agent.runtime.turn_process import process_event

        process_event("按需读取决策证据", {"topics": sorted(new), "read_only": True})
