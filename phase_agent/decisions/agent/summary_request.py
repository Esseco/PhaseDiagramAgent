"""Summary-first, bounded, read-only evidence transport for scientific proposals."""

from copy import deepcopy
import hashlib
import json
from phase_agent.decisions.agent.analysis_context import (
    ON_DEMAND_HISTORY,
    ON_DEMAND_SUPPORT,
    ScientificEvidenceRequest,
    stage_snapshot,
    PLANNING_INSTRUCTION,
    ANALYSIS_GUIDANCE,
)

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
        "dft_DH_coverage_review",
        "dft_selection_policy",
        "dft_validation_feedback",
        "dft_candidate_costs",
        "dft_stage_limits",
        "independent_audit_candidate_ids",
        "dft_audit_candidate_ids",
    },
    "tasks": {
        "task_stage_evidence",
        "task_round_evidence",
        "refresh_result_facts",
        "retryable_tasks",
        "task_wait_cancellation",
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
    "user_message",
    "user_instruction",
    "human_comment",
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
    if tool == "prepare_local_batch_files" and mode in {
        "relax_mc_results",
        "ordering_search_results",
    }:
        # Existing-result inspection does not select candidates or design a new calculation.
        return common | {"history"}
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


def required_fields(action, sources):
    """Keep original scientific prerequisites; distinguish existing-result inspection."""
    from phase_agent.tools.dispatch.scientific_operations import bind_operation

    bound = bind_operation(action) if action.get("operation") is not None else action
    if (bound.get("tool") or bound.get("action_type")) == "prepare_local_batch_files" and (
        bound.get("parameters") or {}
    ).get("mode") in {"relax_mc_results", "ordering_search_results"}:
        names = {
            "current_project",
            "confirmed_config",
            "confirmed_configuration",
            "config_version",
            "confirmed_config_version",
            "active_model_version",
            "mlip_status",
            "model_refresh",
            "new_model_generation",
            "post_refresh_search",
            "task_stage_evidence",
            "task_round_evidence",
            "tasks",
            "short_term_memory",
            "task_wait_cancellation",
            "refresh_result_facts",
            "failure_attempt_facts",
            "retryable_tasks",
            "budget_usage",
            "available_budget",
            "relax_mc_batches",
            "ordering_searches",
            "batch_return_facts",
            "pending_plans",
            "pending_execution_policies",
            "current_convex_hull",
            "current_phase_diagram",
            "human_constraints",
            "current_settings_authority",
            "decision_authority",
        }
        return names.intersection(sources)
    return topic_fields(sources, required_topics(action))


def comparison_actions(plan, allowed_tools):
    from phase_agent.tools.dispatch.scientific_operations import bind_operation

    operations = [
        bind_operation({"operation": name}) for name in plan.get("comparison_operations") or []
    ]
    if any(action["tool"] not in allowed_tools for action in operations):
        raise ValueError("comparison operation is outside the allowed tool catalog")
    covered = {action["tool"] for action in operations}
    return operations + [
        {"tool": tool} for tool in plan.get("comparison_tools") or [] if tool not in covered
    ]


def source_topic(key):
    return next((name for name, keys in TOPICS.items() if key in keys), "history")


def topic_fields(sources, topics):
    return {
        key
        for key in sources
        if key != "decision_context"
        and source_topic(key) in topics
        and key not in ON_DEMAND_HISTORY | ON_DEMAND_SUPPORT
    }


def expand_topics(request, sources, topics, *, fields=None):
    """Load complete requested fields; execution snapshots require an exact read."""
    names = topic_fields(sources, topics) if fields is None else set(fields)
    found = {source_topic(key) for key in names} | {
        source_topic(key)
        for section in (request["state"], request["decision_context"])
        for key in section
        if key in sources
    }
    for key in sorted(names):
        value = sources[key]
        if key == "current_project":
            request["state"][key] = deepcopy(value)
        elif key not in request["state"]:
            request["decision_context"][key] = deepcopy(value)
    unavailable = (set(request["decision_context"].get("unavailable_topics") or []) - found) | (
        set(topics) - found
    )
    request["decision_context"]["unavailable_topics"] = sorted(unavailable)
    return names


def project_summary(project):
    if not isinstance(project, dict):
        return deepcopy(project)
    return {key: deepcopy(value) for key, value in project.items() if key != "configuration"}


def _evidence_revision(sources, sizes=None):
    # Encode canonical bytes one field at a time with the fast C encoder.
    digest = hashlib.sha256()
    digest.update(b"{")
    for index, key in enumerate(sorted(sources)):
        if index:
            digest.update(b", ")
        digest.update(json.dumps(key, ensure_ascii=False).encode())
        digest.update(b": ")
        encoded = json.dumps(sources[key], ensure_ascii=False, sort_keys=True, default=str).encode()
        digest.update(encoded)
        if sizes is not None:
            sizes[key] = len(encoded)
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
    sizes = {}
    revision = _evidence_revision(sources, sizes)
    catalog = {}
    for key, value in sources.items():
        if key == "decision_context":
            continue
        topic = source_topic(key)
        catalog.setdefault(topic, {})[key] = (
            {"count": len(value)}
            if isinstance(value, (dict, list))
            else {"available": value is not None}
        )
        catalog[topic][key]["serialized_bytes"] = sizes[key]
        if key in ON_DEMAND_HISTORY | ON_DEMAND_SUPPORT:
            catalog[topic][key]["on_demand"] = True
        if key in ON_DEMAND_SUPPORT:
            catalog[topic][key]["supporting_memory"] = True
    view = {
        key: deepcopy(value)
        for key, value in state.items()
        if key in ESSENTIAL and key != "current_project"
    }
    if "current_project" in state:
        view["current_project"] = project_summary(state["current_project"])
    if sources.get("task_wait_cancellation"):
        view["task_wait_cancellation"] = deepcopy(sources["task_wait_cancellation"])
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
    result["decision_context"]["stage_snapshot"] = stage_snapshot(sources)
    from phase_agent.decisions.agent.evidence_overview import summarize

    result["decision_context"]["scientific_overview"] = {
        key: {
            "complete_field": False,
            "source_field": key,
            "evidence_revision": revision,
            "summary": summarize(sources[key]),
        }
        for key in ("branch_pool_facts", "qbc_candidates", "scientific_decision_brief")
        if key in sources
    }
    result["decision_context"]["scientific_overview_rule"] = (
        "All rows contribute to these previews; identities, correlations and detailed limitations "
        "may remain unobserved. These are not complete candidate pools. Exact full source "
        "fields remain indexed and must be read whenever they can affect scientific judgment."
    )
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
    result["evidence_revision"] = revision
    result["output_contracts"] = {"evidence_request": ScientificEvidenceRequest.model_json_schema()}
    result["instruction"] = PLANNING_INSTRUCTION
    result["evidence_reader"] = {
        "kind": "inspect_evidence",
        "phase": "select_topics",
        "topics": list(TOPICS),
        "field_index": catalog,
        "max_reads": PROGRESSIVE_READS + 1,
        "instruction": "Only a summary is loaded. Request needed topics together as {kind: inspect_evidence, topics: [...]}. Missing details are unobserved, not absent. Candidate selection requires the complete relevant pool. No permission to execute or alter approvals. Reuse loaded topics; do not request the same topic again.",
    }
    from phase_agent.tools.dispatch.scientific_operations import operation_catalog

    result["evidence_reader"]["operation_scopes"] = {
        row["operation"]: {"tool": row["tool"], "parameters": row["parameters"]}
        for row in operation_catalog(payload.get("allowed_tools") or [])
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


def evidence_validation_errors(cause):
    from pydantic import ValidationError

    if isinstance(cause, ValidationError):
        return [
            {
                "field": ".".join(str(part) for part in row["loc"]),
                "type": row["type"],
                "message": row["msg"][:500],
            }
            for row in cause.errors(include_input=False, include_url=False, include_context=False)[
                :12
            ]
        ]
    return [{"field": "evidence_request", "type": "value_error", "message": str(cause)[:1500]}]


def request_summary_first(client, payload):
    if (
        getattr(client, "_validates_original_facts", False)
        or payload.get("mode") != "autonomous_search"
        or payload.get("semantic_dialogue")
    ):
        return client(payload)
    request, sources = summary_payload(payload)
    loaded = set()
    loaded_fields = set()
    usage = {}
    scoped_reads = False
    initial_full_fields = {
        key
        for section in (request.get("state") or {}, request.get("conversation_facts") or {})
        for key, value in section.items()
        if key in sources
        and key != "decision_context"
        and json.dumps(value, sort_keys=True, default=str)
        == json.dumps(sources[key], sort_keys=True, default=str)
    }
    step = 0
    corrections = 0
    correction_pending = False
    for _ in range(PROGRESSIVE_READS + 3):
        try:
            action = client(request)
        except Exception as error:
            error.llm_usage = merge_usage(usage, getattr(error, "llm_usage", None))
            raise
        if not isinstance(action, dict):
            raise ValueError("模型未返回有效对象；未执行任务。")
        usage = merge_usage(usage, action.get("_llm_usage"))
        inspecting = action.get("kind") == "inspect_evidence"
        if correction_pending and not inspecting:
            error = ValueError("只读取证纠正必须返回 inspect_evidence；未执行任务。")
            error.llm_usage = usage
            raise error
        fields = None
        view = "full"
        previous_plan = request.get("analysis_plan")
        next_plan = previous_plan
        if inspecting:
            try:
                read = ScientificEvidenceRequest.model_validate(
                    {
                        k: v
                        for k, v in action.items()
                        if k not in {"_llm_usage", "_proposal_harness", "_analysis_route"}
                    }
                )
                fields = None
                unresolved_requests = None
                expanded_requests = None
                if read.fields is not None:
                    fields = []
                    unresolved = []
                    for field in read.fields:
                        parts = field.split(".")
                        if len(parts) > 1 and parts[0] in read.topics:
                            root = parts[1]
                            if root not in sources:
                                unresolved.append(field)
                                continue
                        else:
                            root = parts[0]
                        fields.append(root)
                    if unresolved:
                        unresolved_requests = {
                            "fields": unresolved,
                            "status": "not_in_source_index; unobserved, not empty or scientifically absent",
                            "instruction": "Use exact indexed root names or request the relevant topic. No facts were fabricated for these names.",
                        }
                    expanded_requests = {
                        field: (
                            field.split(".")[1]
                            if field.split(".")[0] in read.topics
                            else field.split(".")[0]
                        )
                        for field in read.fields
                        if field not in unresolved
                    }
                view = read.view
                if read.full_for is not None and any(
                    tool not in (payload.get("allowed_tools") or []) for tool in read.full_for
                ):
                    raise ValueError(
                        "unknown full_for tool: "
                        + ", ".join(
                            tool[:120]
                            for tool in read.full_for
                            if tool not in (payload.get("allowed_tools") or [])
                        )
                    )
                if fields is not None and any(
                    key not in sources
                    or key == "decision_context"
                    or source_topic(key) not in read.topics
                    for key in fields
                ):
                    raise ValueError(
                        "unknown evidence field or topic mismatch: "
                        + ", ".join(
                            key[:120]
                            for key in fields
                            if key not in sources
                            or key == "decision_context"
                            or source_topic(key) not in read.topics
                        )
                    )
                if read.analysis_plan is not None:
                    if any(
                        tool not in (payload.get("allowed_tools") or [])
                        for tool in read.analysis_plan.comparison_tools
                    ):
                        raise ValueError(
                            "comparison tool is outside the allowed tool catalog: "
                            + ", ".join(
                                tool[:120]
                                for tool in read.analysis_plan.comparison_tools
                                if tool not in (payload.get("allowed_tools") or [])
                            )
                        )
                    next_plan = read.analysis_plan.model_dump()
                    comparison_actions(next_plan, payload.get("allowed_tools") or [])
                    if read.analysis_plan.needs_reasoning():
                        next_plan["reasoning"] = "scientific"
            except ValueError as cause:
                from phase_agent.runtime.turn_process import process_event

                details = evidence_validation_errors(cause)
                process_event(
                    "取证请求校验未通过",
                    {
                        "errors": details,
                        "read_only": True,
                        "correction_available": corrections == 0,
                    },
                )
                if corrections == 0:
                    corrections += 1
                    correction_pending = True
                    request["evidence_reader"]["validation_feedback"] = {
                        "errors": details,
                        "attempt": corrections,
                        "max_attempts": 1,
                        "instruction": "Return one corrected inspect_evidence request using exact indexed topics/fields, allowed tools and the supplied schema. Do not return a scientific action. No evidence was expanded; no execution or approval is authorized.",
                    }
                    continue
                error = ValueError(
                    "取证请求不符合只读契约；一次纠正后仍无效；未执行任务。详情："
                    + "; ".join(row["field"] + ": " + row["message"] for row in details)
                )
                error.llm_usage = usage
                raise error from cause
            correction_pending = False
            request["evidence_reader"].pop("validation_feedback", None)
            scoped_reads = scoped_reads or "view" in action or "full_for" in action
            if scoped_reads:
                loaded_fields.update(initial_full_fields)
                request["evidence_reader"]["initial_full_fields"] = sorted(initial_full_fields)
            if unresolved_requests is not None:
                request["evidence_reader"]["unresolved_field_requests"] = unresolved_requests
            if expanded_requests is not None:
                request["evidence_reader"]["expanded_field_requests"] = expanded_requests
        needed = set() if inspecting else required_topics(action)
        from phase_agent.decisions.agent.evidence_overview import action_uses_brief_core

        missing_fields = (set() if inspecting else required_fields(action, sources)) - loaded_fields
        if scoped_reads and not inspecting and action_uses_brief_core(action):
            missing_fields.discard("scientific_decision_brief")
        topics = (
            action.get("topics")
            if inspecting
            else sorted((needed - loaded) | {source_topic(key) for key in missing_fields})
        )
        if scoped_reads and not inspecting:
            absent_topics = {topic for topic in needed if not topic_fields(sources, {topic})}
            topics = sorted(
                {source_topic(key) for key in missing_fields} | ((needed - loaded) & absent_topics)
            )
        preflight_topics = set()
        preflight_fields = set()
        if inspecting and next_plan is not None and view == "full":
            compared_actions = comparison_actions(next_plan, payload.get("allowed_tools") or [])
            full_for = read.full_for if read.full_for is not None else [] if scoped_reads else None
            if full_for is not None:
                compared_actions = [
                    action for action in compared_actions if action["tool"] in full_for
                ]
                covered = {action["tool"] for action in compared_actions}
                compared_actions += [{"tool": tool} for tool in full_for if tool not in covered]
            for compared in compared_actions:
                preflight_topics.update(required_topics(compared))
                compared_fields = required_fields(compared, sources)
                if scoped_reads and action_uses_brief_core(compared):
                    compared_fields.discard("scientific_decision_brief")
                preflight_fields.update(compared_fields)
            topics = sorted(set(topics) | preflight_topics)
        if not inspecting and not topics and not missing_fields:
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
        names = (
            (topic_fields(sources, read.topics) if fields is None else set(fields))
            if inspecting
            else missing_fields
        )
        # Resolve the original prerequisites before reasoning, not after a draft.
        names |= preflight_fields
        new_fields = names - loaded_fields
        if view == "overview":
            new_fields -= set(request["decision_context"].get("field_overviews") or {})
        new = set(topics) - loaded
        plan_changed = next_plan != previous_plan
        if (not new and not new_fields and not plan_changed) or step == PROGRESSIVE_READS + 1:
            error = ValueError("按需取证已达上限或重复请求；未执行任务。")
            error.llm_usage = usage
            raise error
        if step == PROGRESSIVE_READS:
            new = set(TOPICS) - loaded
            new_fields |= topic_fields(sources, set(TOPICS)) - loaded_fields
            view = "full"
        from phase_agent.decisions.agent.evidence_overview import split_read

        full_fields, previews = split_read(sources, new_fields, view, request["evidence_revision"])
        # Only exact reads satisfy original candidate/prerequisite completeness checks.
        expand_topics(
            request, sources, new | {source_topic(key) for key in new_fields}, fields=full_fields
        )
        loaded_fields.update(full_fields)
        if (
            scoped_reads
            and "scientific_decision_brief" in sources
            and "scientific_decision_brief" not in loaded_fields
        ):
            from phase_agent.decisions.agent.evidence_overview import decision_core, summarize

            core, deferred = decision_core(sources["scientific_decision_brief"])
            request["decision_context"]["scientific_decision_brief"] = core
            if deferred:
                request["evidence_reader"]["deferred_brief_components"] = sorted(deferred)
                # Exact core plus component previews supersede the original duplicated preview.
                request["decision_context"].get("scientific_overview", {}).pop(
                    "scientific_decision_brief", None
                )
                request["decision_context"].setdefault("field_overviews", {})[
                    "scientific_decision_brief"
                ] = {
                    "complete_field": False,
                    "source_field": "scientific_decision_brief",
                    "evidence_revision": request["evidence_revision"],
                    "summary": {
                        key: summarize(sources["scientific_decision_brief"][key])
                        for key in deferred
                    },
                }
            else:
                loaded_fields.add("scientific_decision_brief")
        if "scientific_decision_brief" in loaded_fields:
            request["evidence_reader"].pop("deferred_brief_components", None)
        initial_overview = request["decision_context"].get("scientific_overview") or {}
        for key in full_fields:
            initial_overview.pop(key, None)
        if previews or "field_overviews" in request["decision_context"]:
            overviews = request["decision_context"].setdefault("field_overviews", {})
            overviews.update(previews)
            for key in full_fields:
                overviews.pop(key, None)
            unavailable = set(request["decision_context"].get("unavailable_topics") or [])
            request["decision_context"]["unavailable_topics"] = sorted(
                unavailable - {source_topic(key) for key in overviews}
            )
            request["evidence_reader"]["view_rule"] = (
                "An overview is not a complete candidate pool or complete source field. "
                "All records contribute to exact counts/ranges, but identities, correlations "
                "and detailed limitations may be unobserved. Read full evidence whenever "
                "they affect judgment. Complete candidate pools and critical facts are required. "
                "For current-pool MC/DFT selection, exact decision core may defer historical "
                "coverage/task maps; generation or convergence requires the full original brief. "
                "All legal routes and exact source reads remain available."
            )
        request["evidence_reader"]["fully_loaded_fields"] = sorted(loaded_fields)
        if not inspecting:
            request["provisional_proposal"] = {
                "status": "unvalidated_hypothesis_not_authority",
                "evidence_revision": request["evidence_revision"],
                "unvalidated_action": {
                    k: deepcopy(v) for k, v in action.items() if k != "_llm_usage"
                },
            }
        if next_plan is not None:
            request["analysis_plan"] = next_plan
        request["instruction"] = payload.get("instruction", "")
        if next_plan is not None:
            request["instruction"] += " " + ANALYSIS_GUIDANCE
        request["output_contracts"] = {
            **(payload.get("output_contracts") or {}),
            "evidence_request": request["output_contracts"]["evidence_request"],
        }
        request["evidence_reader"]["phase"] = "analyze"
        request["evidence_reader"]["current_view"] = view
        loaded.update(set(topics) | new)
        request["decision_context"]["loaded_topics"] = sorted(loaded)
        from phase_agent.runtime.turn_process import process_event

        process_event(
            "按需读取决策证据",
            {
                "topics": sorted(set(topics) | new),
                "fields": sorted(full_fields),
                "overview_fields": sorted(previews),
                "view": view,
                "preflight_topics": sorted(preflight_topics),
                "preflight_fields": sorted(preflight_fields),
                "analysis_plan": next_plan,
                "read_only": True,
                "loaded_serialized_bytes": sum(
                    request["evidence_reader"]["field_index"][source_topic(key)][key][
                        "serialized_bytes"
                    ]
                    for key in loaded_fields
                ),
            },
        )
        step += 1
