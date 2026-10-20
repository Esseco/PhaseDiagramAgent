"""Activation barrier using the existing approval, task and budget interfaces."""

from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.analysis.feedback.model_refresh_plan import (
    supplemental_relaxation_required,
)
from phase_agent.tools.budget.release_budget import release_budget
from phase_agent.tools.policy.execution_policy import (
    apply_execution_policy,
    build_agent_proposal,
)
from phase_agent.tools.policy.validate_tool_action import validate_tool_action
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.workflows.model_refresh_state import (
    refresh_active,
    refresh_preview,
    fingerprint,
)


def advance_refresh(state, user_message=""):
    from phase_agent.analysis.convergence.mlip_relaxation import (
        reconcile_relaxation_quality,
        DISCARDED,
    )

    current = reconcile_relaxation_quality(state)
    refresh = current["model_refresh"]
    if refresh.get("status") != "waiting_results":
        return current, None
    task_ids = refresh.get("task_ids") or []
    if not task_ids:
        return current, "刷新任务清单为空，需核对本轮任务登记；尚未完成刷新。"
    if len(set(task_ids)) != len(task_ids):
        return current, "刷新任务清单存在重复ID，需核对登记；尚未完成刷新。"
    registered = current.get("tasks") or []
    for tid in task_ids:
        if sum(task.get("task_id") == tid for task in registered) > 1:
            return current, f"刷新任务{tid}存在重复登记，需核对结果；尚未完成刷新。"
    tasks = {t["task_id"]: t for t in registered}
    rows = [tasks.get(tid, {}) for tid in refresh["task_ids"]]

    def accepted(task):
        outputs = task.get("outputs") or {}
        return (
            task.get("status") == "completed"
            and task.get("model_version") == refresh["new_model_version"]
            and task.get("checks_passed", True) is True
            and (task.get("converged") is True or outputs.get("single_point_completed") is True)
        )

    discarded = {t.get("task_id") for t in rows if t.get("result_disposition") == DISCARDED}
    if discarded or "discarded_task_ids" in refresh:
        refresh["discarded_task_ids"] = sorted(discarded)
    missing = [
        tid
        for tid in refresh["task_ids"]
        if tid not in discarded and not accepted(tasks.get(tid, {}))
    ]
    if missing and user_message.strip() == "不再等待刷新":
        refresh["waived_task_ids"] = sorted(set(refresh.get("waived_task_ids", [])) | set(missing))
        for tid in missing:
            if tid in tasks:
                tasks[tid]["refresh_wait_waived"] = True
    unresolved = set(missing) - set(refresh.get("waived_task_ids", []))
    if unresolved:
        from phase_agent.analysis.state.workflow_blockers import refresh_result_facts

        facts = refresh_result_facts(current)
        failed = sum(facts["status_counts"].get(s, 0) for s in ("failed", "timeout"))
        if failed:
            reasons = "；".join(
                f"{r['count']} 个：{r['reason']}" for r in facts["failure_groups"][:3]
            )
            return current, (
                f"刷新第{refresh['wave'] + 1}批：{failed} 个失败，有效回收{len(rows) - len(missing)}/{len(rows)}。\n"
                f"失败原因：{reasons}\n"
                "先修复上述依赖，再提出具体重试方案；反复回复“继续”或“同意”不会恢复缺失文件。原任务、结果和已用 GPU 耗时保留。"
            )
        return (
            current,
            f"本批有效回收 {len(rows) - len(missing)}/{len(rows)}，还有 {len(unresolved)} 个结果未通过回收。\n"
            + "\n".join(
                f"未读到：{tid} → {tasks.get(tid, {}).get('result_path') or '结果路径未登记'}"
                for tid in sorted(unresolved)[:3]
            )
            + "\n本次只检查回传，未新增任务、未追加弛豫。",
        )
    plan = refresh["plan"]
    diagram = (current.get("phase_diagrams") or {}).get("mlip") or {}
    if (
        diagram.get("status") != "completed"
        or diagram.get("model_version") != refresh["new_model_version"]
    ):
        return (
            current,
            "刷新结果尚不能构建新模型相图，请补齐相识别或Na端点；不会沿用旧模型能量。",
        )
    diagram_entries = diagram.get("entries") or []
    record_ids = [entry["record_id"] for entry in diagram_entries]
    if len(set(record_ids)) != len(record_ids):
        return current, "新模型相图存在重复记录ID，需核对能量证据；尚未完成刷新。"
    new_records = [
        record
        for record in current.get("phase_records") or []
        if record.get("source_version") == refresh["new_model_version"]
    ]
    for tid in task_ids:
        if sum(record.get("source_task_id") == tid for record in new_records) > 1:
            return (
                current,
                f"刷新任务{tid}存在多个相图记录，需核对唯一结果；尚未完成刷新。",
            )
    entries = {e["record_id"]: e for e in diagram_entries}
    phases = {
        r.get("source_task_id"): entries.get(r.get("record_id"))
        for r in current.get("phase_records") or []
        if r.get("source_version") == refresh["new_model_version"]
    }
    outside = set()
    records_by_task = {record.get("source_task_id"): record for record in new_records}
    for task in rows:
        evidence = (task.get("outputs") or {}).get("phase_identification") or {}
        record = records_by_task.get(task.get("task_id"), {})
        if (
            accepted(task)
            and task.get("phase_identification_status") == "outside_boundary"
            and evidence.get("status") == "outside_boundary"
            and evidence.get("observed_phase")
            and evidence.get("structure_sha256")
            and evidence.get("classification_scope")
            and record.get("status") == "outside_search_boundary"
            and record.get("phase_identification") == evidence
        ):
            outside.add(task["task_id"])
    refresh["outside_boundary_count"] = len(outside)
    missing_phases = [
        task
        for task in rows
        if accepted(task) and phases.get(task["task_id"]) is None and task["task_id"] not in outside
    ]
    if missing_phases:
        from collections import Counter

        reasons = Counter(
            ((task.get("outputs") or {}).get("phase_identification") or {}).get("reason")
            or task.get("phase_identification_reason")
            or "缺少相图记录或能量证据"
            for task in missing_phases
        )
        explanations = "；".join(
            f"{count} 个：{str(reason)[:220]}" for reason, count in reasons.items()
        )
        return (
            current,
            f"刷新结果有 {len(missing_phases)} 项未进入新版本相图，相识别或能量校验受阻。\n"
            f"原因：{explanations}\n"
            "下一步：先核对这些结构的相识别和能量证据，必要的处理方案须单独审阅；"
            "核对完成后先展示新branch生成方案，再审批生成。未新增计算，重复‘同意’不会解除此阻塞。",
        )
    supplements = []
    if refresh["wave"] == 0:
        original = {r["structure_sha256"]: r for r in plan["candidates"]}
        for task in rows:
            if (
                not accepted(task)
                or task.get("task_id") in outside
                or (task.get("parameters") or {}).get("model_refresh_operation") != "predict"
            ):
                continue
            entry = phases.get(task["task_id"])
            if entry is None or entry.get("ehull_unit") != "eV/atom":
                return (
                    current,
                    f"刷新任务{task['task_id']}缺少唯一新版本Ehull证据，未追加计算。",
                )
            outputs = task["outputs"]
            if supplemental_relaxation_required(float(entry["ehull"]), outputs.get("forces")):
                path = Path(outputs["structure_path"])
                source = original[task["input_structure_sha256"]]
                supplements.append(
                    {
                        **source,
                        "operation": "relax",
                        "approved_input_structure_sha256": source["structure_sha256"],
                        "structure_path": str(path),
                        "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    refresh["supplemental_candidates"] = supplements
    if supplements:
        refresh.update(status="supplement_ready", wave=1)
        return current, None
    key = refresh.get("supplemental_reservation_key")
    if key:
        current = release_budget(
            current,
            task_key=key,
            reason="refresh supplemental envelope completed or unused",
        )["state"]
        refresh = current["model_refresh"]
    initial_ids = {
        t.get("input_structure_sha256")
        for t in current.get("tasks") or []
        if t.get("model_refresh_id") == plan["checksum"] and accepted(t)
    }
    refresh.update(
        status="completed",
        refreshed=len(initial_ids),
        deferred=plan["total_unique"] - len(initial_ids),
        partial=len(initial_ids) < plan["total_unique"],
    )
    diagram["refresh_coverage"] = {k: refresh[k] for k in ("refreshed", "deferred", "partial")}
    current["phase_diagrams"]["mlip"]["refresh_coverage"] = deepcopy(diagram["refresh_coverage"])
    current.setdefault("phase_diagrams_by_model", {}).setdefault(refresh["new_model_version"], {})[
        "refresh_coverage"
    ] = deepcopy(diagram["refresh_coverage"])
    return current, None


def run_model_refresh_gate(
    state,
    session,
    *,
    registry,
    context,
    human_feedback=None,
    mode="interactive",
    pending_key="__single_interactive_action__",
):
    if not refresh_active(state):
        return None
    try:
        current, waiting = advance_refresh(state, str(context.get("user_message") or ""))
    except (ValueError, OSError, KeyError) as error:
        return {
            "status": "not_configured",
            "state": deepcopy(state),
            "submitted": False,
            "reason": f"刷新结果证据不完整：{error}；未追加计算，原始结果保留。",
        }
    if waiting:
        return {
            "status": "not_configured",
            "state": current,
            "reason": waiting,
            "submitted": False,
        }
    if not refresh_active(current):
        refresh = current["model_refresh"]
        diagram = (current.get("phase_diagrams") or {}).get("mlip") or {}
        return {
            "status": "completed",
            "state": current,
            "submitted": False,
            "completed_refresh": {
                "model_version": refresh["new_model_version"],
                "refreshed": refresh["refreshed"],
                "deferred": refresh["deferred"],
                "partial": refresh["partial"],
                "task_count": len(refresh.get("task_ids") or []),
                "phase_diagram_entry_count": len(diagram.get("entries") or []),
                **(
                    {"discarded_unconverged_count": len(refresh["discarded_task_ids"])}
                    if refresh.get("discarded_task_ids")
                    else {}
                ),
                **(
                    {"outside_boundary_count": refresh["outside_boundary_count"]}
                    if refresh.get("outside_boundary_count")
                    else {}
                ),
            },
            "reason": "新模型结构刷新已完成；下一步先审阅并批准新branch生成方案，展示生成结果后再单独审阅Relax/MC方案。",
        }
    config = context["effective_config"]
    refresh = current["model_refresh"]
    automatic = refresh.get("status") == "supplement_ready"
    # Supplemental candidates are a new decision, never covered by first-wave approval.
    approval_identity = fingerprint(
        {
            "config": session["confirmed_snapshot"],
            "active_model": current.get("active_model"),
        }
    )
    authorization = refresh.get("input_authorization") or {}
    resume_approved = (
        not automatic
        and authorization.get("identity") == approval_identity
        and authorization.get("plan")
    )
    from phase_agent.tools.workflows.model_refresh_scope import (
        approved_refresh_scope,
        refresh_scope_errors,
        consume_refresh_scope,
    )

    if automatic:
        # The old envelope bounds candidates, not approval of a new configuration.
        # Bind current settings to a new proposal identity below, then require approval.
        scope_errors = refresh_scope_errors(
            refresh, (refresh.get("approved_scope") or {}).get("config_version")
        )
        if (refresh.get("plan") or {}).get("model") != current.get("active_model"):
            scope_errors.append("当前模型与原刷新模型不一致")
        if scope_errors:
            return {
                "status": "confirmation_required",
                "state": current,
                "submitted": False,
                "reason": "补充刷新暂停："
                + "；".join(scope_errors)
                + "。请修订并审阅刷新范围；未追加输入或计算。",
            }
    try:
        plan = (
            refresh["plan"]
            if automatic
            else (
                deepcopy(authorization["plan"])
                if resume_approved
                else refresh_preview(current, context["manager"], config)
            )
        )
    except (ValueError, OSError) as error:
        return {
            "status": "not_configured",
            "state": current,
            "reason": str(error),
            "submitted": False,
        }
    action = {
        "tool": "prepare_local_batch_files",
        "action_type": "prepare_local_batch_files",
        "task_key": f"refresh-plan:{plan['checksum']}:{refresh.get('wave', 0)}",
        "stage": "relax_and_feature",
        "budget": (
            sum(r["relative_cost"] for r in refresh.get("supplemental_candidates", []))
            if automatic
            else plan["maximum_total_cost"]
        ),
        "target_ids": [
            r["structure_id"]
            for r in (
                refresh.get("supplemental_candidates", []) if automatic else plan["candidates"]
            )
        ],
        "decision_source": "approved_refresh_policy",
        "reason": (
            f"首批结果已回收，建议另行追加 {len(refresh.get('supplemental_candidates', []))} 个弛豫任务；须单独批准，当前未生成输入。"
            if automatic
            else "新模型激活后先刷新累计结构；不混用旧模型能量。"
        ),
        "parameters": {
            "mode": "model_refresh_inputs",
            "preview_checksum": plan["checksum"],
            "refresh_preview": plan,
        },
    }
    if automatic:
        relax_parameters = deepcopy((config.get("mlip") or {}).get("relax_parameters") or {})
        action["parameters"]["supplemental_relax_parameters"] = relax_parameters
        if relax_parameters:
            action["reason"] += (
                f"本批弛豫最多 {relax_parameters.get('relax_steps', 300)} 步，"
                f"优化器 {relax_parameters.get('optimizer', 'LBFGS')}。"
            )
        action["task_key"] += (
            ":supplement:"
            + fingerprint(
                {
                    "candidates": refresh.get("supplemental_candidates", []),
                    "config": session["confirmed_snapshot"],
                    "relax_parameters": relax_parameters,
                }
            )[:16]
        )
    proposal = build_agent_proposal(action, current, runtime_state=current)
    key = pending_key
    stored = (current.get("pending_execution_policies") or {}).get(key) or {}
    previous = ((stored.get("agent_proposal") or {}).get("raw_action") or {}).get("task_key")
    # A previous action's approval must never approve a new refresh or changed files.
    if resume_approved:
        human_feedback = "同意"
    elif previous != action["task_key"]:
        human_feedback = None
    policy = apply_execution_policy(
        proposal,
        execution_mode="dry_run" if mode == "dry_run" else "interactive",
        human_feedback=human_feedback,
    )
    if not policy.get("execute"):
        from phase_agent.tools.policy.native_approval import native_approval_gate

        blocked = native_approval_gate(
            {
                "mode": mode,
                "context": context,
                "pending_key": action["task_key"],
                "current": current,
                "proposal": proposal,
                "stored": stored,
            },
            policy,
        )
        if blocked is not None:
            return blocked
    if policy["status"] == "awaiting_approval":
        current.setdefault("pending_execution_policies", {})[key] = {
            "agent_proposal": proposal,
            "record_id": action["task_key"],
        }
        return {
            "status": "awaiting_approval",
            "agent_proposal": proposal,
            "action": action,
            "record_id": action["task_key"],
            "state": current,
            "submitted": False,
        }
    if not policy["execute"]:
        current.setdefault("pending_execution_policies", {}).pop(key, None)
        return {
            "status": policy["status"],
            "state": current,
            "agent_proposal": proposal,
            "submitted": False,
        }
    validation_state = current
    if automatic:
        validation_state = release_budget(
            current,
            task_key=refresh["supplemental_reservation_key"],
            reason="replace envelope with actual supplemental tasks",
        )["state"]
    validation = validate_tool_action(action, validation_state, session, registry)
    budget_errors = {
        "total_cost_limit",
        "stage_task_limit",
        "stage_cost_limit",
        "reserved_total_cost_limit",
    }
    deferred_budget_errors = [error for error in validation["errors"] if error in budget_errors]
    if not validation["valid"] and (
        automatic or any(error not in budget_errors for error in validation["errors"])
    ):
        return {
            "status": "rejected",
            "state": current,
            "reason": "刷新方案校验失败，未截断：" + str(validation["errors"]),
            "submitted": False,
        }
    if not resume_approved:
        from phase_agent.tools.policy.native_approval import native_approval_gate

        blocked = native_approval_gate(
            {
                "mode": mode,
                "context": context,
                "pending_key": action["task_key"],
                "current": current,
                "proposal": proposal,
                "stored": stored,
            },
            policy,
        )
        if blocked is not None:
            return blocked
    if not automatic:
        # Preserve a delivered approval before a budget preflight can block it.
        # No execution receipt or files exist until that preflight succeeds.
        refresh["input_authorization"] = {
            "identity": approval_identity,
            "config_version": session["confirmed_snapshot"]["config_version"],
            "plan": deepcopy(plan),
            "approved_by": action["task_key"],
        }
        from phase_agent.tools.budget.check_budget import check_budget

        checked = check_budget(
            current,
            {
                "stage": "relax_and_feature",
                "tasks": len(plan["candidates"]) + plan["maximum_supplemental_count"],
                "relative_cost": plan["maximum_total_cost"],
            },
            config["budgets"],
        )
        if not checked["allowed"] or deferred_budget_errors:
            current.setdefault("pending_execution_policies", {}).pop(key, None)
            return {
                "status": "budget_exhausted",
                "state": current,
                "submitted": False,
                "reason": "刷新方案已批准，尚未生成输入；预算阻塞："
                + ",".join(sorted(set(checked["reasons"] + deferred_budget_errors)))
                + "。范围和配置不变时，继续将复用本次批准。",
            }
    if automatic:
        refresh["supplemental_input_authorization"] = {
            "approved_by": action["task_key"],
            "checksum": fingerprint(refresh.get("supplemental_candidates", [])),
            "config_version": session["confirmed_snapshot"]["config_version"],
            "relax_parameters_checksum": fingerprint(
                action["parameters"]["supplemental_relax_parameters"]
            ),
        }
    execution = execute_tool_action(
        action,
        registry=registry,
        context={
            **context,
            "event_state": current,
            "config_version": session["confirmed_snapshot"]["config_version"],
            "approval_record_id": action["task_key"],
            "invocation_id": action["task_key"],
        },
    )
    result = execution.get("result") or {}
    updated = result.get("state", current)
    if result.get("status") == "prepared":
        from phase_agent.tools.remote.submission_handoff import ensure_waiting_handoffs

        from phase_agent.runtime.turn_process import timed_call

        timed_call(
            "刷新：整批压缩与交付校验", ensure_waiting_handoffs, updated, category="processing"
        )
        updated.setdefault("pending_execution_policies", {}).pop(key, None)
        if automatic:
            consume_refresh_scope(updated["model_refresh"])
        else:
            updated["model_refresh"]["approved_scope"] = approved_refresh_scope(
                plan, session["confirmed_snapshot"]["config_version"]
            )
    if execution.get("status") in {"completed", "failed"}:
        updated.setdefault("invocations", {})[action["task_key"]] = {
            "status": result.get("status", execution["status"]),
            "execution": {
                "status": execution["status"],
                "tool": execution.get("tool"),
                "error": execution.get("error"),
            },
        }
    updated.setdefault("action_records", []).append(
        {
            "record_id": action["task_key"],
            "final_action": action,
            "status": result.get("status", execution["status"]),
            "approval_basis": "explicit_user_approval",
        }
    )
    return {
        "status": result.get("status", execution["status"]),
        "state": updated,
        "execution": execution,
        "execution_result": result,
        "reason": result.get("reason") or execution.get("error"),
        "submitted": False,
    }
