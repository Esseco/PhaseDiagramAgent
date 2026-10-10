"""Activation barrier using the existing approval, task and budget interfaces."""

from copy import deepcopy
import hashlib
from pathlib import Path

from phase_agent.analysis.feedback.model_refresh_plan import supplemental_relaxation_required
from phase_agent.tools.budget.release_budget import release_budget
from phase_agent.tools.policy.execution_policy import apply_execution_policy, build_agent_proposal
from phase_agent.tools.policy.validate_tool_action import validate_tool_action
from phase_agent.tools.dispatch.execute_tool_action import execute_tool_action
from phase_agent.tools.workflows.model_refresh_state import refresh_active, refresh_preview


def advance_refresh(state, user_message=""):
    current = deepcopy(state)
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

    missing = [tid for tid in refresh["task_ids"] if not accepted(tasks.get(tid, {}))]
    if missing and user_message.strip() == "不再等待刷新":
        refresh["waived_task_ids"] = sorted(set(refresh.get("waived_task_ids", [])) | set(missing))
        for tid in missing:
            if tid in tasks:
                tasks[tid]["refresh_wait_waived"] = True
    unresolved = set(missing) - set(refresh.get("waived_task_ids", []))
    if unresolved:
        return (
            current,
            f"刷新第{refresh['wave'] + 1}批：有效回收{len(rows) - len(missing)}/{len(rows)}；等待或失败{len(unresolved)}。可继续回传，或明确回复“不再等待刷新”。不会标记缺失任务完成。",
        )
    plan = refresh["plan"]
    diagram = (current.get("phase_diagrams") or {}).get("mlip") or {}
    if (
        diagram.get("status") != "completed"
        or diagram.get("model_version") != refresh["new_model_version"]
    ):
        return current, "刷新结果尚不能构建新模型相图，请补齐相识别或Na端点；不会沿用旧模型能量。"
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
            return current, f"刷新任务{tid}存在多个相图记录，需核对唯一结果；尚未完成刷新。"
    entries = {e["record_id"]: e for e in diagram_entries}
    phases = {
        r.get("source_task_id"): entries.get(r.get("record_id"))
        for r in current.get("phase_records") or []
        if r.get("source_version") == refresh["new_model_version"]
    }
    for task in rows:
        if accepted(task) and phases.get(task["task_id"]) is None:
            return current, f"刷新任务{task['task_id']}尚未进入新版本相图，需完成相识别和能量校验。"
    supplements = []
    if refresh["wave"] == 0:
        original = {r["structure_sha256"]: r for r in plan["candidates"]}
        for task in rows:
            if (
                not accepted(task)
                or (task.get("parameters") or {}).get("model_refresh_operation") != "predict"
            ):
                continue
            entry = phases.get(task["task_id"])
            if entry is None or entry.get("ehull_unit") != "eV/atom":
                return current, f"刷新任务{task['task_id']}缺少唯一新版本Ehull证据，未追加计算。"
            outputs = task["outputs"]
            if supplemental_relaxation_required(float(entry["ehull"]), outputs.get("forces")):
                path = Path(outputs["structure_path"])
                source = original[task["input_structure_sha256"]]
                supplements.append(
                    {
                        **source,
                        "operation": "relax",
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
            current, task_key=key, reason="refresh supplemental envelope completed or unused"
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
        return {"status": "not_configured", "state": current, "reason": waiting, "submitted": False}
    if not refresh_active(current):
        return {
            "status": "completed",
            "state": current,
            "submitted": False,
            "reason": "新模型结构刷新已完成；下一次继续将评估微调、补DFT、新branch或收敛建议。",
        }
    config = context["effective_config"]
    refresh = current["model_refresh"]
    automatic = refresh.get("status") == "supplement_ready"
    try:
        plan = (
            refresh["plan"] if automatic else refresh_preview(current, context["manager"], config)
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
        "target_ids": [r["structure_id"] for r in plan["candidates"]],
        "decision_source": "approved_refresh_policy",
        "reason": "新模型激活后先刷新累计结构；不混用旧模型能量。",
        "parameters": {
            "mode": "model_refresh_inputs",
            "preview_checksum": plan["checksum"],
            "refresh_preview": plan,
        },
    }
    proposal = build_agent_proposal(action, current, runtime_state=current)
    key = pending_key
    stored = (current.get("pending_execution_policies") or {}).get(key) or {}
    previous = ((stored.get("agent_proposal") or {}).get("raw_action") or {}).get("task_key")
    # A previous action's approval must never approve a new refresh or changed files.
    if not automatic and previous != action["task_key"]:
        human_feedback = None
    policy = apply_execution_policy(
        proposal,
        execution_mode="autonomous"
        if automatic
        else ("dry_run" if mode == "dry_run" else "interactive"),
        human_feedback=human_feedback,
    )
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
    if not validation["valid"]:
        return {
            "status": "rejected",
            "state": current,
            "reason": "刷新方案校验失败，未截断：" + str(validation["errors"]),
            "submitted": False,
        }
    execution = execute_tool_action(
        action,
        registry=registry,
        context={
            **context,
            "event_state": current,
            "config_version": session["confirmed_snapshot"]["config_version"],
            "approval_record_id": action["task_key"],
        },
    )
    result = execution.get("result") or {}
    updated = result.get("state", current)
    if result.get("status") == "prepared":
        updated.setdefault("pending_execution_policies", {}).pop(key, None)
    updated.setdefault("action_records", []).append(
        {
            "record_id": action["task_key"],
            "final_action": action,
            "status": result.get("status", execution["status"]),
            "approval_basis": "bounded_initial_approval" if automatic else "explicit_user_approval",
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
