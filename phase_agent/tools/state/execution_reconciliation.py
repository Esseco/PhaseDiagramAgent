"""Audited administrative reconciliation; never dispatch or reconstruct science."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from phase_agent.graphs.contracts import ExecutionIdentity
from phase_agent.tools.policy.file_approval import proposal_hash
from phase_agent.tools.state.execution_recovery_evidence import inspect_execution
from phase_agent.tools.state.execution_receipts import recovery_report

RESOLUTIONS = {"verified_no_effect", "verified_registered_effects"}


def business_fingerprint(state):
    excluded = {"pending_execution_recoveries", "execution_recovery_report"}
    payload = {key: value for key, value in state.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _issue(state_path, state, invocation):
    matches = [
        row
        for row in recovery_report(state_path, state)["unsettled"]
        if (row.get("identity") or {}).get("invocation_id") == invocation
        and (row.get("identity") or {}).get("action_hash")
    ]
    if len(matches) != 1:
        raise ValueError("未找到唯一完整执行身份；先核对原回执与业务记录")
    ExecutionIdentity.model_validate(matches[0]["identity"])
    return matches[0]


def _validate_checks(checks, resolution, attestation):
    if resolution not in RESOLUTIONS:
        raise ValueError("unsupported recovery resolution")
    if not str(attestation.get("evidence_note") or "").strip():
        raise ValueError("请填写实际核对依据；缺少文件不能证明没有副作用")
    if attestation.get("external_jobs") not in {"none_found", "registered_only", "not_applicable"}:
        raise ValueError("需核对外部作业；未知作业不能解除阻塞")
    if checks.get("unregistered_returned_task_ids"):
        raise ValueError("工具返回的任务尚未完整登记；先核对并修复登记，不能直接解除阻塞")
    bad_paths = {"missing", "unreadable", "relative_path_requires_owner"}
    if any(row["status"] in bad_paths for row in checks["artifact_checks"]):
        raise ValueError("先修复或核对缺失文件及路径迁移；不能直接解除阻塞")
    if resolution == "verified_no_effect":
        import math

        reserved = 0.0
        for row in checks["budget_reservations"]:
            amount = row.get("reserved_cost", row.get("relative_cost"))
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(amount)
                or amount < 0
            ):
                raise ValueError("预算预留金额不完整或非法；先核对预算台账")
            if row.get("status") == "settled":
                if row.get("accounted_cost", 0):
                    raise ValueError("旧动作已有结算费用；不能作为零成本无副作用处理")
            else:
                reserved += amount
        total = checks["reserved_relative_cost"]
        if (
            isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(total)
            or total < reserved
        ):
            raise ValueError("动作预留与总预留不一致；先核对预算台账")
        if (
            checks["task_count"]
            or checks["phase_record_ids"]
            or any(row["status"] == "file" for row in checks["artifact_checks"])
        ):
            raise ValueError("存在已登记任务或文件，不能声明无副作用；应核对并复用现有工作")
        if (
            attestation.get("external_jobs") not in {"none_found", "not_applicable"}
            or attestation.get("output_inventory") != "checked_no_outputs"
            or attestation.get("external_cost") != 0
            or isinstance(attestation.get("external_cost"), bool)
        ):
            raise ValueError("无副作用需要明确核对无外部作业、无产出、科学计算成本为零")
    elif attestation.get("output_inventory") != "registered_outputs":
        raise ValueError("复用前需明确核对已有登记任务和产出")
    elif not checks["task_count"]:
        raise ValueError("没有已登记任务可复用；先通过原结果回收流程完成登记，不凭文件存在推定完成")


def propose_reconciliation(state_path, state, invocation_id, resolution, attestation):
    issue = _issue(state_path, state, invocation_id)
    checks = inspect_execution(state, issue, state_path=state_path)
    _validate_checks(checks, resolution, attestation)
    from uuid import uuid4

    matching = [
        (key, value)
        for key, value in (state.get("pending_execution_recoveries") or {}).items()
        if value["proposal"]["identity"]["invocation_id"] == invocation_id
    ]
    plan_id, old = matching[0] if matching else ("review:recovery:" + uuid4().hex, {})
    plan = {
        "identity": deepcopy(issue["identity"]),
        "resolution": resolution,
        "attestation": deepcopy(attestation),
        "checks": checks,
        "business_fingerprint": business_fingerprint(state),
    }
    updated = deepcopy(state)
    if old.get("proposal") == plan:
        return {"status": "awaiting_recovery_review", "plan_id": plan_id, "state": updated}
    record = {
        "proposal": plan,
        "revision": int(old.get("revision", -1)) + 1,
        "feedback_history": deepcopy(old.get("feedback_history") or []),
    }
    if old:
        record["feedback_history"].append({"prior_proposal": deepcopy(old["proposal"])})
    updated.setdefault("pending_execution_recoveries", {})[plan_id] = record
    return {"status": "awaiting_recovery_review", "plan_id": plan_id, "state": updated}


def apply_reconciliation(state_path, state, plan_id, decision, *, reviewer_comment=""):
    record = (state.get("pending_execution_recoveries") or {}).get(plan_id)
    if not record:
        raise ValueError("recovery plan is not pending")
    if decision not in {"confirm_sensitive", "reject"}:
        raise ValueError("恢复处理需要 confirm_sensitive；先核对范围和预算影响")
    updated = deepcopy(state)
    plan = record["proposal"]
    if decision == "reject":
        updated["pending_execution_recoveries"].pop(plan_id)
        updated.setdefault("recovery_review_history", []).append(
            {
                "plan_id": plan_id,
                "proposal": deepcopy(plan),
                "decision": "reject",
                "reviewer_comment": str(reviewer_comment).strip(),
            }
        )
        updated.setdefault("invocations", {})[plan_id] = {"status": "recovery_rejected"}
        return {"status": "recovery_rejected", "state": updated}
    if plan["business_fingerprint"] != business_fingerprint(state):
        raise ValueError("recovery_state_changed; 核对期间业务状态已变化，请重新生成处理方案")
    invocation = plan["identity"]["invocation_id"]
    issue = _issue(state_path, state, invocation)
    checks = inspect_execution(state, issue, state_path=state_path)
    if issue["identity"] != plan["identity"] or checks != plan["checks"]:
        raise ValueError("recovery_evidence_changed; 文件、任务或回执已变化，请重新核对")
    _validate_checks(checks, plan["resolution"], plan["attestation"])
    keys = checks["task_keys"]
    if plan["resolution"] == "verified_no_effect":
        from phase_agent.tools.budget.settle_budget import settle_budget

        for key in keys:
            updated = settle_budget(
                updated,
                task_key=key,
                settlement_id=plan_id,
                task_status="failed",
                actual_cost=0,
                cost_source="human_verified_no_effect",
                failure_reason=plan["attestation"]["evidence_note"],
            )["state"]
            current = updated.setdefault("effective_decisions", {}).get(key)
            if current:
                current.update(status="failed")
                if key in updated.get("budget_reservations", {}):
                    current["verified_no_effect"] = True
            reservation = updated.get("budget_reservations", {}).get(key)
            if reservation:
                reservation.pop("reconciliation_required", None)
        recovered_status = "reconciled_no_effect"
    else:
        # Registered tasks/results stay authoritative; collectors settle costs.
        recovered_status = "reconciled_existing_effects"
        for key in keys:
            current = updated.setdefault("effective_decisions", {}).get(key)
            if current:
                current.update(
                    status="pending"
                    if checks["pending_task_ids"]
                    else "completed"
                    if len(checks["completed_task_ids"]) == checks["task_count"]
                    else "failed"
                )
            reservation = updated.get("budget_reservations", {}).get(key)
            if reservation:
                reservation.pop("reconciliation_required", None)
    audit = {
        "resolution": plan["resolution"],
        "identity": deepcopy(plan["identity"]),
        "evidence": {"checks": checks, "attestation": deepcopy(plan["attestation"])},
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan_id,
        "proposal_hash": proposal_hash(plan),
        "reviewer_comment": str(reviewer_comment).strip(),
    }
    updated.setdefault("execution_reconciliations", {})[invocation] = audit
    for key, pending in list((updated.get("pending_execution_policies") or {}).items()):
        action = (pending.get("agent_proposal") or {}).get("raw_action") or {}
        if key == invocation or action.get("task_key") in keys:
            updated.setdefault("reconciled_proposals", []).append(
                {"plan_id": key, "record": deepcopy(pending), "reconciliation": plan_id}
            )
            updated["pending_execution_policies"].pop(key)
    updated.setdefault("invocations", {}).setdefault(
        invocation,
        {
            "status": recovered_status,
            "record_id": invocation,
            "reconciliation_id": plan_id,
            "submitted": False,
        },
    )
    updated["pending_execution_recoveries"].pop(plan_id)
    updated.setdefault("invocations", {})[plan_id] = {
        "status": recovered_status,
        "reconciliation_id": invocation,
    }
    updated.setdefault("recovery_review_history", []).append(
        {"plan_id": plan_id, "decision": "confirm_sensitive", "audit": audit}
    )
    return {
        "status": recovered_status,
        "state": updated,
        "submitted": False,
        "reason": "核对记录已保存，原回执保留，旧动作禁止重放；下一项科学动作仍需单独审批。",
    }
