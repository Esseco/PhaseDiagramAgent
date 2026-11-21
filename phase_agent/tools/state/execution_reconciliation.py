"""Audited administrative reconciliation; never dispatch or reconstruct science."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from phase_agent.graphs.contracts import ExecutionIdentity
from phase_agent.policies.file_approval import proposal_hash
from phase_agent.tools.state.execution_recovery_evidence import inspect_execution
from phase_agent.tools.state.execution_receipts import recovery_report

RESOLUTIONS = {"verified_no_effect", "verified_registered_effects", "verified_abandoned_staging"}


def business_fingerprint(state):
    excluded = {"pending_execution_recoveries", "execution_recovery_report"}
    payload = {key: value for key, value in state.items() if key not in excluded}
    digest = hashlib.sha256()
    for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=False, default=str).iterencode(payload):
        digest.update(chunk.encode("utf8"))
    return digest.hexdigest()


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



def _attach_legacy_staging(state, identity, checks, *, state_path=None, issue=None, attestation=None):
    invocation = identity.get("invocation_id")
    record = next((r for r in state.get("action_records") or [] if r.get("record_id") == invocation), {})
    action = record.get("final_action") or {}
    from phase_agent.policies.region_sampling import is_region

    if (state_path and identity.get("tool") == "generate_branches" and is_region(action)
            and (record.get("execution_result") or {}).get("error") == "MemoryError: "):
        from phase_agent.tools.state.abandoned_region_generation import inspect_abandoned_region_generation

        staging = inspect_abandoned_region_generation(state_path, state, identity)
        if staging is not None:
            checks["abandoned_staging"] = staging
    if (action.get("operation") == "prepare_perturbation_recovery"
            and (record.get("execution_result") or {}).get("error") == "ValueError: perturbation_upload_incomplete"):
        from phase_agent.tools.state.abandoned_perturbation import inspect_abandoned_perturbation
        checks["abandoned_staging"] = inspect_abandoned_perturbation(state, identity)
    if state_path and issue and identity.get("tool") == "prepare_local_batch_files":
        from phase_agent.tools.state.execution_receipts import execution_events
        events = execution_events(state_path, invocation)
        actions = [row["payload"].get("action") or {} for row in events if row["event"] == "claimed"]
        if len(actions) == 1 and (((actions[0].get("parameters") or {}).get("refresh_preview") or {}).get("refresh_scope") == "frozen_baseline_hull_vertices"):
            from phase_agent.tools.state.abandoned_refresh import inspect_abandoned_refresh
            checks["abandoned_staging"] = inspect_abandoned_refresh(state_path, state, issue)
    if state_path and issue and (attestation or {}).get("staging_directory"):
        from phase_agent.tools.state.abandoned_exact_relax import inspect_abandoned_exact_relax

        checks["abandoned_staging"] = inspect_abandoned_exact_relax(
            state_path, state, issue, attestation["staging_directory"]
        )
    return checks

def _validate_checks(checks, resolution, attestation):
    if resolution not in RESOLUTIONS:
        raise ValueError("unsupported recovery resolution")
    if not str(attestation.get("evidence_note") or "").strip():
        raise ValueError("请填写实际核对依据；缺少文件不能证明没有副作用")
    if attestation.get("external_jobs") not in {"none_found", "registered_only", "not_applicable"}:
        raise ValueError("需核对外部作业；未知作业不能解除阻塞")
    staging = checks.get("abandoned_staging") or {}
    if resolution == "verified_abandoned_staging" and staging.get("kind") == "abandoned_unsubmitted_refresh_staging":
        if (staging.get("submitted") is not False
                or sorted(checks.get("unregistered_returned_task_ids") or []) != staging.get("task_ids")
                or checks.get("task_count") or checks.get("budget_reservations")
                or attestation.get("external_jobs") != "none_found"
                or attestation.get("output_inventory") != "abandoned_local_staging"
                or type(attestation.get("external_cost")) not in (int, float)
                or attestation.get("external_cost") != 0):
            raise ValueError("refresh_staging_requires_verified_local_only_evidence")
        return
    if resolution == "verified_abandoned_staging" and staging.get("kind") == "abandoned_exact_candidate_relax_staging":
        if (staging.get("submitted") is not False or checks.get("task_count")
                or checks.get("unregistered_returned_task_ids") or checks.get("phase_record_ids")
                or checks.get("budget_reservations")
                or attestation.get("external_jobs") != "not_applicable"
                or attestation.get("output_inventory") != "abandoned_local_staging"
                or type(attestation.get("external_cost")) not in (int, float)
                or attestation.get("external_cost") != 0):
            raise ValueError("exact_staging_requires_verified_local_only_evidence")
        return
    if resolution == "verified_abandoned_staging" and staging.get("kind") == "abandoned_region_generation_staging":
        reservations = checks.get("budget_reservations") or []
        outputs = checks.get("registered_branch_outputs") or []
        if (staging.get("submitted") is not False or checks.get("task_count")
                or checks.get("unregistered_returned_task_ids") or checks.get("phase_record_ids")
                or checks.get("artifact_checks") or len(outputs) != 1
                or outputs[0].get("task_key") != staging.get("task_key")
                or outputs[0].get("status") != "invalid"
                or len(reservations) != 1
                or reservations[0].get("task_key") != staging.get("task_key")
                or reservations[0].get("status") != "reserved"
                or type(reservations[0].get("reserved_cost", reservations[0].get("relative_cost"))) not in (int, float)
                or reservations[0].get("reserved_cost", reservations[0].get("relative_cost")) != 0
                or attestation.get("external_jobs") != "not_applicable"
                or attestation.get("output_inventory") != "abandoned_local_staging"
                or type(attestation.get("external_cost")) not in (int, float)
                or attestation.get("external_cost") != 0):
            raise ValueError("region_staging_requires_verified_local_only_evidence")
        return
    if checks.get("unregistered_returned_task_ids"):
        raise ValueError("工具返回的任务尚未完整登记；先核对并修复登记，不能直接解除阻塞")
    branch_outputs = checks.get("registered_branch_outputs") or []
    if any(row.get("status") != "verified" for row in branch_outputs):
        raise ValueError("branch产出与账本、身份或源文件不一致；不能解除恢复阻塞")
    bad_paths = {"missing", "unreadable", "relative_path_requires_owner"}
    if any(row["status"] in bad_paths for row in checks["artifact_checks"]):
        raise ValueError("先修复或核对缺失文件及路径迁移；不能直接解除阻塞")
    if resolution == "verified_abandoned_staging":
        staging = checks.get("abandoned_staging") or {}
        if (staging.get("kind") != "abandoned_legacy_perturbation_staging"
                or staging.get("submitted") is not False
                or checks.get("task_count") or checks.get("registered_branch_outputs")
                or checks.get("budget_reservations")
                or attestation.get("external_jobs") not in {"none_found", "not_applicable"}
                or attestation.get("output_inventory") != "abandoned_local_staging"
                or attestation.get("external_cost") != 0
                or isinstance(attestation.get("external_cost"), bool)):
            raise ValueError("abandoned_staging_requires_verified_local_only_evidence")
        return
    if resolution == "verified_no_effect":
        if checks.get("abandoned_staging"):
            raise ValueError("abandoned_staging_is_not_no_effect")
        import math

        if (checks.get("preparation_progress") or {}).get("boundary") == "complete":
            raise ValueError("输入准备已进入文件写入；先核对并恢复产出，不能声明无副作用")
        if (checks.get("preparation_progress") or {}).get("boundary") == "effects_possible":
            note = str(attestation.get("evidence_note") or "")
            if "归档" not in note and "清理" not in note:
                raise ValueError("部分输入已写入；请先归档或清理现场并在核对依据中说明")
        if checks.get("preparation_progress") and any(
            row["status"] == "directory" for row in checks["artifact_checks"]
        ):
            raise ValueError("输入准备目录已存在；先核对其中产出，不能声明无副作用")

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
            or branch_outputs
            or checks["phase_record_ids"]
            or any(row["status"] in {"file", "directory"} for row in checks["artifact_checks"])
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
    elif not checks["task_count"] and not any(
        row.get("registered_structure_count", 0) for row in branch_outputs
    ):
        raise ValueError("没有已登记任务可复用；先通过原结果回收流程完成登记，不凭文件存在推定完成")


def propose_reconciliation(state_path, state, invocation_id, resolution, attestation):
    issue = _issue(state_path, state, invocation_id)
    checks = _attach_legacy_staging(state, issue["identity"],
                                    inspect_execution(state, issue, state_path=state_path),
                                    state_path=state_path, issue=issue, attestation=attestation)
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
    if (checks.get("abandoned_staging") or {}).get("kind") == "abandoned_exact_candidate_relax_staging":
        updated = dict(state)
        updated["pending_execution_recoveries"] = deepcopy(state.get("pending_execution_recoveries") or {})
    else:
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
    plan = record["proposal"]
    if (plan.get("checks", {}).get("abandoned_staging") or {}).get("kind") == "abandoned_exact_candidate_relax_staging":
        updated = dict(state)
        names = ("pending_execution_recoveries", "recovery_review_history", "invocations", "execution_reconciliations",
                 "pending_execution_policies", "reconciled_proposals", "effective_decisions", "budget_reservations")
        updated.update(deepcopy({k:state[k] for k in names if k in state}))
    else:
        updated = deepcopy(state)
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
    original_pending = (updated.get("pending_execution_policies") or {}).get(invocation)
    issue = _issue(state_path, state, invocation)
    checks = _attach_legacy_staging(state, issue["identity"],
                                    inspect_execution(state, issue, state_path=state_path),
                                    state_path=state_path, issue=issue, attestation=plan["attestation"])
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
    elif plan["resolution"] == "verified_abandoned_staging":
        # Preserve old files and audit; no science output is registered or reused.
        if (checks.get("abandoned_staging") or {}).get("kind") == "abandoned_region_generation_staging":
            from phase_agent.tools.budget.settle_budget import settle_budget

            for key in keys:
                updated = settle_budget(updated, task_key=key, settlement_id=plan_id,
                                        task_status="failed", actual_cost=0,
                                        cost_source="human_verified_abandoned_staging",
                                        failure_reason=plan["attestation"]["evidence_note"])["state"]
                decision = updated.setdefault("effective_decisions", {}).get(key)
                if decision:
                    decision.update(status="failed", verified_abandoned_staging=True)
                reservation = updated.setdefault("budget_reservations", {}).get(key)
                if reservation:
                    reservation.pop("reconciliation_required", None)
        recovered_status = "reconciled_abandoned_staging"
    else:
        # Registered tasks/results stay authoritative; collectors settle costs.
        recovered_status = "reconciled_existing_effects"
        for key in keys:
            current = updated.setdefault("effective_decisions", {}).get(key)
            if current:
                current.update(
                    status="pending"
                    if checks["pending_task_ids"]
                    else "superseded_after_registered_effects"
                    if checks.get("registered_branch_outputs") and not checks["task_count"]
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
    outputs_to_adopt = [] if plan["resolution"] == "verified_abandoned_staging" else checks.get("registered_branch_outputs") or []
    for output in outputs_to_adopt:
        gate = updated.get("dedup_gate") or {}
        if gate.get("status") in {None, "ready"}:
            updated["dedup_gate"] = {
                "status": "ready",
                "source": "verified_generation_reconciliation",
                "valid_structure_ids": sorted(
                    set(gate.get("valid_structure_ids") or []) | set(output["structure_ids"])
                ),
            }
        if any(
            row.get("task_key") == output["task_key"]
            for row in updated.get("generation_history") or []
        ):
            continue
        updated.setdefault("generation_history", []).append(
            {
                "task_key": output["task_key"],
                "model_version": state.get("active_model_version"),
                "refresh_plan_checksum": ((state.get("model_refresh") or {}).get("plan") or {}).get(
                    "checksum"
                ),
                "new_branch_ids": output["new_branch_ids"],
                "registered_ids": output["structure_ids"],
                "summary": {
                    "registered_branches": output["registered_branch_count"],
                    "registered_structures": output["registered_structure_count"],
                    "unique_structures": output["registered_structure_count"],
                },
                "artifacts": {"json": output["artifact_path"]},
                "reconciliation_id": plan_id,
                "recovery_scope": "registered_branch_component_only; downstream scope not completed",
            }
        )
    updated.setdefault("execution_reconciliations", {})[invocation] = audit
    for key, pending in list((updated.get("pending_execution_policies") or {}).items()):
        action = (pending.get("agent_proposal") or {}).get("raw_action") or {}
        if key == invocation or action.get("task_key") in keys:
            updated.setdefault("reconciled_proposals", []).append(
                {"plan_id": key, "record": deepcopy(pending), "reconciliation": plan_id}
            )
            updated["pending_execution_policies"].pop(key)
    updated.setdefault("invocations", {})[invocation] = {
        "status": recovered_status,
        "record_id": invocation,
        "reconciliation_id": plan_id,
        "submitted": False,
    }
    for record in updated.get("action_records") or []:
        if record.get("record_id") == invocation:
            record["status"] = recovered_status
    updated["pending_execution_recoveries"].pop(plan_id)
    updated.setdefault("invocations", {})[plan_id] = {
        "status": recovered_status,
        "reconciliation_id": invocation,
    }
    updated.setdefault("recovery_review_history", []).append(
        {"plan_id": plan_id, "decision": "confirm_sensitive", "audit": audit}
    )
    restored_plan = None
    if plan["resolution"] == "verified_no_effect":
        from phase_agent.tools.state.recovered_input_plan import reoffer_input_plan

        restored_plan = reoffer_input_plan(
            updated, original_pending, identity=plan["identity"], review_id=plan_id
        )
    return {
        "status": recovered_status,
        "state": updated,
        "submitted": False,
        "restored_plan_id": restored_plan,
        "reason": "核对记录已保存，原回执保留，旧动作禁止重放；下一项科学动作仍需单独审批。",
    }
