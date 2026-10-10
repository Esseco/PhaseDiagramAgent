"""Explicit candidate review commands; generic continue/approve never activates."""

import re
from pathlib import Path
from phase_agent.tools.local.training_handoff import digest_file
from phase_agent.tools.local.recover_remote_training import inspect_training_results
from phase_agent.tools.workflows.create_model_update_handler import create_model_update_handler


def review_candidate_command(message, state, *, state_path=None):
    simple = str(message).strip()
    if simple in {"同意", "拒绝"}:
        eligible = [
            (version, candidate)
            for version, candidate in (state.get("candidate_models") or {}).items()
            if candidate.get("status") == "validated_candidate"
            and candidate.get("old_model_version") == state.get("active_model_version")
            and (candidate.get("agent_review") or {}).get("choice") == "activate"
            and any(
                (job.get("training_handoff") or {}).get("stage") == "awaiting_activation_approval"
                and (job.get("training_handoff") or {}).get("candidate_model_version") == version
                for job in state.get("remote_finetune_jobs", {}).values()
            )
        ]
        if len(eligible) > 1:
            return {"state": state, "reason": "多个模型方案待审批，请明确对象；未激活。"}
        if len(eligible) == 1:
            version, candidate = eligible[0]
            message = (
                ("激活候选" if simple == "同意" else "拒绝候选")
                + " "
                + version
                + " 原因："
                + (
                    candidate["agent_review"]["reason"]
                    if simple == "同意"
                    else "用户拒绝Agent执行方案"
                )
            )
    match = re.fullmatch(
        r"(激活候选|拒绝候选)\s+(\S+)\s+原因[：:]\s*(.+)", str(message).strip(), re.DOTALL
    )
    if not match:
        return None
    command, version, reason = match.groups()
    if not reason.strip():
        return {"state": state, "reason": "请提供明确的审阅原因；未激活。"}
    candidate = (state.get("candidate_models") or {}).get(version)
    if not candidate:
        return {"state": state, "reason": "候选版本不存在；未激活。"}
    if command == "激活候选":
        if candidate.get("status") == "user_rejected":
            return {"state": state, "reason": "该候选已被拒绝；需重新审阅，未激活。"}
        if state.get("active_model_version") == version:
            return {"state": state, "reason": "该候选已经激活；未重复执行。"}
        if state.get("active_model_version") != candidate.get("old_model_version"):
            return {"state": state, "reason": "当前旧模型版本已改变，需重新验证；未激活。"}
        jobs = [
            job
            for job in (state.get("remote_finetune_jobs") or {}).values()
            if (job.get("training_handoff") or {}).get("candidate_model_version") == version
        ]
        if len(jobs) != 1:
            return {"state": state, "reason": "候选训练来源不唯一；未激活。"}
        job = jobs[0]
        handoff = job["training_handoff"]
        result = inspect_training_results(job)
        cv = handoff.get("evidence_type") == "grouped_cross_validation_review"
        artifact = (
            Path(job["directory"])
            / "results"
            / ("cv_baseline_review.json" if cv else "validation-" + handoff["request_id"] + ".json")
        )
        expected_digest = handoff.get("review_sha256") if cv else handoff.get("validation_sha256")
        if (
            not result
            or result.get("fingerprint") != handoff.get("training_fingerprint")
            or not artifact.is_file()
            or digest_file(artifact) != expected_digest
        ):
            return {"state": state, "reason": "训练或验证回传已改变；请先继续重新回收，未激活。"}
    if state_path:
        jobs = [
            job
            for job in state.get("remote_finetune_jobs", {}).values()
            if (job.get("training_handoff") or {}).get("candidate_model_version") == version
        ]
        proposal = (
            (jobs[0].get("training_handoff") or {}).get("direction_proposal")
            if len(jobs) == 1
            else None
        )
        if not proposal:
            return {"state": state, "reason": "请先继续，让Agent生成持久化方向方案；未执行。"}
        from phase_agent.graphs.direction_review_graph import direction_checkpoint

        checkpoint = direction_checkpoint(
            state_path, proposal, "approve" if command == "激活候选" else "reject"
        )
        expected = "approved" if command == "激活候选" else "rejected"
        if checkpoint.get("status") != expected:
            return {"state": state, "reason": "方向审批尚未完成；未执行。"}
    trigger = {
        "action": "ACTIVATE_CANDIDATE_MODEL" if command == "激活候选" else "REJECT_CANDIDATE_MODEL",
        "candidate_model_version": version,
        "user_approval_reason"
        if command == "激活候选"
        else "user_rejection_reason": reason.strip(),
    }
    result = create_model_update_handler()(trigger=trigger, state=state, manager=None, config={})
    updated = result["state"]
    if result["status"] == "activated":
        for job in updated.get("remote_finetune_jobs", {}).values():
            if (job.get("training_handoff") or {}).get("candidate_model_version") == version:
                job["activated"] = True
                job["training_handoff"]["stage"] = "activated"
    return {
        "state": updated,
        "reason": f"候选 {version}：{result['status']}。"
        + ("下一步说“继续”，审阅新模型结构刷新方案。" if result["status"] == "activated" else ""),
    }
