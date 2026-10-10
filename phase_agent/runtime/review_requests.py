"""One read-only queue for existing approval owners; no parallel executor."""

from copy import deepcopy
from phase_agent.tools.policy.file_approval import proposal_hash


def additional_reviews(state, session=None):
    rows = []
    for job_id, job in (state.get("remote_finetune_jobs") or {}).items():
        handoff = job.get("training_handoff") or {}
        proposal = handoff.get("direction_proposal") or {}
        version = handoff.get("candidate_model_version")
        candidate = (state.get("candidate_models") or {}).get(version) or {}
        stage = handoff.get("stage")
        if stage == "awaiting_activation_approval":
            if (
                candidate.get("status") != "validated_candidate"
                or candidate.get("old_model_version") != state.get("active_model_version")
                or (candidate.get("agent_review") or {}).get("choice") != "activate"
            ):
                continue
            kind, title, scope = (
                "model_activation",
                "切换候选模型",
                f"{state.get('active_model_version')} → {version}",
            )
            boundary = "仅切换模型；结构刷新、搜索及提交另行审批"
        elif stage == "awaiting_direction_approval":
            kind, title, scope = (
                "training_direction",
                "审阅科研方向",
                str(proposal.get("direction")),
            )
            boundary = "仅确认保存的方向；具体科学动作另行审批"
        else:
            continue
        if (
            not proposal
            or job.get("activated")
            or handoff.get("direction_status") in {"rejected", "completed"}
        ):
            continue
        identity = {
            "kind": kind,
            "job_id": job_id,
            "proposal": deepcopy(proposal),
            "candidate_status": candidate.get("status"),
            "active_model_version": state.get("active_model_version"),
        }
        rows.append(
            _row(
                f"review:{kind}:{job_id}",
                kind,
                identity,
                title,
                scope,
                boundary,
                proposal.get("reason") or handoff.get("reason"),
                sensitive=kind == "model_activation",
                version=version,
            )
        )
    for plan_id, record in (state.get("pending_execution_recoveries") or {}).items():
        proposal = record["proposal"]
        no_effect = proposal["resolution"] == "verified_no_effect"
        row = _row(
            plan_id,
            "execution_recovery",
            deepcopy(proposal),
            "结束经核对无副作用的旧动作" if no_effect else "复用已登记任务与结果",
            f"旧动作 {proposal['identity']['invocation_id']}；仅修复恢复记录",
            "原回执保留，旧调用禁止重放；不提交任务、不改科学结果。"
            + (
                "仅释放该动作尚未结算的预算预留。"
                if no_effect
                else "已有任务继续回收，预算不在此结算。"
            ),
            proposal["attestation"]["evidence_note"],
            sensitive=True,
        )
        row["revision"] = record.get("revision", 0)
        row["review_card"]["evidence_refs"] = proposal["checks"]
        if no_effect:
            released = sum(
                item.get("reserved_cost", item.get("relative_cost", 0))
                for item in proposal["checks"]["budget_reservations"]
                if item.get("status") != "settled"
            )
            row["review_card"]["scope"] += f"；释放相对成本预留 {released:g}"
        rows.append(row)
    session = session or {}
    if session.get("status") == "draft":
        rows.append(
            _row(
                "review:configuration",
                "configuration",
                deepcopy(session),
                "确认配置草稿",
                f"草稿修订 {session.get('draft_revision', 0)}",
                "仅保存确认配置；不准备输入、不计算、不激活模型",
                "请核对当前完整配置",
                sensitive=False,
            )
        )
    for row in rows:
        if row["review_kind"] == "configuration":
            row["revision"] = session.get("draft_revision", 0)
    return rows


def _row(plan_id, kind, identity, title, scope, boundary, reason, *, sensitive, version=None):
    return {
        "plan_id": plan_id,
        "invocation_id": plan_id,
        "review_kind": kind,
        "review_identity": identity,
        "revision": 0,
        "recommended_action": kind,
        "target_ids": [version] if version else [],
        "parameters": identity,
        "reason": reason,
        "estimated_cost": 0.0,
        "proposal_hash": proposal_hash(identity),
        "review_card": {
            "title": title,
            "scope": scope,
            "reason": reason,
            "purpose": title,
            "budget": 0,
            "sensitive": sensitive,
            "approval_boundary": boundary,
        },
    }


def review_additional(handler, state, row, decision, comment):
    from phase_agent.tools.step_runner.file_protocol import write_json

    kind = row["review_kind"]
    if kind == "execution_recovery":
        if decision == "modify":
            raise ValueError("请在核对卡片修改依据并重新生成恢复方案；旧方案不会自动批准")
        from filelock import FileLock
        from pathlib import Path
        from phase_agent.tools.state.execution_reconciliation import apply_reconciliation
        from phase_agent.tools.step_runner.file_protocol import read_json

        with FileLock(
            str(Path(handler.state_path).parent / "langgraph_lifecycle.sqlite") + ".lock",
            timeout=10,
        ):
            current = read_json(handler.state_path, {}) or {}
            fresh = (current.get("pending_execution_recoveries") or {}).get(row["plan_id"]) or {}
            if proposal_hash(fresh.get("proposal") or {}) != row["proposal_hash"]:
                raise ValueError("recovery_proposal_changed; 请重新查看处理方案")
            result = apply_reconciliation(
                handler.state_path, current, row["plan_id"], decision, reviewer_comment=comment
            )
            write_json(handler.state_path, result["state"])
        return {"status": result["status"], "reason": result.get("reason")}
    if kind in {"model_activation", "training_direction"} and decision != "reject":
        from phase_agent.tools.state.execution_receipts import recovery_report

        if recovery_report(handler.state_path, state)["unsettled"]:
            raise ValueError("请先核对中断执行，再审批模型或科研方向；未执行。")
    if kind == "configuration":
        from phase_agent.runtime.local_agent_control import LocalAgentControl

        control = LocalAgentControl(handler)
        if decision == "modify":
            import json

            try:
                patch = json.loads(comment)
            except ValueError:
                raise ValueError(
                    '配置修改请填写 JSON，例如 {"run.batch_size": 100}，或在聊天中描述修改要求'
                ) from None
            return control.patch_config(patch)
        if decision == "reject":
            return {"status": "draft_not_confirmed", "reason": "配置未确认，草稿保留供修改。"}
        return control.confirm_config(explicit=True)
    if decision == "modify":
        if not str(comment).strip():
            raise ValueError("请填写修改要求")
        result = handler._run(row["plan_id"], None, comment)
        return {"status": result.get("status"), "result": result}
    if kind == "model_activation" and decision == "approve":
        raise ValueError("模型切换需要 confirm_sensitive，先核对切换范围")
    if kind == "model_activation":
        from phase_agent.tools.local.review_candidate_command import review_candidate_command

        version = row["target_ids"][0]
        command = "拒绝候选" if decision == "reject" else "激活候选"
        result = review_candidate_command(
            f"{command} {version} 原因：{comment.strip() or row['reason']}",
            state,
            state_path=handler.state_path,
        )
        updated = result["state"]
        status = (
            "activated"
            if updated.get("active_model_version") == version
            else (
                "rejected_by_user"
                if (updated.get("candidate_models", {}).get(version) or {}).get("status")
                == "user_rejected"
                else "not_executed"
            )
        )
    else:
        from phase_agent.tools.local.review_direction_command import review_direction_command

        # This historical adapter accepts only one pending direction. Do not approve
        # a different object when the queue contains ambiguous old records.
        result = review_direction_command(
            "拒绝" if decision == "reject" else "同意", state, handler.state_path
        )
        updated = result["state"]
        status = "direction_reviewed" if updated != state else "not_executed"
    write_json(handler.state_path, updated)
    return {"status": status, "result": result}


def pending_reviews(state, session=None):
    """Versioned queue shared by HTTP, chat and Studio status."""
    from phase_agent.runtime.review_presentation import review_card
    from phase_agent.tools.step_runner.build_status_summary import build_status_summary

    version = build_status_summary(state, config_version=state.get("confirmed_config_version"))[
        "summary_id"
    ]
    rows = []
    for key, record in (state.get("pending_execution_policies") or {}).items():
        proposal = record.get("agent_proposal") or {}
        rows.append(
            {
                "plan_id": key,
                "invocation_id": key,
                "review_kind": "scientific_action",
                "revision": record.get("revision", 0),
                "review_card": review_card(key, record),
                "recommended_action": proposal.get("recommended_action"),
                "target_ids": ((proposal.get("raw_action") or {}).get("target_ids") or []),
                "parameters": proposal.get("action_parameters"),
                "reason": proposal.get("reason"),
                "estimated_cost": proposal.get("estimated_cost"),
                "missing_evidence": proposal.get("missing_evidence"),
                "proposal_hash": proposal_hash(proposal),
            }
        )
    rows.extend(additional_reviews(state, session))
    for row in rows:
        row.update(
            state_version=version,
            config_version=state.get("confirmed_config_version"),
            model_version=state.get("active_model_version"),
        )
    return rows
