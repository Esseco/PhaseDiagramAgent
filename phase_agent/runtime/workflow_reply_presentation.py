"""Render workflow responses without HTTP, runtime assembly or state writes."""

from phase_agent.decisions.agent.dialogue_contract import DIALOGUE_STATUSES
import os
from pathlib import Path

from phase_agent.runtime.chat_approval_rules import (
    is_sensitive_proposal as _is_sensitive_proposal,
)


def _is_model_failure_proposal(proposal):
    from phase_agent.tools.workflows.run_tool_step import _model_failure_action

    return _model_failure_action((proposal or {}).get("raw_action") or {})


def format_workflow_reply(result: dict, state_path, *, verbose=None) -> str:
    from phase_agent.runtime.response_preferences import detailed_response

    proposal = result.get("agent_proposal") or {}
    brief_reply = None
    if (
        not (detailed_response() if verbose is None else verbose)
        and result.get("status") == "awaiting_approval"
        and not _is_sensitive_proposal(proposal)
    ):
        from phase_agent.tools.workflows.mc_coverage_choice import needs_choice

        if not needs_choice(proposal):
            from phase_agent.runtime.concise_dialogue import approval_reply

            brief_reply = approval_reply(proposal, result.get("state") or {})
    reply = (
        brief_reply
        if brief_reply is not None
        else _format_workflow_reply(result, state_path, verbose=verbose)
    )
    # Brief approval cards and special status replies must pass through the same
    # recovered-round presentation as detailed replies.
    reply = _with_post_dft_reply(
        result, reply, verbose=detailed_response() if verbose is None else verbose
    )
    completed = result.get("completed_generation") or {}
    if brief_reply is not None and completed:
        reply = (
            f"结果：已生成 {completed.get('registered_branches', 0)} 个 branch、{completed.get('registered_structures', 0)} 个初始结构。\n"
            + reply
        )
    from phase_agent.runtime.review_file_presentation import file_location_lines

    if result.get("status") == "awaiting_approval" and brief_reply is None:
        from phase_agent.tools.workflows.generation_artifacts import proposal_artifacts

        plan_artifacts = proposal_artifacts(proposal, result.get("state") or {})
        if plan_artifacts:
            reply += "\n" + "\n".join(
                file_location_lines("Branch 分布与构成文件", plan_artifacts["text"])
            )
        from phase_agent.tools.workflows.dft_selection_artifacts import (
            proposal_artifacts as dft_artifacts,
        )

        dft_details = dft_artifacts(proposal, result.get("state") or {})
        if dft_details:
            reply += "\n" + "\n".join(file_location_lines("DFT 选点文件", dft_details["text"]))
    # State retains all past artifacts. Link one only when this turn actually
    # identifies the matching final action; a training return is not a DFT selection.
    current_actions = [
        row.get("final_action")
        or (row.get("action") if (row.get("execution") or {}).get("status") == "completed" else {})
        or {}
        for row in [result, *(result.get("events") or [])]
    ]
    for tool, field, label in (
        ("select_dft_candidates", "qbc_selection_history", "DFT 选点文件"),
        ("generate_branches", "generation_history", "Branch 分布与构成文件"),
    ):
        keys = {
            action["task_key"]
            for action in current_actions
            if action.get("tool") == tool and action.get("task_key")
        }
        if not keys:
            continue
        for row in (result.get("state") or {}).get(field) or []:
            artifact = (row.get("artifacts") or {}).get("json")
            if row.get("task_key") in keys and artifact:
                from phase_agent.tools.workflows.compact_review import review_artifacts

                if Path(artifact).is_file():
                    details = review_artifacts(
                        artifact,
                        kind="branch" if tool == "generate_branches" else "dft",
                        state=result.get("state"),
                    )
                    path = details["text"]
                else:
                    path = artifact
                    reply += "\n需注意：已登记的清单文件当前缺失，请核对文件位置。"
                line = "\n".join(file_location_lines(label, path))
                if line not in reply:
                    reply += "\n" + line
    if detailed_response() or verbose:
        from phase_agent.runtime.evidence_reference_presentation import evidence_reference_lines

        evidence_lines = evidence_reference_lines(proposal.get("raw_action") or {})
        for line in evidence_lines:
            if line not in reply:
                reply += "\n" + line
    usage = (result.get("state") or {}).get("gpu_usage_summary") or {}
    if usage.get("record_count") and "GPU 占用统计" not in reply:
        line = f"GPU 占用统计：已知累计 {usage['total_gpu_hours']:.3f} GPU 小时。"
        if usage.get("unknown_records"):
            line += f"另有 {usage['unknown_records']} 条记录缺少 GPU 耗时，不按零结算。"
        reply += "\n" + line
    return _with_refresh_completion(result, reply)


def _with_refresh_completion(result, reply):
    """Keep this turn's verified refresh facts beside its next approval."""
    facts = result.get("completed_refresh") or next(
        (
            event["completed_refresh"]
            for event in reversed(result.get("events") or [])
            if event.get("completed_refresh")
        ),
        None,
    )
    state = result.get("state") or {}
    if (
        not facts
        or facts.get("model_version") != state.get("active_model_version")
        or (state.get("model_refresh") or {}).get("status") != "completed"
    ):
        return reply
    lines = [
        "本次：已回收新模型结构刷新结果并更新相图。",
        f"结果：已刷新 {facts['refreshed']} 个独立结构；新模型相图已核对，包含 {facts['phase_diagram_entry_count']} 条条目。",
    ]
    if facts.get("partial"):
        lines.append(
            f"覆盖：另有 {facts['deferred']} 个结构未刷新；本次为部分刷新，不代表全面覆盖或科学收敛。"
        )
    if facts.get("discarded_unconverged_count"):
        lines.append(
            f"排除：{facts['discarded_unconverged_count']} 个弛豫未收敛结果已弃用，原始文件与耗时保留；不进入相图，不自动重试。"
        )
    if facts.get("outside_boundary_count"):
        lines.append(
            f"边界：{facts['outside_boundary_count']} 个结果已确认属于搜索边界外，记录保留，不进入当前相图。"
        )
    lines.extend(line for line in reply.splitlines() if line != "已完成：`workflow`。")
    if result.get("status") == "awaiting_approval" and state.get("pending_execution_policies"):
        if not any("确认范围：" in line for line in lines):
            from phase_agent.runtime.review_presentation import review_card

            proposal = result.get("agent_proposal") or {}
            match = next(
                (
                    (key, row)
                    for key, row in state["pending_execution_policies"].items()
                    if row.get("agent_proposal") == proposal
                ),
                None,
            )
            if match:
                lines.append("确认范围：" + review_card(*match)["approval_boundary"])
    if not state.get("pending_execution_policies"):
        if result.get("followup_error") or result.get("followup_note"):
            lines.append(
                "方案说明：" + str(result.get("followup_note") or result["followup_error"])
            )
            lines.append(
                "下一步：下一步方案尚未准备成功，当前尚无待批准方案；请处理上述问题后重新请求下一步方案。"
            )
        elif not any(line.startswith(("下一步：", "建议：")) for line in lines):
            lines.append(
                "下一步：基于刷新后的相图和覆盖缺口评估后续方案；当前尚无待批准方案，请请求下一步具体方案，展示范围与预算后再批准。"
            )
    return "\n".join(lines)


def _format_workflow_reply(result: dict, state_path, *, verbose=None) -> str:
    from phase_agent.tools.workflows.mc_coverage_choice import needs_choice, prompt

    if result.get("status") == "awaiting_approval" and needs_choice(result.get("agent_proposal")):
        return (
            result.get("reason")
            if result.get("mc_coverage_question")
            else prompt(result["agent_proposal"])
        )
    if result.get("status") == "awaiting_reference_dft_choice":
        return result["reason"]
    if result.get("status") == "insufficient_evidence":
        payload = (result.get("execution_result") or result.get("execution") or {}).get(
            "result"
        ) or result
        if not payload.get("missing_evidence"):
            for event in reversed(result.get("events") or []):
                data = (event.get("execution_result") or event.get("execution") or {}).get(
                    "result"
                ) or event
                if data.get("missing_evidence"):
                    payload = data
                    break
        labels = {
            "consecutive_unique_model_update_epochs": "连续不同模型轮次的收敛记录",
            "model_epoch_hull_change": "模型轮次凸包变化",
            "model_epoch_ground_state_stability": "模型轮次基态稳定性",
            "final_frame_dft_error": "末帧 DFT 误差验证",
            "competing_phase_same_frame_dft": "竞争相的同帧 DFT 证据",
        }
        missing = "、".join(
            labels.get(value, value) for value in payload.get("missing_evidence") or []
        )
        return (
            "收敛检查已完成，当前证据不足。"
            + (
                "缺少：" + missing + "。"
                if missing
                else "原因：" + str(payload.get("reason") or "必要证据未记录") + "。"
            )
            + "\n应基于已有结果提出可行的下一步方案；重复“同意”或检查同一批数据不会补齐证据。"
        )
    recovered = result.get("verified_recovery") or {}
    if recovered:
        names = {
            "relax_and_feature": "Relax",
            "deep_search": "MC",
            "dft_single_point": "DFT 单点",
            "dft_relax": "DFT 弛豫",
        }
        summary = "；".join(
            f"{names.get(stage, stage)} 完成 {counts['completed']}、失败/中止 {counts['failed']}"
            for stage, counts in recovered.items()
        )
        return (
            "本次已核对回传并更新记录："
            + summary
            + "。\n"
            + _format_workflow_reply(
                {**result, "verified_recovery": None}, state_path, verbose=verbose
            )
        )
    if result.get("status") == "awaiting_approval" and not result.get("_review_card_rendered"):
        pending = (result.get("state") or {}).get("pending_execution_policies") or {}
        proposal = result.get("agent_proposal") or {}
        match = next(
            ((key, row) for key, row in pending.items() if row.get("agent_proposal") == proposal),
            None,
        )
        if match:
            from phase_agent.runtime.review_presentation import (
                review_card,
                revision_lines,
            )

            key, row = match
            card = review_card(key, row)
            heading = (
                f"方案：{card['title']} · 修订 {card['revision']} · {card['proposal_hash'][:12]}"
            )
            original = _format_workflow_reply(
                {**result, "_review_card_rendered": True}, state_path, verbose=verbose
            )
            return "\n".join(
                [
                    heading,
                    *revision_lines(row),
                    original,
                    "审批范围：" + card["approval_boundary"],
                ]
            )
    if result.get("status") in DIALOGUE_STATUSES:
        return str(result.get("answer") or "未取得有效答复。")
    if result.get("status") == "execution_reconciliation_required":
        report = result.get("recovery_report") or {}
        checks = report.get("checks") or []
        if checks:
            lines = ["当前：中断执行需要对账，本轮未执行新动作。"]
            for check in checks[:4]:
                identity = check.get("identity") or {}
                lines.extend(
                    [
                        f"动作：{identity.get('tool') or '未知'}；编号：{identity.get('invocation_id') or '未记录'}",
                        f"已登记完成任务 {len(check['completed_task_ids'])} 个、待执行/回传 {len(check['pending_task_ids'])} 个；已核对文件 {len(check['artifact_checks'])} 个。",
                        "恢复建议：" + check["next_action"],
                    ]
                )
                for artifact in check["artifact_checks"][:4]:
                    lines.append(f"- {artifact['path']}：{artifact['status']}")
            lines.append("上述为只读核对；原动作不会自动重跑。没有记录或文件缺失不等于没有副作用。")
            import os

            lines.append(
                "恢复入口：http://127.0.0.1:"
                + os.environ.get("PHASE_CONTROL_PORT", "8765")
                + "/phase/approval；先生成并确认恢复处理方案，结束旧记录；原输入方案会重新待确认。重复回复‘同意’不会解除这条阻塞。"
            )
            return "\n".join(lines)
        rows = report.get("unsettled") or []
        names = {
            "generate_branches": "初始结构生成",
            "prepare_local_batch_files": "输入文件准备",
        }
        tools = list(
            dict.fromkeys(
                names.get((row.get("identity") or {}).get("tool"), "旧动作") for row in rows
            )
        )
        actions = "、".join(tools) or "旧动作"
        if any(row.get("reason") == "failed_action_still_reserved" for row in rows):
            return (
                f"当前：{actions}的失败记录与任务占位不一致，本轮未执行。\n"
                "建议：核对输出和台账；确认无产出后释放占位，再提出新方案。"
            )
        return (
            f"当前：{actions}有未核对的中断记录，本轮未执行。\n"
            "建议：先核对已有文件、任务与台账，再决定复用结果或重新提出方案；反复发送‘继续’不会解除此阻塞。"
        )
    if result.get("status") in {"training_results_received", "training_handoff"}:
        from phase_agent.runtime.training_reply import concise_training_reply

        handoffs = result.get("training_handoffs") or []
        concise = [concise_training_reply(row, result.get("state") or {}) for row in handoffs]
        if handoffs and all(concise):
            return "\n".join(concise)
        return result["reason"]
    if verbose is None:
        from phase_agent.runtime.response_preferences import detailed_response

        verbose = detailed_response()
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (row.get("agent_proposal") for row in reversed(events) if row.get("agent_proposal")),
        {},
    )
    raw_action = proposal.get("raw_action") or {}
    params = raw_action.get("parameters") or {}
    preview = params.get("refresh_preview")
    refresh = (result.get("state") or {}).get("model_refresh") or {}
    if preview and result.get("status") == "awaiting_approval":
        if "supplemental_relax_parameters" in params:
            settings = params["supplemental_relax_parameters"]
            return (
                "当前：首批刷新已回收，补充弛豫方案待审批。\n"
                f"本次：拟为 {len(raw_action.get('target_ids') or [])} 个单点候选准备补充弛豫输入。\n"
                f"设置：最多 {settings.get('relax_steps', 300)} 步，{settings.get('optimizer', 'LBFGS')}；"
                "未收敛结果排除，不自动延长或重试。\n"
                f"预计相对成本 {raw_action.get('budget', 0):.1f}；耗时暂无可靠估计。\n"
                "下一步：查看具体目标后批准、修改或拒绝；本次未创建任务或输入。"
                "批准仅准备本批输入，不提交计算；首批结果与历史方案保留。"
            )
        rows = preview["candidates"]
        relax = sum(r["operation"] == "relax" for r in rows)
        seconds = (preview.get("runtime_budget") or {}).get("serial_seconds")
        timing = (
            f"串行耗时粗估 {seconds / 3600:.1f} 小时（不含排队）"
            if seconds is not None
            else "耗时暂无同类实测"
        )
        return (
            f"当前：{preview['new_model_version']} 已激活，结构刷新待批准。\n"
            f"首批：弛豫 {relax}、单点 {len(rows) - relax}；暂缓 {preview['deferred']}。\n"
            "标准：<1 meV/atom弛豫，1–10单点，>10按Na×相抽样10%。\n"
            f"预计相对成本 {preview['initial_cost']:.1f}；最多追加一轮弛豫 "
            f"{preview['maximum_supplemental_count']} 个、成本上限 {preview['maximum_supplemental_cost']:.1f}。\n"
            "补充标准：新Ehull<1，或1–10且最大原子力>0.1 eV/Å；已弛豫项不再追加。\n"
            f"未覆盖的分层 {len(preview['coverage_gaps'])} 个；{timing}，单点成本暂按弛豫上界。\n"
            "同意仅批准首批输入生成；回收后若需补充，按具体目标另行审批。\n"
            "仅生成超算输入；你提交GPU.sh并回传results。不本机计算。"
        )
    if refresh and refresh.get("status") != "completed":
        if refresh.get("status") == "waiting_results":
            from phase_agent.analysis.state.workflow_blockers import (
                refresh_result_facts,
            )

            facts = refresh_result_facts(result.get("state") or {})
            failed = sum(facts["status_counts"].get(s, 0) for s in ("failed", "timeout"))
            if failed:
                reason = result.get("reason")
                if not reason:
                    reason = next(
                        (e.get("reason") for e in reversed(events) if e.get("reason")),
                        None,
                    )
                causes = "；".join(
                    f"{row['count']} 个：{row['reason']}" for row in facts["failure_groups"][:3]
                )
                return reason or (
                    f"当前刷新 {failed}/{facts['task_count']} 个失败。\n原因：{causes}\n先修复依赖，再审核重试方案；重复“同意”不能恢复缺失文件。"
                )
            from phase_agent.runtime.chat_state_presentation import brief_chat_state
            from phase_agent.runtime.waiting_files_presentation import (
                waiting_file_lines,
            )

            state = result.get("state") or {}
            if (
                facts["task_count"]
                and facts["status_counts"].get("completed", 0) == facts["task_count"]
            ):
                reason = result.get("reason") or next(
                    (e.get("reason") for e in reversed(events) if e.get("reason")), None
                )
                return "\n".join([brief_chat_state(state), *([str(reason)] if reason else [])])
            return "\n".join([brief_chat_state(state), *waiting_file_lines(state)])
        reason = result.get("reason") or next(
            (e.get("reason") for e in reversed(events) if e.get("reason")), None
        )
        if reason:
            return "当前：新模型结构刷新。\n" + str(reason)
    text = _workflow_reply_body(result, state_path, verbose=verbose)
    if result.get("followup_error") or result.get("followup_note"):
        text += "\n下一步方案暂未准备成功；已生成结果保留，未执行后续任务。"
        if verbose:
            text += "\n" + str(result.get("followup_error") or result.get("followup_note"))
    if not result.get("post_dft_review") and result.get("status") in {
        "confirmation_required",
        "awaiting_remote_training",
        "awaiting_manual_submission",
        "failed",
        "rejected",
        "rejected_by_user",
        "not_configured",
        "prepared",
        "already_prepared",
        "tasks_in_progress",
    }:
        # Follow-up replies report the transition, not the previous round again.
        if verbose or result.get("status") == "failed":
            return text
        return text + f"\n完整记录：`{state_path}`"
    return text


def _with_post_dft_reply(result, text, *, verbose):
    """Reuse round-scoped analysis at the common presentation boundary."""
    from phase_agent.runtime.post_dft_presentation import (
        post_dft_lines,
        post_dft_review_lines,
        reply_assessment,
    )

    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (row.get("agent_proposal") for row in reversed(events) if row.get("agent_proposal")),
        {},
    )
    state = result.get("state") or {}
    review_action = (proposal or {}).get("raw_action") or {
        "post_dft_review": result.get("post_dft_review")
    }
    round_assessment = reply_assessment(result, review_action)
    if not round_assessment and not review_action.get("post_dft_review"):
        return text
    assessment, summary = "", ""
    if round_assessment:
        assessment = post_dft_lines(state, assessment=round_assessment)
        from phase_agent.analysis.feedback.dft_magnetic_tables import magnetic_chat_summary

        if verbose:
            summary = magnetic_chat_summary(
                state, task_ids=set(round_assessment["recovered_task_ids"]), verbose=True
            )
    review = post_dft_review_lines(
        review_action,
        verbose=verbose,
        finetune_enabled=round_assessment.get("finetune_enabled") if round_assessment else None,
    )
    if review and result.get("status") == "awaiting_approval":
        text = "下一步建议（待批准）：\n" + text
    if not verbose and assessment:
        return _compact_post_dft_reply(assessment, summary, review, text)
    return "\n\n".join(part for part in (assessment, summary, review, text) if part)


def _compact_post_dft_reply(assessment, magnetic, review, action_text):
    """Show the round metrics and next action; full reports remain on disk."""
    visible = assessment.splitlines()[:4]
    return "\n\n".join(part for part in ("\n".join(visible), action_text) if part)


def _workflow_reply_body(result: dict, state_path, *, verbose=False) -> str:
    """Keep routine chat concise; full evidence stays in existing records."""
    if isinstance(result, dict) and result.get("status") == "confirmation_required":
        events = result.get("events") or []
        reason = (
            result.get("message")
            or result.get("reason")
            or next(
                (
                    event.get("reason")
                    or ((event.get("execution") or {}).get("result") or {}).get("reason")
                    for event in reversed(events)
                    if event.get("reason")
                    or ((event.get("execution") or {}).get("result") or {}).get("reason")
                ),
                None,
            )
        )
        return "操作仍需确认，当前未报告执行完成。" + (
            f"\n原因：{reason}"
            if reason
            else "\n流程未提供具体原因，请查看完整记录中的审批与执行结果。"
        )
    if isinstance(result, dict) and result.get("status") == "awaiting_dft_recovery_decision":
        return _dft_recovery_question_reply(result)
    if verbose or not isinstance(result, dict):
        return _format_workflow_reply_verbose(result, state_path, verbose=True)
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (row.get("agent_proposal") for row in reversed(events) if row.get("agent_proposal")),
        {},
    )
    status = result.get("status")
    if status == "awaiting_approval" and proposal:
        if _is_model_failure_proposal(proposal):
            return "旧建议由模型回复失败产生，不是暂停决策。请重启 Agent 服务后说“继续”刷新；未执行任务。"
        # Preserve all approval limits, budget interceptions and sensitive warnings.
        text = _format_workflow_reply_verbose(result, state_path)
        from phase_agent.runtime.evidence_reference_presentation import (
            evidence_reference_lines,
        )

        evidence_lines = evidence_reference_lines(proposal.get("raw_action") or {})
        labels = {
            "generate_branches": "生成 branch",
            "allocate_mc_bohb": "分配 MC",
            "prepare_local_batch_files": "准备输入文件",
            "select_dft_candidates": "筛选 DFT",
        }
        tool = proposal.get("recommended_action")
        if tool == "select_dft_candidates":
            review = proposal.get("dft_template_review") or (proposal.get("raw_action") or {}).get(
                "dft_template_review"
            )
            if review and review.get("status") == "pending":
                return (
                    "当前：DFT 公共参数示例待确认。\n"
                    f"代表结构：{review['candidate_id']}；模板版本：{review['revision']}。\n"
                    f"检查目录：{review['directory']}\n"
                    "请检查 INCAR、KPOINTS、GPU.sh 和 settings.json。K 点与元素相关参数按各结构生成。\n"
                    "可提出修改；或编辑 settings.json 后回复‘刷新模板’。回复‘同意’确认本版本并生成所选批次；‘拒绝’取消。\n"
                    "目前只生成一组示例，未生成批量 DFT 输入或提交任务。"
                )
            params = proposal.get("action_parameters") or {}
            preview = params.get("dft_input_preview") or {}
            if not preview.get("task_count"):
                return "DFT 旧方案缺少结构数量与成本预览，暂不可批准。请说“继续”刷新方案。"
            if not preview.get("selected_structures"):
                return "当前是旧 DFT 方案，缺少 Na/相/Ehull 清单及新版采点信息。请说“继续”刷新，未执行任务。"
            text = text.replace(
                "建议：`select_dft_candidates`",
                (
                    f"建议：生成 {preview['task_count']} 个 DFT 输入（单点 {preview.get('single_point_count', 0)}、优化 {preview.get('relax_count', preview['task_count'])}）"
                    if "single_point_count" in preview
                    else f"建议：生成 {preview['task_count']} 个 DFT 结构优化输入"
                ),
            )
            selected = preview.get("selected_structures") or []
            if params.get("plan_revision_notice"):
                text += "\n方案修订：" + params["plan_revision_notice"]
            if selected:
                text += "\n\n入选结构（Na/O₂；相；Ehull eV/atom）：\n"

                def number(value):
                    return f"{value:.6g}" if isinstance(value, (int, float)) else "未知"

                text += "\n".join(
                    f"- {row['candidate_id']}：{number(row.get('x_Na_per_O2'))}；"
                    f"{row.get('phase') or '未知'}；{number(row.get('ehull'))}"
                    for row in selected
                )
            text += f"\nQBC：{preview.get('qbc_available_count', 0)}/{preview['task_count']} 个有数据；缺失值不参与评分。"
            for warning in preview.get("warnings") or []:
                text += "\n提醒：" + warning
            text += "\n批准后生成输入文件，不提交超算。"
        text = text.replace(f"建议：`{tool}`", f"建议：{labels.get(tool, tool)}")
        lines = [line for line in text.splitlines() if not line.startswith("完整参数与依据：")]
        lines = [
            line.replace(
                "回复“同意”执行；回复“拒绝”取消；也可以直接提出修改意见。",
                "回复“同意”执行、“拒绝”取消，或提出修改。",
            )
            for line in lines
        ]
        return "\n".join(lines + evidence_lines)
    if status == "awaiting_manual_submission":
        wait = result.get("manual_wait") or {}
        recovered = int(wait.get("recovered_count") or result.get("recovered_count") or 0)
        names = {
            "deep_search": "MC",
            "relax_and_feature": "Relax",
            "dft_single_point": "DFT 单点",
            "dft_relax": "DFT 优化",
        }
        stages = wait.get("waiting_by_stage") or {}
        summary = "、".join(
            f"{names.get(stage, stage)} {count} 个" for stage, count in sorted(stages.items())
        )
        lines = [f"已回收 {recovered} 个结果。"] if recovered else []
        for row in wait.get("dft_recovery_rounds") or []:
            if row["pending_task_ids"]:
                lines.append(
                    f"{row['scope'].get('model_version') or '未知模型'} 的 DFT："
                    f"已回收 {row['recovered_tasks']}/{row['expected_tasks']}（{row['recovery_ratio']:.1%}），"
                    f"成功 {row['successful_tasks']}、失败 {row['failed_tasks']}。"
                )
        lines.append(f"等待结果：{summary or str(wait.get('waiting_task_count', 0)) + ' 个任务'}。")
        collection = wait.get("result_collection") or {}
        issues = int(collection.get("invalid_count", 0)) + int(
            collection.get("missing_structure_count", 0)
        )
        markers = int(collection.get("missing_marker_count", 0))
        if issues or markers:
            lines.append(f"需检查：校验/结构问题 {issues} 个，缺完成标记 {markers} 个。")
        interception = (result.get("state") or {}).get("mc_budget_interception") or {}
        if interception.get("rejected_count"):
            verbose_text = _format_workflow_reply_verbose(result, state_path)
            lines.extend(
                line for line in verbose_text.splitlines() if line.startswith("预算拦截：")
            )
        if wait.get("handoffs"):
            lines.extend(_handoff_lines(wait["handoffs"], collection=wait.get("result_collection")))
            return "\n".join(lines)
        from phase_agent.runtime.waiting_files_presentation import waiting_file_lines

        file_lines = waiting_file_lines(result.get("state") or {})
        if file_lines:
            lines.extend(file_lines)
            errors = collection.get("transfer_errors") or []
            if errors:
                lines.append("结果包导入失败：" + "；".join(row["reason"] for row in errors[:3]))
            return "\n".join(lines)
        if wait.get("upload_root"):
            lines.append(f"目录：`{wait['upload_root']}`")
        else:
            roots = list(
                dict.fromkeys(
                    str(Path(path).parent) for path in wait.get("batch_directories") or []
                )
            )
            if roots:
                lines.append("目录：" + "、".join(f"`{path}`" for path in roots))
        lines.append("尚未本机提交。Relax/MC 提交批次 GPU.sh；DFT 提交单任务 GPU.sh。")
        if wait.get("upload_archives"):
            lines.append(
                "上传输入包：" + "、".join(f"`{path}`" for path in wait["upload_archives"])
            )
        lines.append(
            "完成后只需下载 results/transfer/all.results.zip 到对应 results 目录，再说“继续”自动校验回收；旧批次仍可回传精简 results 文件夹。未回传任务保持等待。"
        )
        errors = collection.get("transfer_errors") or []
        if errors:
            lines.append("结果包导入失败：" + "；".join(row["reason"] for row in errors[:3]))
        return "\n".join(lines)
    text = _format_workflow_reply_verbose(result, state_path)
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith(
            (
                "详细记录：",
                "完整参数与结果：",
                "完整记录：",
                "示例目录：",
                "结构路径与详细结果：",
            )
        )
    )


def _dft_recovery_question_reply(result):
    question = result.get("dft_recovery_question") or {}
    state = result.get("state") or {}
    scope = question.get("scope") or {}
    version = scope.get("model_version") or "未知模型"
    model_round = ((state.get("upload_layout") or {}).get("model_rounds") or {}).get(version)
    label = f"第 {model_round} 轮（{version}）" if model_round is not None else f"模型 {version}"
    lines = [
        f"{label}：DFT 已回收 {question.get('recovered_tasks', 0)}/{question.get('expected_tasks', 0)}"
        f"（{question.get('recovery_ratio', 0):.1%}）。",
        f"成功 {question.get('successful_tasks', 0)}、失败 {question.get('failed_tasks', 0)}；"
        f"未回传 {len(question.get('pending_task_ids') or [])} 个。",
    ]
    if state.get("round_summary_csv_path"):
        lines.append(f"轮次记录：`{state['round_summary_csv_path']}`")
    lines.extend(
        [
            "还要继续回收剩余结果吗？",
            "回复“继续回收”则等待；回复“不再回收”或“否”，则取消剩余DFT本地跟踪并基于已有结果提出下一步方案。",
            "取消后不再回收这些任务，不会把缺失结果标为完成，也不会向超算发送终止作业命令。",
        ]
    )
    return "\n".join(lines)


def _format_workflow_reply_verbose(result: dict, state_path, *, verbose=False) -> str:
    if not isinstance(result, dict):
        return str(result)
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (event.get("agent_proposal") for event in reversed(events) if event.get("agent_proposal")),
        None,
    )
    if result.get("status") == "awaiting_approval" and proposal:
        proposed_tool = proposal.get("recommended_action") or (
            proposal.get("raw_action") or {}
        ).get("tool")
        if not isinstance(proposed_tool, str) or not proposed_tool:
            return (
                "Agent 返回了无效动作，本轮没有可批准的建议。"
                "请拒绝旧建议后重新提出；不会执行空动作。"
            )
        cost = proposal.get("estimated_cost") or {}
        from phase_agent.tools.policy.training_input_action import (
            is_training_input_action,
        )

        training_inputs = is_training_input_action(
            proposal.get("raw_action")
            or {
                "tool": proposed_tool,
                "parameters": proposal.get("action_parameters") or {},
            }
        )
        approval_files = next(
            (
                event.get("approval_files")
                for event in reversed(events)
                if event.get("approval_files")
            ),
            {},
        )
        detail = approval_files.get("directory") or _proposal_directory(state_path, proposal)
        from phase_agent.runtime.proposal_presentation import TOOL_LABELS

        lines = [
            f"建议：{TOOL_LABELS.get(proposed_tool, proposed_tool)}",
            f"目的：{proposal.get('expected_purpose') or proposal.get('reason') or '未提供'}",
            f"预计相对成本：{cost.get('estimated_total_cost') if cost.get('estimated_total_cost') is not None else '待任务展开后估算'}",
        ]
        if (
            proposed_tool == "prepare_local_batch_files"
            and (
                (proposal.get("raw_action") or {}).get("parameters")
                or proposal.get("action_parameters")
                or {}
            ).get("mode")
            == "relax_inputs"
        ):
            lines = [
                "建议：准备首轮 MLIP 弛豫输入和提交脚本",
                "批准后生成弛豫输入、提交脚本和上传清单，由你上传超算并提交。",
                "回收弛豫结果后，分析并构建 MLIP 相图，再提出 MC 步数分配与文件准备方案；本次不分配 MC 步数。",
            ]
        if training_inputs:
            lines = [
                "建议：微调模型，生成超算训练提交文件",
                "目的：按本轮分析准备committee、K折评估和全数据主模型训练输入。",
                "训练耗时：尚无训练实测，待估算；输入生成成本不代表超算训练成本。",
                "批准后仅生成输入与GPU.sh，由你上传提交；不本机训练、不自动激活模型。",
            ]
        recovered_count = int(result.get("recovered_count") or 0)
        runtime = cost.get("runtime_budget") or {}
        if runtime.get("serial_seconds") is not None:
            hours = runtime["serial_seconds"] / 3600
            sample_range = runtime.get("sample_range_seconds")
            observed = ""
            if sample_range and len(sample_range) == 2:
                low, high = sample_range
                observed = f"（观测范围 {low / 3600:.2g}–{high / 3600:.2g} 小时）"
            lines.append(
                f"耗时粗估：串行累计约 {hours:.2g} 小时{observed}；不含排队，并行后实际完成时间另计。"
            )
        elif proposed_tool in {
            "allocate_mc_bohb",
            "prepare_local_batch_files",
            "select_dft_candidates",
            "run_calculation_stage",
        }:
            lines.append("耗时粗估：暂无足够实测或任务规模信息，暂无法估算；相对成本不代表小时。")
        if recovered_count:
            lines.insert(0, f"已校验并回收 {recovered_count} 个任务结果，结果与成本已入账。")
        if proposed_tool == "generate_branches" and not verbose:
            from phase_agent.runtime.proposal_presentation import (
                concise_generation_lines,
            )

            params = (
                proposal.get("action_parameters")
                or (proposal.get("raw_action") or {}).get("parameters")
                or {}
            )
            lines = concise_generation_lines(params, cost)
            if recovered_count:
                lines.insert(0, f"已校验并回收 {recovered_count} 个任务结果。")
        if proposed_tool == "generate_branches" and verbose:
            params = (
                proposal.get("action_parameters")
                or (proposal.get("raw_action") or {}).get("parameters")
                or {}
            )
            if params.get("sampling_mode") == "region_adaptive":
                from phase_agent.runtime.proposal_presentation import region_scope_lines

                lines.extend(region_scope_lines(params))
            else:
                quotas = params.get("quotas")
                names = {
                    "coverage": "补覆盖",
                    "composition": "补Na组分",
                    "competing_phase": "竞争相",
                    "periodic_extension": "扩展超胞",
                    "tm_ordering": "TM排布",
                }
                strategy = (
                    "、".join(f"{names.get(name, name)} {count}" for name, count in quotas.items())
                    if isinstance(quotas, dict) and quotas
                    else f"覆盖补充（预计 {params.get('total_quota', '按配置')} 个；具体分配由现有 branch 覆盖决定）"
                )
                if verbose or not params.get("generation_plan"):
                    lines.append(f"生成策略：{strategy}")
                plan = params.get("generation_plan") or []
                for row in plan:
                    target = "全部合法相" if row["phase"] == "all" else row["phase"]
                    x = (
                        f"，Na/O₂ {row['na_min']:g}–{row['na_max']:g}"
                        if row.get("na_min") is not None
                        else ""
                    )
                    cap = f"，det(H)≤{row['max_det_H']}" if row.get("max_det_H") is not None else ""
                    lines.append(
                        f"- {target}{x}{cap}：{names.get(row['strategy'], row['strategy'])}候选 {row['quota']}；{row['reason']}"
                    )
                if params.get("max_det_H") is not None:
                    lines.append(f"本轮超胞上限：det(H) ≤ {params['max_det_H']}")
                else:
                    lines.append("本轮超胞上限：未单独设置，按已确认的 H 边界")
                initial_count = params.get("initial_states_per_branch")
                lines.append(
                    f"入选上限：{params.get('batch_size', '按配置')} 个 branch；"
                    f"每个 branch 静电能前 10 中至多取 {initial_count if initial_count is not None else '按配置'} 个初态"
                )
                preview = params.get("generation_cost_preview") or {}
                if preview:
                    lines.append(
                        f"后续Relax+MC成本粗估上界：{preview['relax_mc_cost_upper']:.3g}（非本次派发授权）。"
                    )
                    seconds = preview.get("serial_seconds_upper")
                    lines.append(
                        f"Relax+MC串行耗时粗估：{seconds / 3600:.2g} 小时，不含排队。"
                        if seconds is not None
                        else "耗时：实测样本不足，暂不换算小时。"
                    )
                    lines.append("本次仅生成branch；后续Relax/MC单独审批。")
        batch_params = (
            proposal.get("action_parameters")
            or (proposal.get("raw_action") or {}).get("parameters")
            or {}
        )
        batch_preview = batch_params.get("relax_mc_preview") or {}
        if batch_preview:
            workload = batch_preview["workload"]
            lines.append(
                f"封装估算：Relax {workload['relax_structure_tasks']} 个结构；三段 MC 任务上限 {workload['mc_task_upper']}；GPU 计算作业最多 {sum(workload['gpu_jobs_upper_by_wave'])} 个；"
                + (
                    f"另最多 {workload.get('cpu_control_jobs_upper', 5)} 个短 CPU 入口/汇总作业；阶段内并行，全部结束并核验汇总后推进，无常驻总控。"
                    if workload.get("submission_policy") == "phase_dependencies"
                    else "另 1 个控制作业；同轮全部提交，实际并行由超算调度；回收并更新凸包后才进入下一轮。"
                    if workload.get("submission_policy") == "whole_wave"
                    else f"另 1 个控制作业；计算并发 {workload['max_concurrent_jobs']}。"
                )
            )
            lines.append("人工上传并提交一次；固定当前模型，追加由已确认规则按反馈选择。")
            if batch_preview.get("bootstrap_first_mc"):
                lines.append(
                    "Relax 回收后先构建当前模型凸包并计算 Ehull，再按规则派发首轮 MC；每段回收后更新凸包，再决定追加。缺少有效凸包则暂停。"
                )
            lines.extend(
                "诊断提示：" + warning for warning in batch_preview["diagnosis"].get("warnings", [])
            )
        if proposal.get("recommended_action") == "allocate_mc_bohb":
            params = (
                proposal.get("action_parameters")
                or (proposal.get("raw_action") or {}).get("parameters")
                or {}
            )
            from phase_agent.runtime.proposal_presentation import mc_scope_lines

            lines.extend(mc_scope_lines(params))
        completed = result.get("completed_generation") or {}
        if completed:
            lines.insert(
                0,
                f"结果：已生成 {completed.get('registered_branches', 0)} 个 branch、{completed.get('registered_structures', 0)} 个初始结构。",
            )
        summary = proposal.get("decision_summary") or {}
        if summary.get("observation"):
            lines.insert(0, "观察：" + summary["observation"])
        if summary.get("rationale") and (summary.get("observation") or summary.get("alternatives")):
            lines.insert(1 if summary.get("observation") else 0, "判断：" + summary["rationale"])
        if verbose:
            for alternative in summary.get("alternatives") or []:
                lines.append(
                    f"其他路线：{TOOL_LABELS.get(alternative['tool'], alternative['tool'])}；{alternative['reason']}"
                )
            if summary.get("expected_outcome"):
                lines.append("预期结果：" + summary["expected_outcome"])
            if summary.get("success_criterion"):
                lines.append("验证方式：" + summary["success_criterion"])
        fallback_reason = (proposal.get("raw_action") or {}).get("fallback_reason")
        if fallback_reason:
            lines.append(f"回退原因：`{fallback_reason}`")
        if detail:
            lines.append(f"完整参数与依据：`{detail}`")
        lines.extend(["", "回复“同意”执行；回复“拒绝”取消；也可以直接提出修改意见。"])
        if _is_sensitive_proposal(proposal):
            lines.extend(
                [
                    "该建议涉及敏感操作，仍需在本机审批页确认具体影响：",
                    f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval",
                ]
            )
        return "\n".join(lines)
    status = result.get("status") or "unknown"
    if status == "config_migrated":
        migration = result.get("migration") or {}
        source = result.get("from_config_version") or migration.get("from") or "旧版本"
        target = result.get("config_version") or migration.get("to") or "新版本"
        changes = migration.get("budget_changes") or []
        change_text = ""
        if changes:
            details = "、".join(
                f"{row.get('field')} {row.get('from')}→{row.get('to')}" for row in changes
            )
            change_text = f"预算调整：{details}。"
        policy_changes = migration.get("policy_changes") or []
        policy_text = ""
        if policy_changes:
            details = "、".join(
                f"{row.get('field')} "
                f"{'隐含默认值 ' if row.get('from_basis') == 'implicit_default' else ''}"
                f"{row.get('from')}→{row.get('to')}"
                for row in policy_changes
            )
            policy_text = f"已批准的 MC 策略/成本估算调整：{details}。"
        compatibility = migration.get("compatibility") or {}
        compatibility_text = ""
        if compatibility:
            count = compatibility.get("relax_task_count", 0)
            unknown = compatibility.get("unrecorded_fields") or []
            compatibility_text = (
                f"已核对 {count} 条旧 Relax 结果与 mh-1 的既有默认设置一致；"
                f"未记录项：{'、'.join(unknown) if unknown else '无'}。"
            )
        return (
            f"运行配置已从 {source} 迁移到 {target}；历史任务和结果仍保留原版本归属。"
            f"{change_text}{policy_text}{compatibility_text}"
            "本次只更新配置绑定，没有生成、提交或运行计算。请发送下一步任务指令。"
        )
    if status == "config_migration_not_needed":
        return "当前运行已绑定最新确认配置，无需迁移；本次没有执行任何计算。"
    if status in {"configuration_revision_required", "configuration_version_mismatch"}:
        reason = result.get("reason") or next(
            (
                event.get("reason")
                or ((event.get("execution") or {}).get("result") or {}).get("reason")
                for event in reversed(events)
                if event.get("reason")
                or ((event.get("execution") or {}).get("result") or {}).get("reason")
            ),
            None,
        )
        return f"{reason or '本轮要求超过已确认配置；请先修订配置。'}\n未执行生成或计算。"
    if status == "awaiting_remote_training":
        from phase_agent.runtime.chat_state_presentation import waiting_training_job
        from phase_agent.runtime.training_reply import training_delivery_lines

        training = waiting_training_job(result.get("state") or {})
        if training:
            delivery = training_delivery_lines(training)
            if delivery:
                return "\n".join(delivery)
        payload = (result.get("execution") or {}).get("result") or next(
            (
                ((event.get("execution") or {}).get("result") or {})
                for event in reversed(events)
                if ((event.get("execution") or {}).get("result") or {}).get("reason")
            ),
            {},
        )
        return (
            payload.get("reason")
            or result.get("reason")
            or "微调输入已准备，等待超算训练；未在本机训练或激活模型。"
        )
    action = (
        (result.get("final_action") or {}).get("tool")
        or (proposal or {}).get("recommended_action")
        or next(
            (
                (event.get("final_action") or {}).get("tool")
                for event in reversed(events)
                if (event.get("final_action") or {}).get("tool")
            ),
            None,
        )
    )
    validation = result.get("validation") or next(
        (event.get("validation") for event in reversed(events) if event.get("validation")),
        {},
    )
    errors = validation.get("errors") or []
    detail_path = str(Path(state_path))
    if status == "awaiting_manual_submission":
        wait = result.get("manual_wait") or {}
        recovered_count = int(wait.get("recovered_count") or result.get("recovered_count") or 0)
        waiting_count = int(wait.get("waiting_task_count") or 0)
        by_stage = wait.get("waiting_by_stage") or {}
        stage_label = (
            "MC"
            if set(by_stage) == {"deep_search"}
            else "Relax"
            if set(by_stage) == {"relax_and_feature"}
            else "DFT"
            if set(by_stage) <= {"dft_single_point", "dft_relax"}
            else "计算"
        )
        lines = []
        if recovered_count:
            lines.append(f"已校验并回收 {recovered_count} 个任务结果；预算和台账已更新。")
        collection = wait.get("result_collection") or {}
        if collection:
            lines.append(
                "已直接检查 upload_batches 原任务目录（未复制旧文件）："
                f"扫描 {collection.get('considered_count', 0)} 个，"
                f"回收 {collection.get('recovered_count', 0)} 个；"
                f"缺结果 {collection.get('missing_result_count', 0)} 个，"
                f"缺完成标记 {collection.get('missing_marker_count', 0)} 个，"
                f"校验/结构问题 {int(collection.get('invalid_count', 0)) + int(collection.get('missing_structure_count', 0))} 个。"
            )
        if recovered_count:
            lines.append(f"当前有 {waiting_count} 个 {stage_label} 任务待提交或回传。")
        else:
            lines.append(f"{stage_label} 输入已准备：{waiting_count} 个任务待提交或回传。")
        interception = (result.get("state") or {}).get("mc_budget_interception") or {}
        if stage_label == "MC" and interception.get("rejected_count"):
            breakdown = "、".join(
                f"{key} 入选{row['accepted']}/拦截{row['rejected']}"
                for key, row in sorted((interception.get("strata") or {}).items())
            )
            lines.append(
                f"预算拦截：入选 {interception['accepted_count']} 个，"
                f"拦截 {interception['rejected_count']} 个；原因："
                f"{', '.join(interception.get('reasons') or [])}。"
                f"按层级/相分布：{breakdown}。"
                "如果要运行全部，请先查看完整估计成本并明确批准全量方案。"
            )
        lines.append(
            "本机没有提交作业。请将同一阶段的批次目录放在同一轮次目录下：Relax/MC 每个批次根目录的 `GPU.sh` 提交一次；DFT 则在单任务目录提交 `GPU.sh`。"
        )
        if wait.get("handoffs"):
            lines.extend(_handoff_lines(wait["handoffs"], collection=wait.get("result_collection")))
            return "\n".join(lines)
        if wait.get("upload_root"):
            lines.append(f"本地任务总目录：`{wait['upload_root']}`")
        if wait.get("upload_plan_path"):
            lines.append(
                f"按 branch 分组的完整上传清单：`{wait['upload_plan_path']}`。每个 branch 的所有任务目录都列在一起。"
            )
        if wait.get("upload_archives"):
            lines.append(
                "必要输入包：" + "、".join(f"`{path}`" for path in wait["upload_archives"])
            )
        directories = wait.get("batch_directories") or wait.get("task_directories") or []
        if directories:
            lines.append("待提交的批次目录：")
            lines.extend(f"- `{path}`" for path in directories[:5])
            if len(directories) > 5:
                lines.append(f"- 其余 {len(directories) - 5} 个批次见任务清单和 state 文件。")
        results_directories = wait.get("results_directories") or []
        if results_directories:
            lines.append(
                "本阶段统一结果目录：" + "、".join(f"`{path}`" for path in results_directories)
            )
        lines.append(
            "本阶段任务完成后，只需下载 `results/transfer/all.results.zip`，放到登记的本地 `results/` 目录，"
            "其中含所有任务的 `result.json`、`task.finished.json` 和最终结构；再对 Agent 说“继续”。"
            "Agent 会按任务校验并回收；未回传任务继续等待，不会重复准备或提交。"
            "旧批次仍按原任务目录回收。"
        )
        return "\n".join(lines)
    if status == "failed":
        execution = result.get("execution") or next(
            (event.get("execution") for event in reversed(events) if event.get("execution")),
            {},
        )
        reason = (
            (execution or {}).get("error")
            or ((execution or {}).get("result") or {}).get("reason")
            or result.get("reason")
            or "未知错误"
        )
        return f"本轮失败：{str(reason)[:500]}\n详细记录：`{detail_path}`"
    if status == "rejected_by_user":
        return f"已取消本轮建议；未执行任何动作。\n详细记录：`{detail_path}`"
    if status == "rejected":
        reason = _friendly_validation_errors(errors)
        workflow_reason = (
            result.get("message")
            or result.get("reason")
            or next(
                (event.get("reason") for event in reversed(events) if event.get("reason")),
                None,
            )
        )
        if not errors and workflow_reason:
            reason = _friendly_workflow_rejection(workflow_reason)
        if not errors and not workflow_reason:
            execution = result.get("execution") or next(
                (event.get("execution") for event in reversed(events) if event.get("execution")),
                {},
            )
            payload = (execution or {}).get("result") or {}
            reason = _friendly_workflow_rejection(
                payload.get("reason") or execution.get("error") or "执行动作被拒绝"
            )
        return f"本轮未执行：{reason}\n详细记录：`{detail_path}`"
    if status == "not_configured":
        execution = result.get("execution") or next(
            (event.get("execution") for event in reversed(events) if event.get("execution")),
            {},
        )
        payload = (execution or {}).get("result") or {}
        reason = (
            (execution or {}).get("error")
            or payload.get("reason")
            or result.get("reason")
            or next(
                (event.get("reason") for event in reversed(events) if event.get("reason")),
                None,
            )
            or "执行接口未配置"
        )
        if action == "allocate_mc_bohb" and payload.get("actions"):
            return (
                f"已分配 {len(payload['actions'])} 个 MC 任务的预算，但输入文件尚未准备好：{reason}。"
                f"任务已保留；修正原因后可继续准备 MC 输入。\n详细记录：`{detail_path}`"
            )
        reason_labels = {
            "remote_mlip_model_missing": "运行配置中未找到已确认的远端 MACE 模型路径或版本",
            "upload_directory_or_worker_command_missing": "上传目录或远端 worker 执行命令未配置",
            "structure_dedup_not_ready": "结构去重检查尚未完成",
            "no_legal_existing_relax_structures": "没有可用于 Relax 的合法现有结构",
            "current_mlip_phase_diagram_unavailable": "当前 MLIP 相图未就绪，不能按 Ehull/atom 派发 MC",
            "approved_phase_diagram_changed": "相图在审批后变化，需重新预览并批准 MC 方案",
            "relax_structure_missing_from_current_phase_diagram": "所选 Relax 结构在当前相图中缺少唯一匹配的 Ehull/atom",
            "mc_requires_current_phase_diagram_ehull_per_atom": "MC 派发缺少当前相图的 Ehull/atom 证据",
        }
        reason = reason_labels.get(str(reason), str(reason))
        if "llm_failed:" in str(reason) and any(
            word in str(reason).lower() for word in ("timeout", "timed out", "超时")
        ):
            return "本轮未执行：模型服务请求超时；已有结果仍保留，本次未生成新方案或执行任务。\n下一步：网络或服务恢复后，再重新请求下一步方案。"
        return f"本轮未执行：`{action or 'workflow'}` 未能准备（{reason}）。\n详细记录：`{detail_path}`"
    if status in {"prepared", "already_prepared"} and action == "prepare_local_batch_files":
        execution = result.get("execution") or next(
            (event.get("execution") for event in reversed(events) if event.get("execution")),
            {},
        )
        payload = (execution or {}).get("result") or {}
        if payload.get("batches"):
            locations = "、".join(row.get("directory", "") for row in payload["batches"][:3])
            return (
                f"已准备 {payload.get('task_count', 0)} 个 Relax 任务、"
                f"{payload.get('batch_count', 0)} 个上传批次；未提交超算。\n"
                f"按 branch 上传清单：`{payload.get('upload_plan_path') or detail_path}`。\n"
                f"示例目录：`{locations}`"
            )
        return f"Relax 输入已准备过，没有重复创建或扣费。\n详细记录：`{detail_path}`"
    if status in {"completed", "planned_only", "tasks_in_progress"}:
        if status == "completed" and action == "generate_branches":
            execution = result.get("execution") or next(
                (event.get("execution") for event in reversed(events) if event.get("execution")),
                {},
            )
            summary = ((execution or {}).get("result") or {}).get("summary") or {}
            if summary:
                phases = (
                    "、".join(
                        f"{name} {count}"
                        for name, count in sorted((summary.get("phase_counts") or {}).items())
                    )
                    or "无"
                )
                det_range = summary.get("det_H_range") or []
                cells = (
                    f"入选 det(H) {det_range[0]}–{det_range[1]}"
                    if len(det_range) == 2
                    else "无入选超胞"
                )
                cap = summary.get("max_det_H")
                if cap is not None:
                    cells += f"（本轮设定上限 {cap}）"
                missing = (summary.get("phase_coverage") or {}).get("missing") or []
                shortfall = summary.get("unfilled_quota", 0)
                notice = (
                    f"缺少目标相：{', '.join(missing)}；补齐前不能准备整批计算输入。\n"
                    if missing
                    else ""
                )
                if shortfall:
                    notice += f"策略配额有 {shortfall} 个未生成，原因见分配记录；未自动转移配额。\n"
                return (
                    notice + f"已生成 {summary.get('registered_branches', 0)} 个 branch、"
                    f"{summary.get('registered_structures', 0)} 个初始结构。"
                    f"去重前入选相分布：{phases}；{cells}。\n"
                    f"生成 {summary.get('proposed_branches', 0)} 个候选，"
                    f"入选 {summary.get('selected_branches', 0)} 个 branch，"
                    f"静电能初始化失败 {summary.get('initialization_failed_branches', 0)} 个，"
                    f"构型去重后 {summary.get('unique_structures', 0)} 个。\n"
                    "建议：是否现在审查已生成结构并准备 Relax 任务方案？确认后先展示结构范围、预算与提交方式；你批准具体方案后，才生成输入文件、提交脚本和上传清单。\n"
                    "文件生成后，按清单上传超算并提交；本次尚未运行 Relax。\n"
                    f"结构路径与详细结果：`{detail_path}`"
                )
        labels = {
            "completed": "已完成",
            "planned_only": "已生成计划但未执行",
            "tasks_in_progress": "任务已进入处理中",
        }
        return f"{labels[status]}：`{action or 'workflow'}`。\n完整参数与结果：`{detail_path}`"
    return f"工作流状态：`{status}`。\n完整记录：`{detail_path}`"


def _friendly_validation_errors(errors):
    mapping = {
        "formal_tool_requires_task_key": "动作缺少内部幂等编号",
        "configuration_not_confirmed": "配置尚未确认",
        "new_model_branch_generation_required": "新模型刷新后须先生成新branch，展示实际数量和分布，再单独审批Relax/MC方案",
        "post_refresh_branch_generation_separate_review_required": "首次新branch生成须独立审批并展示实际结果，不能捆绑后续输入或暂缓生成",
        "config_version_mismatch": "当前运行仍绑定旧配置版本；请新建运行，或在尚无科学任务时重新发送该指令",
        "invalid_budget": "Agent 返回的预算格式无效；预算必须是一个非负数值",
        "tool_handler_not_configured": "该动作尚无执行函数，请改选已接通的工具",
        "mc_requires_recovered_relax_and_current_hull": "尚未回收当前模型的弛豫结果并构建有效相图；先准备独立的 Relax 输入方案，提交并回收后再分配 MC",
        "duplicate_effective_decision": "相同任务已有占位或执行记录；需核对原任务与执行回执，本次未重复执行",
        "reserved_total_cost_limit": "预算不足",
        "retry_target_not_failed_task": "重试目标不是失败的计算任务；历史动作记录不能作为 task_id",
        "calculation_stage_requires_one_structure_target": "单项计算动作必须指向一个结构；批量任务应先准备输入文件",
        "debug_mode_requires_remote_preparation": "调试模式只能准备远端计算输入，不在本地执行计算",
    }
    if not errors:
        return "动作校验未通过"
    return "；".join(mapping.get(item, item) for item in errors)


def _friendly_workflow_rejection(reason):
    mapping = {
        "approval_required": "当前运行绑定旧配置版本，已有结果时需要明确批准迁移",
        "rejected_non_budget_change": "新旧配置不只是预算不同，不能合并到已有运行",
        "rejected_budget_decrease": "不能在已有运行中降低已确认预算",
        "configuration_not_confirmed": "配置尚未确认",
        "task_not_found": "目标计算任务不存在；历史动作记录不能作为 task_id",
        "task_not_retryable": "该计算任务当前不允许重试",
        "retry_limit": "该计算任务已达到重试次数上限",
    }
    return mapping.get(str(reason), str(reason))


def _proposal_directory(state_path, proposal):
    analysis = proposal.get("current_state_analysis") or {}
    if not isinstance(analysis, dict):
        return None
    record_id = analysis.get("pending_approval_record_id")
    if not record_id:
        return None
    return str(Path(state_path).parent / "approvals" / str(record_id))


def _handoff_lines(handoffs, *, collection=None):
    from phase_agent.runtime.handoff_presentation import handoff_lines

    return handoff_lines(handoffs, collection=collection)
