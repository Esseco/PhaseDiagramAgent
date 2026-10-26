"""Read-only conversation state presentation, separate from HTTP/workflow."""

from phase_agent.tools.state.task_waiting import active_pending_tasks


def waiting_training_job(state):
    jobs = [
        job
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("status") in {"inputs_prepared", "submitted", "running"}
        and not job.get("activated")
        and job.get("original_model_version") == state.get("active_model_version")
    ]
    return jobs[-1] if jobs else None


def _current_dft_review_line(state):
    """Expose current-model DFT failures before older refresh summaries."""
    model = state.get("active_model_version")
    config = state.get("confirmed_config_version")
    if not model or not config:
        return None
    rows = [
        row
        for row in state.get("tasks") or []
        if row.get("stage") in {"dft_single_point", "dft_relax"}
        and row.get("model_version") == model
        and row.get("config_version") == config
    ]
    failed = [row for row in rows if row.get("status") in {"failed", "timeout"}]
    if not failed:
        return None
    completed = sum(row.get("status") == "completed" for row in rows)
    pending = sum(
        row.get("status") in {"pending", "running", "submitted", "unknown", "submission_unknown"}
        for row in rows
    )
    pending_text = f"，另有 {pending} 个仍待回收" if pending else ""
    return (
        f"当前：当前模型 DFT 本轮已回收 {completed}/{len(rows)} 个，"
        f"{len(failed)} 个失败{pending_text}；可继续使用已回收结果，或只补全失败项；不会重跑整批。"
    )


def _current_mc_partial_review_line(state):
    """Expose a verified current-model MC partial return as an explicit choice."""
    model = state.get("active_model_version")
    candidates = []
    for bundle_id, batch in (state.get("relax_mc_batches") or {}).items():
        if (
            batch.get("model_version") != model
            or batch.get("status") in {"cancelled", "superseded"}
            or not batch.get("import_checksum")
        ):
            continue
        members = [
            row
            for row in state.get("tasks") or []
            if row.get("relax_mc_bundle_id") == bundle_id
        ]
        if not members:
            continue
        pending = [
            row
            for row in members
            if row.get("status") in {"pending", "running", "submitted", "unknown"}
        ]
        failed = [
            row
            for row in members
            if row.get("status") in {"failed", "timeout"}
        ]
        successful = [row for row in members if row.get("status") == "completed"]
        summary = batch.get("result_summary") or {}
        failed_ids = set(summary.get("failed_task_ids") or []) | {
            row.get("task_id") for row in failed
        }
        expected = int(summary.get("task_count") or len(members))
        resolved_ids = {row.get("task_id") for row in successful} | failed_ids
        if pending or not failed_ids or not successful or len(resolved_ids) < expected:
            continue
        candidates.append((bundle_id, len(successful), expected, len(failed_ids)))
    if not candidates:
        return None
    _, successful, expected, failed = candidates[-1]
    return (
        f"当前：Relax+MC 本轮已回收 {successful}/{expected} 个，{failed} 个失败；"
        "可继续使用已回收结果，或只补全失败项；不会重跑整批。"
    )


def _current_imported_mc_identity_review(state):
    """Report a completed current-model MC batch whose generation identity is unresolved."""
    model = state.get("active_model_version")
    candidates = []
    for bundle_id, batch in (state.get("relax_mc_batches") or {}).items():
        if (
            batch.get("model_version") != model
            or batch.get("status") != "completed"
            or not batch.get("import_checksum")
        ):
            continue
        members = [
            row
            for row in state.get("tasks") or []
            if row.get("relax_mc_bundle_id") == bundle_id
        ]
        if not members or any(
            row.get("status") in {"pending", "running", "submitted", "unknown"}
            for row in members
        ):
            continue
        batch_branch_ids = set(batch.get("branch_ids") or [])
        registered_branch_ids = {
            row.get("branch_id")
            for row in state.get("branch_candidates") or []
            if row.get("branch_id")
        }
        missing_branch_ids = batch_branch_ids - registered_branch_ids
        if not missing_branch_ids:
            continue
        summary = batch.get("result_summary") or {}
        total = summary.get("task_count") or len(members)
        failed = summary.get("failed_task_ids") or [
            row.get("task_id")
            for row in members
            if row.get("status") in {"failed", "timeout"}
        ]
        candidates.append((bundle_id, total, len(failed), len(missing_branch_ids)))
    if not candidates:
        return None
    _, total, failed, missing = candidates[-1]
    return (
        f"当前：Relax+MC 本轮已回收并识别 {total}/{total} 个任务，失败 {failed} 个；"
        + ("无需补全；" if not failed else "可只补全失败项；")
        + f"仍有 {missing} 个 batch branch 未登记，先核对批次身份。"
    )


def brief_chat_state(state, *, configuring=False):
    """One factual state line, without re-listing files or scientific evidence."""
    if configuring:
        return "当前：配置修订中，等待确认。"
    returned = [
        job.get("returned_results")
        for job in (state.get("remote_finetune_jobs") or {}).values()
        if job.get("status") == "results_received" and not job.get("activated")
    ]
    if returned:
        if any((row or {}).get("status") == "metadata_required" for row in returned):
            return "当前：微调结果已回收，模型清单待补齐；未激活。"
        return "当前：微调完成，仍在使用旧模型。"
    dft_review = _current_dft_review_line(state)
    if dft_review:
        return dft_review
    mc_partial = _current_mc_partial_review_line(state)
    if mc_partial:
        return mc_partial
    refresh = state.get("model_refresh") or {}
    if refresh and refresh.get("status") != "completed":
        if refresh.get("status") == "waiting_results":
            from phase_agent.analysis.state.workflow_blockers import (
                refresh_result_facts,
            )

            facts = refresh_result_facts(state)
            failed = sum(facts["status_counts"].get(s, 0) for s in ("failed", "timeout"))
            if failed:
                return (
                    f"当前：新模型结构刷新 {failed}/{facts['task_count']} 个失败，需修复后再重试。"
                )
            if (
                facts["task_count"]
                and facts["status_counts"].get("completed", 0) == facts["task_count"]
            ):
                ids = set(refresh.get("task_ids") or [])
                unknown = sum(
                    t.get("phase_identification_status") == "unknown"
                    for t in state.get("tasks") or []
                    if t.get("task_id") in ids
                    and t.get("result_disposition") != "discarded_unconverged"
                )
                if unknown:
                    return f"当前：刷新结果已全部回收，{unknown} 个结果相识别受阻；需核对识别证据后才能完成刷新。"
                if facts["discarded_unconverged_count"]:
                    return f"当前：刷新结果已回收，{facts['discarded_unconverged_count']} 个弛豫未收敛结果已排除；等待核对有效相图，不自动重试。"
                return "当前：刷新结果已全部回收，正在核对新相图及补充条件。"
            return "当前：用微调模型重新计算已有结构，输入已生成，等待结果。"
        authorization = refresh.get("input_authorization") or {}
        if (
            authorization.get("config_version") == state.get("confirmed_config_version")
            and authorization.get("approved_by")
            and (authorization.get("plan") or {}).get("model") == state.get("active_model")
        ):
            return "当前：新模型结构刷新方案已批准，输入生成待恢复。"
        return (
            "当前：新模型结构刷新方案待确认。"
            if refresh.get("status") == "approval_required"
            else f"当前：新模型结构刷新第{int(refresh.get('wave', 0)) + 1}批，等待输入准备或结果回传。"
        )
    from phase_agent.tools.policy.model_generation import new_model_generation_facts

    generation = new_model_generation_facts(state)
    active = active_pending_tasks(state.get("tasks") or [])
    if generation["required"] and not generation["ready"] and not active:
        return "当前：新模型刷新已完成，须先生成新branch并展示结果；等待生成方案审批。"
    if (
        generation.get("post_refresh")
        and not generation["first_search_completed"]
        and not active
        and not state.get("pending_execution_policies")
    ):
        mc_identity_review = _current_imported_mc_identity_review(state)
        if mc_identity_review:
            return mc_identity_review
        if generation["ready"]:
            return "当前：本轮新branch已登记，等待准备Relax+MC搜索输入。"
        return "当前：新模型刷新已完成，须先生成新branch并展示结果。"
    completed_refresh = state.get("model_refresh") or {}
    if (
        completed_refresh.get("status") == "completed"
        and completed_refresh.get("supplemental_declined_by_user")
        and not state.get("pending_execution_policies")
        and not active
    ):
        cancelled = len(completed_refresh.get("cancelled_supplemental_task_ids") or [])
        old_dft = sum(
            t.get("stage") in {"dft_single_point", "dft_relax"}
            and t.get("status") in {"pending", "running", "submitted", "unknown"}
            and not t.get("recovery_wait_waived")
            for t in state.get("tasks") or []
        )
        return (
            f"当前：已回收 {completed_refresh.get('recovered_task_count', 0)} 个刷新结果；已取消 {cancelled} 个未批准追加弛豫。"
            + (f"另有 {old_dft} 个旧 DFT 结果缺失。" if old_dft else "")
        )
    if state.get("pending_dft_recovery_question"):
        return "当前：DFT 部分结果已回收，等待是否继续回收的确认。"
    pending = state.get("pending_execution_policies") or {}
    if pending:
        names = {
            "update_mlip": "微调训练输入方案",
            "select_dft_candidates": "补充 DFT 决策",
            "allocate_mc_bohb": "MC 分配方案",
            "prepare_local_batch_files": "输入准备方案",
            "generate_branches": "branch 生成方案",
        }
        if len(pending) == 1:
            proposal = next(iter(pending.values())).get("agent_proposal") or {}
            tool = proposal.get("recommended_action") or (proposal.get("raw_action") or {}).get(
                "tool"
            )
            params = (
                (proposal.get("raw_action") or {}).get("parameters")
                or proposal.get("action_parameters")
                or {}
            )
            if (
                tool == "generate_branches"
                and params.get("sampling_mode") == "region_adaptive"
                and not params.get("regions")
            ):
                return "当前：已有branch复用方案待确认。"
            return f"当前：{names.get(tool, '操作方案')}待确认。"
        return f"当前：{len(pending)} 个方案待确认。"
    training = waiting_training_job(state)
    if training:
        return (
            "当前：微调输入已准备，等待超算提交与训练结果。"
            if training.get("status") == "inputs_prepared"
            else "当前：微调训练等待结果回传。"
        )
    if state.get("pending_mc_regeneration"):
        return "当前：MC 重生成计划待确认。"
    tasks = state.get("tasks") or []
    active = active_pending_tasks(tasks)
    if active:
        from phase_agent.runtime.local_return_status import pending_dft_return_facts

        if pending_dft_return_facts(state):
            return "当前：DFT 回传包已到，待本地校验与分析。"
        names = {
            "deep_search": "MC",
            "relax_and_feature": "Relax",
            "dft_relax": "DFT",
            "dft_single_point": "DFT",
        }
        counts = {}
        for row in active:
            name = names.get(row.get("stage"), "任务")
            batch = (state.get("relax_mc_batches") or {}).get(row.get("relax_mc_bundle_id")) or {}
            label = "项结果待核验" if batch.get("import_checksum") else "个待完成"
            key = (name, label)
            counts[key] = counts.get(key, 0) + 1
        return (
            "当前："
            + "、".join(f"{name} {count} {label}" for (name, label), count in counts.items())
            + "。"
        )
    from phase_agent.analysis.state.post_dft_assessment import post_dft_assessment

    assessment = post_dft_assessment(state, state.get("confirmed_config") or {})
    if assessment:
        if assessment["status"] != "evaluated":
            return "当前：DFT 回收结束，本轮原模型误差评估待补齐。"
        return "当前：DFT 本轮误差已评估，等待微调、补DFT、新branch或收敛/停止建议。"
    failed = sum(row.get("status") in {"failed", "timeout"} for row in tasks)
    if failed:
        return f"当前：无运行中任务，{failed} 个任务失败/超时待处理。"
    completed_mc = [
        row
        for row in tasks
        if row.get("stage") == "deep_search" and row.get("status") == "completed"
    ]
    if completed_mc:
        return f"当前：MC 已完成 {len(completed_mc)} 个任务，等待下一步评估。"
    return "当前：无待运行任务，等待下一步指令。"


def epoch_stage_line(state, *, configuring=False):
    """Display model epoch independently of MC/DFT sub-rounds."""
    from phase_agent.analysis.state.model_epoch import model_epoch

    version = state.get("active_model_version") or (state.get("active_model") or {}).get("version")
    epoch = model_epoch(state, version) or ("epoch0" if not version else "epoch未登记")
    stage = brief_chat_state(state, configuring=configuring).removeprefix("当前：").rstrip("。")
    if (
        not configuring
        and not state.get("tasks")
        and not state.get("pending_execution_policies")
        and not state.get("branches")
        and not state.get("generation_history")
        and "无待运行任务" in stage
    ):
        stage = "初始搜索，等待首轮方案"
    return f"{epoch} · {stage}"


def format_epoch_reply(reply, state, *, configuring=False, project_card=None):
    """Show the epoch once, removing only an identical repeated stage label."""
    # A pending recovery decision must be shown verbatim.  The normal concise
    # renderer derives a stage from pending plans and would otherwise prepend
    # an old input-preparation card before this gate.
    if "请明确回复‘同意回收’" in str(reply):
        return str(reply).strip()
    header = epoch_stage_line(state, configuring=configuring)
    stage = header.split(" · ", 1)[-1].rstrip("。")
    lines = []
    for line in str(reply).splitlines():
        if (
            line.startswith(("当前状态：", "当前："))
            and line.split("：", 1)[1].rstrip("。") == stage
        ):
            continue
        lines.append(line)
    if project_card is not None:
        from phase_agent.runtime.project_context import project_context_line

        header = project_context_line(project_card) + " · " + stage
    from phase_agent.runtime.response_preferences import detailed_response
    from phase_agent.runtime.concise_dialogue import concise_body

    if not detailed_response():
        card = project_card or {}
        project = card.get("display_name") or card.get("project") or state.get("project_name")
        root = card.get("project_path") or (
            (state.get("confirmed_config") or {}).get("storage") or {}
        ).get("workspace_root")
        if root and not card.get("display_name"):
            from pathlib import Path

            path = Path(root)
            if path.parent.name == "diagrams":
                project = path.parent.parent.name
            elif not project:
                project = path.name
        epoch = card.get("epoch") or epoch_stage_line(state).split(" · ", 1)[0]
        header = f"{project or '当前项目'} · {epoch} · {stage}"
        draft_status = (card.get("draft") or {}).get("status")
        if draft_status in {
            "file_changes_not_effective",
            "confirmed_not_applied",
            "pending_confirmation",
        }:
            changed = (card.get("editable_file") or {}).get("changed_paths") or []
            detail = "；差异字段：" + "、".join(changed) if changed else ""
            lines.insert(
                0,
                "设置提醒：设置草稿尚未应用"
                + detail
                + "；当前方案和已有任务回收仍使用原确认配置。",
            )
    return header + "\n" + concise_body("\n".join(lines), state)
