"""Render workflow responses without HTTP, runtime assembly or state writes."""

import os
from pathlib import Path

from run.chat_approval_rules import is_sensitive_proposal as _is_sensitive_proposal


def _is_model_failure_proposal(proposal):
    from execution_layer.workflows.run_tool_step import _model_failure_action
    return _model_failure_action((proposal or {}).get("raw_action") or {})


def format_workflow_reply(result: dict, state_path, *, verbose=None) -> str:
    if result.get("status") in {"training_results_received", "training_handoff"}:
        return result["reason"]
    if verbose is None:
        from run.response_preferences import detailed_response
        verbose = detailed_response()
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next((row.get("agent_proposal") for row in reversed(events)
                                                    if row.get("agent_proposal")), {})
    preview = ((proposal.get("raw_action") or {}).get("parameters") or {}).get("refresh_preview")
    refresh = (result.get("state") or {}).get("model_refresh") or {}
    if preview and result.get("status") == "awaiting_approval":
        rows = preview["candidates"]
        relax = sum(r["operation"] == "relax" for r in rows)
        seconds = (preview.get("runtime_budget") or {}).get("serial_seconds")
        timing = f"串行耗时粗估 {seconds/3600:.1f} 小时（不含排队）" if seconds is not None else "耗时暂无同类实测"
        return (f"当前：{preview['new_model_version']} 已激活，结构刷新待批准。\n"
                f"首批：弛豫 {relax}、单点 {len(rows)-relax}；暂缓 {preview['deferred']}。\n"
                "标准：<1 meV/atom弛豫，1–10单点，>10按Na×相抽样10%。\n"
                f"预计相对成本 {preview['initial_cost']:.1f}；最多追加一轮弛豫 "
                f"{preview['maximum_supplemental_count']} 个、成本上限 {preview['maximum_supplemental_cost']:.1f}。\n"
                "补充标准：新Ehull<1，或1–10且最大原子力>0.05 eV/Å；已弛豫项不再追加。\n"
                f"未覆盖的分层 {len(preview['coverage_gaps'])} 个；{timing}，单点成本暂按弛豫上界。\n"
                "同意将批准首批及上述上限内最多一轮补充输入生成；超限另行审批。\n"
                "仅生成超算输入；你提交GPU.sh并回传results。不本机计算。")
    if refresh and refresh.get("status") != "completed":
        reason = result.get("reason") or next((e.get("reason") for e in reversed(events) if e.get("reason")), None)
        if reason:
            return "当前：新模型结构刷新。\n" + str(reason)
    text = _workflow_reply_body(result, state_path, verbose=verbose)
    if not verbose and not result.get("post_dft_review") and result.get("status") in {
        "confirmation_required", "awaiting_remote_training", "awaiting_manual_submission",
        "failed", "rejected", "rejected_by_user", "not_configured", "prepared",
        "already_prepared", "tasks_in_progress",
    }:
        # Follow-up replies report the transition, not the previous round again.
        if result.get("status") == "failed":
            return text
        return text + f"\n完整记录：`{state_path}`"
    from analysis_layer.feedback.dft_magnetic_tables import magnetic_chat_summary
    question = (result.get("dft_recovery_question") or {}) if isinstance(result, dict) else {}
    task_ids = set(question.get("recovered_task_ids") or []) if question else None
    summary = magnetic_chat_summary((result.get("state") or {}), task_ids=task_ids, verbose=verbose) if isinstance(result, dict) else ""
    from run.post_dft_presentation import post_dft_lines, post_dft_review_lines
    assessment = post_dft_lines(result.get("state") or {}) if isinstance(result, dict) else ""
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (row.get("agent_proposal") for row in reversed(events) if row.get("agent_proposal")), {})
    from analysis_layer.state.post_dft_assessment import post_dft_assessment
    state = result.get("state") or {}
    round_assessment = post_dft_assessment(state, state.get("confirmed_config") or {})
    review_action = (proposal or {}).get("raw_action") or {"post_dft_review": result.get("post_dft_review")}
    review = post_dft_review_lines(review_action, verbose=verbose,
        finetune_enabled=round_assessment.get("finetune_enabled") if round_assessment else None)
    if review and result.get("status") == "awaiting_approval":
        text = "下一步建议（待批准）：\n" + text
    if not verbose and assessment:
        return _compact_post_dft_reply(assessment, summary, review, text)
    return "\n\n".join(part for part in (assessment, summary, review, text) if part)


def _compact_post_dft_reply(assessment, magnetic, review, action_text):
    """Keep metrics and caveats visible; retain every detail in an expandable block."""
    rows = assessment.splitlines()
    visible = rows[:4]
    visible.extend(line for line in rows[4:] if "提醒：" in line)
    visible.extend(line.split(" CSV：", 1)[0] for line in rows[4:]
                   if line.startswith("DFT 校正综合相图："))
    review_rows = review.splitlines()
    key_rows = [line for line in review_rows if line.startswith((
        "本轮发现：", "微调取舍：", "收敛与停止：", "数据局限："))]
    # The complete named tradeoff fields occur once in the expandable report.
    key_rows = [line.replace("微调取舍：", "判断：", 1) for line in key_rows]
    overview = "本轮总结：\n" + "\n".join(key_rows) if key_rows else ""
    details = "\n\n".join(part for part in (assessment, review) if part)
    expanded = ("<details>\n<summary>展开完整分析、备选方案与文件路径</summary>\n\n"
                + details + "\n\n</details>")
    return "\n\n".join(part for part in ("\n".join(visible), magnetic, overview,
                                           expanded, action_text) if part)


def _workflow_reply_body(result: dict, state_path, *, verbose=False) -> str:
    """Keep routine chat concise; full evidence stays in existing records."""
    if isinstance(result, dict) and result.get("status") == "confirmation_required":
        events = result.get("events") or []
        reason = result.get("message") or result.get("reason") or next(
            (event.get("reason") or ((event.get("execution") or {}).get("result") or {}).get("reason")
             for event in reversed(events) if event.get("reason") or
             ((event.get("execution") or {}).get("result") or {}).get("reason")), None)
        return "操作仍需确认，当前未报告执行完成。" + (f"\n原因：{reason}" if reason else
            "\n流程未提供具体原因，请查看完整记录中的审批与执行结果。")
    if isinstance(result, dict) and result.get("status") == "awaiting_dft_recovery_decision":
        return _dft_recovery_question_reply(result)
    if verbose or not isinstance(result, dict):
        return _format_workflow_reply_verbose(result, state_path, verbose=True)
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (row.get("agent_proposal") for row in reversed(events) if row.get("agent_proposal")), {})
    status = result.get("status")
    if status == "awaiting_approval" and proposal:
        if _is_model_failure_proposal(proposal):
            return "旧建议由模型回复失败产生，不是暂停决策。请重启 Agent 服务后说“继续”刷新；未执行任务。"
        # Preserve all approval limits, budget interceptions and sensitive warnings.
        text = _format_workflow_reply_verbose(result, state_path)
        from run.evidence_reference_presentation import evidence_reference_lines
        evidence_lines = evidence_reference_lines(proposal.get("raw_action") or {})
        labels = {"generate_branches": "生成 branch", "allocate_mc_bohb": "分配 MC",
                  "prepare_local_batch_files": "准备输入文件", "select_dft_candidates": "筛选 DFT"}
        tool = proposal.get("recommended_action")
        if tool == "select_dft_candidates":
            review = proposal.get("dft_template_review") or (proposal.get("raw_action") or {}).get("dft_template_review")
            if review and review.get("status") == "pending":
                return ("当前：DFT 公共参数示例待确认。\n"
                    f"代表结构：{review['candidate_id']}；模板版本：{review['revision']}。\n"
                    f"检查目录：{review['directory']}\n"
                    "请检查 INCAR、KPOINTS、GPU.sh 和 settings.json。K 点与元素相关参数按各结构生成。\n"
                    "可提出修改；或编辑 settings.json 后回复‘刷新模板’。回复‘同意’确认本版本并生成所选批次；‘拒绝’取消。\n"
                    "目前只生成一组示例，未生成批量 DFT 输入或提交任务。")
            params = proposal.get("action_parameters") or {}
            preview = params.get("dft_input_preview") or {}
            if not preview.get("task_count"):
                return "DFT 旧方案缺少结构数量与成本预览，暂不可批准。请说“继续”刷新方案。"
            if not preview.get("selected_structures"):
                return "当前是旧 DFT 方案，缺少 Na/相/Ehull 清单及新版采点信息。请说“继续”刷新，未执行任务。"
            text = text.replace("建议：`select_dft_candidates`",
                (f"建议：生成 {preview['task_count']} 个 DFT 输入（单点 {preview.get('single_point_count', 0)}、优化 {preview.get('relax_count', preview['task_count'])}）"
                 if 'single_point_count' in preview else f"建议：生成 {preview['task_count']} 个 DFT 结构优化输入"))
            selected = preview.get("selected_structures") or []
            if params.get("plan_revision_notice"):
                text += "\n方案修订：" + params["plan_revision_notice"]
            if selected:
                text += "\n\n入选结构（Na/O₂；相；Ehull eV/atom）：\n"
                def number(value):
                    return f"{value:.6g}" if isinstance(value, (int, float)) else "未知"
                text += "\n".join(f"- {row['candidate_id']}：{number(row.get('x_Na_per_O2'))}；"
                    f"{row.get('phase') or '未知'}；{number(row.get('ehull'))}" for row in selected)
            text += f"\nQBC：{preview.get('qbc_available_count', 0)}/{preview['task_count']} 个有数据；缺失值不参与评分。"
            for warning in preview.get("warnings") or []:
                text += "\n提醒：" + warning
            text += "\n批准后生成输入文件，不提交超算。"
        text = text.replace(f"建议：`{tool}`", f"建议：{labels.get(tool, tool)}")
        lines = [line for line in text.splitlines()
                 if not line.startswith("完整参数与依据：")]
        lines = [line.replace("回复“同意”执行；回复“拒绝”取消；也可以直接提出修改意见。",
                              "回复“同意”执行、“拒绝”取消，或提出修改。") for line in lines]
        return "\n".join(lines + evidence_lines)
    if status == "awaiting_manual_submission":
        wait = result.get("manual_wait") or {}
        recovered = int(wait.get("recovered_count") or result.get("recovered_count") or 0)
        names = {"deep_search": "MC", "relax_and_feature": "Relax",
                 "dft_single_point": "DFT 单点", "dft_relax": "DFT 优化"}
        stages = wait.get("waiting_by_stage") or {}
        summary = "、".join(f"{names.get(stage, stage)} {count} 个" for stage, count in sorted(stages.items()))
        lines = ([f"已回收 {recovered} 个结果。"] if recovered else [])
        for row in wait.get("dft_recovery_rounds") or []:
            if row["pending_task_ids"]:
                lines.append(f"{row['scope'].get('model_version') or '未知模型'} 的 DFT："
                             f"已回收 {row['recovered_tasks']}/{row['expected_tasks']}（{row['recovery_ratio']:.1%}），"
                             f"成功 {row['successful_tasks']}、失败 {row['failed_tasks']}。")
        lines.append(f"等待结果：{summary or str(wait.get('waiting_task_count', 0)) + ' 个任务'}。")
        collection = wait.get("result_collection") or {}
        issues = int(collection.get("invalid_count", 0)) + int(collection.get("missing_structure_count", 0))
        markers = int(collection.get("missing_marker_count", 0))
        if issues or markers:
            lines.append(f"需检查：校验/结构问题 {issues} 个，缺完成标记 {markers} 个。")
        interception = (result.get("state") or {}).get("mc_budget_interception") or {}
        if interception.get("rejected_count"):
            verbose_text = _format_workflow_reply_verbose(result, state_path)
            lines.extend(line for line in verbose_text.splitlines() if line.startswith("预算拦截："))
        if wait.get("upload_root"):
            lines.append(f"目录：`{wait['upload_root']}`")
        else:
            roots = list(dict.fromkeys(str(Path(path).parent) for path in wait.get("batch_directories") or []))
            if roots:
                lines.append("目录：" + "、".join(f"`{path}`" for path in roots))
        lines.append("尚未本机提交。Relax/MC 提交批次 GPU.sh；DFT 提交单任务 GPU.sh。")
        lines.append("完成后回传对应 results 文件夹，再说“继续”。未回传任务保持等待。")
        return "\n".join(lines)
    text = _format_workflow_reply_verbose(result, state_path)
    return "\n".join(line for line in text.splitlines() if not line.startswith((
        "详细记录：", "完整参数与结果：", "完整记录：", "示例目录：", "结构路径与详细结果：")))


def _dft_recovery_question_reply(result):
    question = result.get("dft_recovery_question") or {}
    state = result.get("state") or {}
    scope = question.get("scope") or {}
    version = scope.get("model_version") or "未知模型"
    model_round = ((state.get("upload_layout") or {}).get("model_rounds") or {}).get(version)
    label = f"第 {model_round} 轮（{version}）" if model_round is not None else f"模型 {version}"
    lines = [f"{label}：DFT 已回收 {question.get('recovered_tasks', 0)}/{question.get('expected_tasks', 0)}"
             f"（{question.get('recovery_ratio', 0):.1%}）。",
             f"成功 {question.get('successful_tasks', 0)}、失败 {question.get('failed_tasks', 0)}；"
             f"未回传 {len(question.get('pending_task_ids') or [])} 个。"]
    if state.get("round_summary_csv_path"):
        lines.append(f"轮次记录：`{state['round_summary_csv_path']}`")
    lines.extend(["还要继续回收剩余结果吗？",
                  "回复“继续回收”则等待；回复“不再回收”或“否”，则基于已有结果提出下一步方案。",
                  "不等待不会取消超算作业，也不会把缺失结果标为完成。"])
    return "\n".join(lines)


def _format_workflow_reply_verbose(result: dict, state_path, *, verbose=False) -> str:
    if not isinstance(result, dict):
        return str(result)
    events = result.get("events") or []
    proposal = result.get("agent_proposal") or next(
        (event.get("agent_proposal") for event in reversed(events) if event.get("agent_proposal")), None
    )
    if result.get("status") == "awaiting_approval" and proposal:
        proposed_tool = proposal.get("recommended_action") or (proposal.get("raw_action") or {}).get("tool")
        if not isinstance(proposed_tool, str) or not proposed_tool:
            return ("Agent 返回了无效动作，本轮没有可批准的建议。"
                    "请拒绝旧建议后重新提出；不会执行空动作。")
        cost = proposal.get("estimated_cost") or {}
        from execution_layer.policy.training_input_action import is_training_input_action
        training_inputs = is_training_input_action(proposal.get("raw_action") or {
            "tool": proposed_tool, "parameters": proposal.get("action_parameters") or {}})
        approval_files = next(
            (event.get("approval_files") for event in reversed(events)
             if event.get("approval_files")), {}
        )
        detail = approval_files.get("directory") or _proposal_directory(state_path, proposal)
        lines = [
            f"建议：`{proposal.get('recommended_action')}`",
            f"目的：{proposal.get('expected_purpose') or proposal.get('reason') or '未提供'}",
            f"预计相对成本：{cost.get('estimated_total_cost') if cost.get('estimated_total_cost') is not None else '待任务展开后估算'}",
        ]
        if training_inputs:
            lines = ["建议：微调模型，生成超算训练提交文件",
                     "目的：按本轮分析准备committee、K折评估和全数据主模型训练输入。",
                     "训练耗时：尚无训练实测，待估算；输入生成成本不代表超算训练成本。",
                     "批准后仅生成输入与GPU.sh，由你上传提交；不本机训练、不自动激活模型。"]
        recovered_count = int(result.get("recovered_count") or 0)
        runtime = cost.get("runtime_budget") or {}
        if runtime.get("serial_seconds") is not None:
            hours = runtime["serial_seconds"] / 3600
            low, high = runtime["sample_range_seconds"]
            lines.append(f"耗时粗估：串行累计约 {hours:.2g} 小时（观测范围 {low/3600:.2g}–{high/3600:.2g} 小时）；不含排队，并行后实际完成时间另计。")
        elif proposed_tool in {"allocate_mc_bohb", "prepare_local_batch_files", "select_dft_candidates", "run_calculation_stage"}:
            lines.append("耗时粗估：暂无足够实测或任务规模信息，暂无法估算；相对成本不代表小时。")
        if recovered_count:
            lines.insert(0, f"已校验并回收 {recovered_count} 个任务结果，结果与成本已入账。")
        if proposal.get("recommended_action") == "generate_branches":
            params = proposal.get("action_parameters") or (proposal.get("raw_action") or {}).get("parameters") or {}
            quotas = params.get("quotas")
            names = {"coverage": "补覆盖", "composition": "补Na组分", "competing_phase": "竞争相",
                     "periodic_extension": "扩展超胞", "tm_ordering": "TM排布"}
            strategy = ("、".join(f"{names.get(name, name)} {count}" for name, count in quotas.items())
                        if isinstance(quotas, dict) and quotas else
                        f"覆盖补充（预计 {params.get('total_quota', '按配置')} 个；具体分配由现有 branch 覆盖决定）")
            if verbose or not params.get("generation_plan"):
                lines.append(f"生成策略：{strategy}")
            plan = params.get("generation_plan") or []
            for row in plan:
                target = "全部合法相" if row["phase"] == "all" else row["phase"]
                x = f"，Na/O₂ {row['na_min']:g}–{row['na_max']:g}" if row.get("na_min") is not None else ""
                cap = f"，det(H)≤{row['max_det_H']}" if row.get("max_det_H") is not None else ""
                lines.append(f"- {target}{x}{cap}：{names.get(row['strategy'], row['strategy'])}候选 {row['quota']}；{row['reason']}")
            if params.get("max_det_H") is not None:
                lines.append(f"本轮超胞上限：det(H) ≤ {params['max_det_H']}")
            else:
                lines.append("本轮超胞上限：未单独设置，按已确认的 H 边界")
            initial_count = params.get("initial_states_per_branch")
            lines.append(f"入选上限：{params.get('batch_size', '按配置')} 个 branch；"
                         f"每个 branch 静电能前 10 中至多取 {initial_count if initial_count is not None else '按配置'} 个初态")
            preview = params.get("generation_cost_preview") or {}
            if preview:
                lines.append(f"后续成本粗估上界：Relax+MC {preview['relax_mc_cost_upper']:.3g}；"
                             f"若全部入选branch做单点DFT {preview['dft_cost_scenario_upper']:.3g}（情景，非授权）。")
                seconds = preview.get("serial_seconds_upper")
                lines.append(f"Relax+MC串行耗时粗估：{seconds/3600:.2g} 小时，不含排队。" if seconds is not None
                             else "耗时：实测样本不足，暂不换算小时。")
                lines.append("以上不含第二段MC及DFT优化；后续派发仍单独审批。")
        if proposal.get("recommended_action") == "allocate_mc_bohb":
            params = proposal.get("action_parameters") or (proposal.get("raw_action") or {}).get("parameters") or {}
            preview = params.get("budget_preview") or {}
            lines.append(f"MC 总步数目标：{params.get('mc_budget', '待定')} 步；"
                         f"本次计划：{preview.get('requested_steps', '待重算')} 步。")
            if preview.get("phase_diagram_version"):
                lines.append(f"Ehull 标准：当前 MLIP 相图版本 {preview['phase_diagram_version']} 的 eV/atom 值；"
                             "批准前会复核相图版本与结构能量。")
        fallback_reason = (proposal.get("raw_action") or {}).get("fallback_reason")
        if fallback_reason:
            lines.append(f"回退原因：`{fallback_reason}`")
        if detail:
            lines.append(f"完整参数与依据：`{detail}`")
        lines.extend(["", "回复“同意”执行；回复“拒绝”取消；也可以直接提出修改意见。"])
        if _is_sensitive_proposal(proposal):
            lines.extend([
                "该建议涉及敏感操作，仍需在本机审批页确认具体影响：",
                f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval",
            ])
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
            (event.get("reason") or ((event.get("execution") or {}).get("result") or {}).get("reason")
             for event in reversed(events) if event.get("reason") or ((event.get("execution") or {}).get("result") or {}).get("reason")), None)
        return f"{reason or '本轮要求超过已确认配置；请先修订配置。'}\n未执行生成或计算。"
    if status == "awaiting_remote_training":
        payload = (result.get("execution") or {}).get("result") or next(
            (((event.get("execution") or {}).get("result") or {}) for event in reversed(events)
             if ((event.get("execution") or {}).get("result") or {}).get("reason")), {})
        return payload.get("reason") or result.get("reason") or "微调输入已准备，等待超算训练；未在本机训练或激活模型。"
    action = ((result.get("final_action") or {}).get("tool")
              or (proposal or {}).get("recommended_action")
              or next(((event.get("final_action") or {}).get("tool") for event in reversed(events)
                       if (event.get("final_action") or {}).get("tool")), None))
    validation = result.get("validation") or next(
        (event.get("validation") for event in reversed(events) if event.get("validation")), {}
    )
    errors = validation.get("errors") or []
    detail_path = str(Path(state_path))
    if status == "awaiting_manual_submission":
        wait = result.get("manual_wait") or {}
        recovered_count = int(wait.get("recovered_count") or result.get("recovered_count") or 0)
        waiting_count = int(wait.get("waiting_task_count") or 0)
        by_stage = wait.get("waiting_by_stage") or {}
        stage_label = ("MC" if set(by_stage) == {"deep_search"} else
                       "Relax" if set(by_stage) == {"relax_and_feature"} else
                       "DFT" if set(by_stage) <= {"dft_single_point", "dft_relax"} else "计算")
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
                for key, row in sorted((interception.get("strata") or {}).items()))
            lines.append(
                f"预算拦截：入选 {interception['accepted_count']} 个，"
                f"拦截 {interception['rejected_count']} 个；原因："
                f"{', '.join(interception.get('reasons') or [])}。"
                f"按层级/相分布：{breakdown}。"
                "如果要运行全部，请先查看完整估计成本并明确批准全量方案。")
        lines.append("本机没有提交作业。请将同一阶段的批次目录放在同一轮次目录下：Relax/MC 每个批次根目录的 `GPU.sh` 提交一次；DFT 则在单任务目录提交 `GPU.sh`。")
        if wait.get("upload_root"):
            lines.append(f"本地任务总目录：`{wait['upload_root']}`")
        if wait.get("upload_plan_path"):
            lines.append(f"按 branch 分组的完整上传清单：`{wait['upload_plan_path']}`。每个 branch 的所有任务目录都列在一起。")
        directories = wait.get("batch_directories") or wait.get("task_directories") or []
        if directories:
            lines.append("待提交的批次目录：")
            lines.extend(f"- `{path}`" for path in directories[:5])
            if len(directories) > 5:
                lines.append(f"- 其余 {len(directories) - 5} 个批次见任务清单和 state 文件。")
        results_directories = wait.get("results_directories") or []
        if results_directories:
            lines.append("本阶段统一结果目录：" + "、".join(f"`{path}`" for path in results_directories))
        lines.append(
            "本阶段任务完成后，只需下载对应的整个 `results/` 文件夹，放回本地同一阶段目录，"
            "其中含所有任务的 `result.json`、`task.finished.json` 和最终结构；再对 Agent 说“继续”。"
            "Agent 会按任务校验并回收；未回传任务继续等待，不会重复准备或提交。"
            "旧批次仍按原任务目录回收。"
        )
        return "\n".join(lines)
    if status == "failed":
        execution = result.get("execution") or next(
            (event.get("execution") for event in reversed(events) if event.get("execution")), {}
        )
        reason = ((execution or {}).get("error") or
                  ((execution or {}).get("result") or {}).get("reason") or
                  result.get("reason") or "未知错误")
        return f"本轮失败：{str(reason)[:500]}\n详细记录：`{detail_path}`"
    if status == "rejected_by_user":
        return f"已取消本轮建议；未执行任何动作。\n详细记录：`{detail_path}`"
    if status == "rejected":
        reason = _friendly_validation_errors(errors)
        workflow_reason = result.get("message") or result.get("reason") or next(
            (event.get("reason") for event in reversed(events) if event.get("reason")), None)
        if not errors and workflow_reason:
            reason = _friendly_workflow_rejection(workflow_reason)
        if not errors and not workflow_reason:
            execution = result.get("execution") or next(
                (event.get("execution") for event in reversed(events) if event.get("execution")), {}
            )
            payload = (execution or {}).get("result") or {}
            reason = _friendly_workflow_rejection(payload.get("reason") or execution.get("error") or "执行动作被拒绝")
        return f"本轮未执行：{reason}\n详细记录：`{detail_path}`"
    if status == "not_configured":
        execution = result.get("execution") or next(
            (event.get("execution") for event in reversed(events) if event.get("execution")), {}
        )
        payload = (execution or {}).get("result") or {}
        reason = ((execution or {}).get("error") or payload.get("reason") or
                  result.get("reason") or next((event.get("reason") for event in reversed(events)
                      if event.get("reason")), None) or "执行接口未配置")
        if action == "allocate_mc_bohb" and payload.get("actions"):
            return (f"已分配 {len(payload['actions'])} 个 MC 任务的预算，但输入文件尚未准备好：{reason}。"
                    f"任务已保留；修正原因后可继续准备 MC 输入。\n详细记录：`{detail_path}`")
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
        return f"本轮未执行：`{action or 'workflow'}` 未能准备（{reason}）。\n详细记录：`{detail_path}`"
    if status in {"prepared", "already_prepared"} and action == "prepare_local_batch_files":
        execution = result.get("execution") or next(
            (event.get("execution") for event in reversed(events) if event.get("execution")), {})
        payload = (execution or {}).get("result") or {}
        if payload.get("batches"):
            locations = "、".join(row.get("directory", "") for row in payload["batches"][:3])
            return (f"已准备 {payload.get('task_count', 0)} 个 Relax 任务、"
                    f"{payload.get('batch_count', 0)} 个上传批次；未提交超算。\n"
                    f"按 branch 上传清单：`{payload.get('upload_plan_path') or detail_path}`。\n"
                    f"示例目录：`{locations}`")
        return f"Relax 输入已准备过，没有重复创建或扣费。\n详细记录：`{detail_path}`"
    if status in {"completed", "planned_only", "tasks_in_progress"}:
        if status == "completed" and action == "generate_branches":
            execution = result.get("execution") or next(
                (event.get("execution") for event in reversed(events) if event.get("execution")), {}
            )
            summary = ((execution or {}).get("result") or {}).get("summary") or {}
            if summary:
                phases = "、".join(f"{name} {count}" for name, count in
                                  sorted((summary.get("phase_counts") or {}).items())) or "无"
                det_range = summary.get("det_H_range") or []
                cells = (f"入选 det(H) {det_range[0]}–{det_range[1]}" if len(det_range) == 2
                         else "无入选超胞")
                cap = summary.get('max_det_H')
                if cap is not None:
                    cells += f"（本轮设定上限 {cap}）"
                return (
                    f"已生成 {summary.get('registered_branches', 0)} 个 branch、"
                    f"{summary.get('registered_structures', 0)} 个初始结构。"
                    f"相：{phases}；{cells}。\n"
                    f"生成 {summary.get('proposed_branches', 0)} 个候选，"
                    f"入选 {summary.get('selected_branches', 0)} 个 branch，"
                    f"静电能初始化失败 {summary.get('initialization_failed_branches', 0)} 个，"
                    f"构型去重后 {summary.get('unique_structures', 0)} 个。\n"
                    f"结构路径与详细结果：`{detail_path}`"
                )
        labels = {"completed": "已完成", "planned_only": "已生成计划但未执行",
                  "tasks_in_progress": "任务已进入处理中"}
        return (f"{labels[status]}：`{action or 'workflow'}`。\n"
                f"完整参数与结果：`{detail_path}`")
    return f"工作流状态：`{status}`。\n完整记录：`{detail_path}`"


def _friendly_validation_errors(errors):
    mapping = {
        "formal_tool_requires_task_key": "动作缺少内部幂等编号",
        "configuration_not_confirmed": "配置尚未确认",
        "config_version_mismatch": "当前运行仍绑定旧配置版本；请新建运行，或在尚无科学任务时重新发送该指令",
        "invalid_budget": "Agent 返回的预算格式无效；预算必须是一个非负数值",
        "tool_handler_not_configured": "该动作尚无执行函数，请改选已接通的工具",
        "duplicate_effective_decision": "相同任务已经提交或完成",
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
