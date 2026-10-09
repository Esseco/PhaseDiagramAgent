"""Interactive chat application; independent of HTTP/OpenAI-compatible transport."""
from __future__ import annotations
import json
import threading
import uuid
from pathlib import Path
from copy import deepcopy
from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.file_protocol import read_json, write_json
from run.chat_approval_rules import is_sensitive_proposal as _is_sensitive_proposal
from run.chat_state_presentation import brief_chat_state
from run.workflow_reply_presentation import format_workflow_reply, _is_model_failure_proposal

class OpenWebUIRequestError(ValueError):
    """Invalid or ambiguous chat request."""


class RunWorkflowChatHandler:
    """Adapt actual user turns to one-step calls of the existing run_workflow.

    ``workflow_kwargs`` is the existing runtime composition: manager,
    phase_references, confirmed config, Agent client and configured adapters.
    """

    def __init__(self, workflow_kwargs: dict, *, workflow=None, history_prompt=False,
                 new_run_factory=None, deepseek_model_switcher=None,
                 execution_mode="interactive", config_revision_factory=None,
                 config_intent_client=None):
        if not isinstance(workflow_kwargs, dict) or not workflow_kwargs.get("state_path"):
            raise ValueError("workflow_kwargs must include a persistent state_path")
        self.workflow_kwargs = dict(workflow_kwargs)
        self.decision_backend = "langgraph"
        self.state_path = Path(workflow_kwargs["state_path"])
        self.workflow = workflow
        self.lock = threading.RLock()
        self.history_prompt = bool(history_prompt)
        self.history_decision = None if history_prompt else "continue"
        self.new_run_factory = new_run_factory
        self.deepseek_model_switcher = deepseek_model_switcher
        self.config_revision_factory = config_revision_factory
        self.config_intent_client = config_intent_client
        self.config_delegate = None
        if execution_mode not in {"interactive", "autonomous"}:
            raise ValueError("execution_mode 必须是 interactive 或 autonomous")
        self.execution_mode = execution_mode
        self.conversation_id = None

    def __call__(self, messages, *, conversation_id=None):
        reply = self._respond(messages, conversation_id=conversation_id)
        if _open_webui_metadata_reply(_latest_user_message(messages)) is not None:
            return reply
        if str(reply).startswith("当前项目状态："):
            return reply  # Progress already contains its stage; avoid duplicate headers.
        state = read_json(self.state_path, {}) or {}
        return brief_chat_state(state, configuring=self.config_delegate is not None) + "\n" + str(reply)

    def _respond(self, messages, *, conversation_id=None):
        user_message = _latest_user_message(messages)
        metadata_reply = _open_webui_metadata_reply(user_message)
        if metadata_reply is not None:
            return metadata_reply
        with self.lock:
            state = read_json(self.state_path, {}) or {}
            if self.conversation_id not in {None, conversation_id}:
                raise OpenWebUIRequestError(
                    "此本地运行时已绑定另一个 Open WebUI 会话。当前审批状态是进程级单用户状态；"
                    "请使用原会话，或为另一用户启动独立服务和 state。"
                )
            # Read-only inspection must precede history recovery and all mutations.
            if self.config_delegate is None:
                from run.plan_queries import is_plan_query, pending_plan_reply
                if is_plan_query(user_message):
                    return pending_plan_reply(state)
            if self.config_delegate is None:
                from run.artifact_queries import finetune_location_reply
                location = finetune_location_reply(user_message, state)
                if location is not None:
                    return location
            if self.config_delegate is None and _is_status_command(user_message):
                return format_status_reply(state)
            from execution_layer.state.dft_recovery_decision import classify_dft_recovery_reply
            recovery_decision = classify_dft_recovery_reply(user_message, state)
            if recovery_decision is not None and self.config_delegate is None:
                self.conversation_id = conversation_id
                result = self._run(f"webui-{uuid.uuid4().hex}", None, user_message,
                                   dft_recovery_decision=recovery_decision)
                prefix = ("已记录：本轮剩余 DFT 不再等待；未取消超算任务，下一步仍需正常审批。\n"
                          if recovery_decision["decision"] == "close" else "已记录：继续等待本轮剩余 DFT。\n")
                if not any(row.get("question_id") == recovery_decision["question_id"]
                           and row.get("decision") == recovery_decision["decision"]
                           for row in (result.get("state") or {}).get("dft_recovery_decisions") or []):
                    prefix = ""
                return prefix + format_workflow_reply(result, self.state_path)
            from execution_layer.local.regenerate_mc_inputs import is_mc_regeneration_request, is_mc_regeneration_confirmation
            from execution_layer.local.identify_rerun_plan import is_rerun_plan_request
            known = (_phase_csv_request(user_message) or _is_status_command(user_message)
                     or (self.history_decision is None and _classify_history_decision(user_message) is not None)
                     or _is_navigation_command(user_message) or is_rerun_plan_request(user_message)
                     or is_mc_regeneration_request(user_message) or is_mc_regeneration_confirmation(user_message)
                     or classify_user_decision(user_message) in {"approve", "reject"}
                     or _is_config_migration_approval(user_message))
            if not known and self.config_delegate is None:
                from run.chat_intent_routing import normalize_chat_intent
                intent_result = normalize_chat_intent(
                    user_message, state, agent_client=self.workflow_kwargs.get("agent_client"),
                    messages=messages)
                if "reply" in intent_result:
                    return intent_result["reply"]
                user_message = intent_result["message"]
            if _phase_csv_request(user_message):
                from analysis_layer.phase.export_current_phase_diagram import export_current_phase_diagram
                method = ("combined" if any(word in user_message.lower() for word in
                          ("综合", "总相图", "校正", "combined")) else
                          "dft" if "dft" in user_message.lower() else "mlip")
                try:
                    before = ((state.get("phase_diagrams") or {}).get(method) or {}).get("csv_path")
                    path, count, version = export_current_phase_diagram(
                        state, directory=self.workflow_kwargs.get("phase_diagram_directory"),
                        method=method)
                    if str(path) != before:
                        write_json(self.state_path, state)
                except (OSError, ValueError) as error:
                    return f"当前相图 CSV 未导出：{error}。未推进搜索或修改任务。"
                return f"当前 {method.upper()} 相图 CSV（版本 {version}，{count} 个结构）：`{path}`"
            if self.config_delegate is not None:
                return self.config_delegate(messages, conversation_id=conversation_id)
            state, stale_pending_removed = _drop_finished_pending(state)
            if stale_pending_removed:
                write_json(self.state_path, state)
            from execution_layer.local.regenerate_finetune_inputs import (
                CONFIRM, is_finetune_regeneration_request, plan_finetune_regeneration,
                regenerate_finetune_inputs,
            )
            pending_training = state.get("pending_finetune_regeneration")
            training_confirm = str(user_message).strip() == CONFIRM
            training_reject = (pending_training and not state.get("pending_execution_policies")
                               and classify_user_decision(user_message) == "reject")
            if training_reject:
                state.pop("pending_finetune_regeneration", None)
                write_json(self.state_path, state)
                return "已取消微调输入重生成方案；文件与训练记录未改变。"
            if training_confirm or is_finetune_regeneration_request(user_message, state):
                from config_layer.runtime.build_effective_run_config import build_effective_run_config
                config = build_effective_run_config(
                    self.workflow_kwargs["config_session"], self.workflow_kwargs["run_config"])
                try:
                    if training_confirm:
                        if not pending_training:
                            return "尚无微调输入重生成方案，请先说“重新生成原轮微调输入”。未执行。"
                        result = regenerate_finetune_inputs(state, config, pending_training)
                        write_json(self.state_path, result["state"])
                        return result["reason"] + "\n原轮编号不变；旧输入已备份，未提交或训练。"
                    plan = plan_finetune_regeneration(state, config)
                    state["pending_finetune_regeneration"] = plan
                    write_json(self.state_path, state)
                    operation = "备份旧输入后重新生成" if plan["exists"] else "目录不存在，直接重新生成"
                    return (f"微调原轮输入：`{plan['directory']}`。\n方案：{operation}，使用当前有效训练数据和参数；保留原轮编号，不追加轮次。\n"
                            f"请确认旧输入是否已在超算提交；已提交则不要重生成。执行请单独回复“{CONFIRM}”；“拒绝”取消。当前未移动或生成文件。")
                except (ValueError, OSError, RuntimeError) as error:
                    return f"微调输入重生成未执行或已回滚：{error}。未提交或训练。"
            from execution_layer.local.regenerate_mc_inputs import (
                is_mc_regeneration_request, is_mc_regeneration_confirmation,
                plan_mc_regeneration, regenerate_mc_inputs,
            )
            pending_mc = state.get("pending_mc_regeneration")
            reject_mc = (pending_mc and not state.get("pending_execution_policies")
                         and classify_user_decision(user_message) == "reject")
            mc_request = is_mc_regeneration_request(user_message)
            mc_confirmed = is_mc_regeneration_confirmation(user_message)
            if mc_request or mc_confirmed or reject_mc:
                if reject_mc:
                    state.pop("pending_mc_regeneration", None)
                    write_json(self.state_path, state)
                    return "已取消 MC 重生成计划；任务状态和文件未改变。"
                if mc_confirmed and not pending_mc:
                    return "尚无待确认的 MC 重生成方案。请先说“重新生成当前轮次 MC 任务”，查看范围和影响。"
                from config_layer.runtime.build_effective_run_config import build_effective_run_config
                config = build_effective_run_config(
                    self.workflow_kwargs["config_session"], self.workflow_kwargs["run_config"])
                root = config["upload_batches_directory"]
                try:
                    plan = plan_mc_regeneration(state, root)
                    if not mc_confirmed:
                        state["pending_mc_regeneration"] = plan
                        write_json(self.state_path, state)
                        existing = ("该目录存在；确认后会删除其中已生成的 MC 输入。"
                                    if plan["exists"] else
                                    "该目录不存在；无需删除文件，但会废弃旧 MC 状态。")
                        stage_limit = ((config.get("budgets") or {}).get("stage_limits") or {}).get("deep_search") or {}
                        cost_limit = stage_limit.get("max_cost")
                        over_limit = (f"；高于当前 MC 阶段成本上限 {float(cost_limit):.2f}"
                                      if cost_limit is not None
                                      and float(plan["estimated_relative_cost"]) > float(cost_limit) else "")
                        strata = "、".join(
                            f"{label} {count} 个"
                            for label, count in plan["allocation_by_tier_phase"].items())
                        return (
                            "当前只提出方案，尚未删除文件或生成任务。\n"
                            f"当前轮 MC 目录：`{plan['directory']}`。{existing}\n"
                            f"将废弃 {len(plan['old_task_ids'])} 个旧 MC 任务及 "
                            f"{plan['old_batch_count']} 条旧批次记录，保留其他阶段和已批准的分配方案。\n"
                            f"重新生成 {plan['allocation_count']} 个独立 MC 算例、"
                            f"共 {plan['total_mc_steps']} 步；估计相对成本 "
                            f"{float(plan['estimated_relative_cost']):.2f}{over_limit}。\n"
                            f"每个 GPU 批次最多 {plan['mc_batch_size']} 个算例，"
                            f"按当前兼容条件预计 {plan['estimated_batch_count']} 个批次；"
                            "每个算例仍有独立结果。\n"
                            f"分层/分相数量：{strata}。\n"
                            f"新 MC-sampling 从 {plan['mc_sampling_start']:04d} 编号，"
                            f"远端批次从 remote-{plan['remote_batch_start']:06d} 编号；"
                            "仅生成本地输入，不提交超算作业。\n"
                            "请核对方案和旧任务是否已在超算提交；确定废弃并执行时，"
                            "单独回复“确认重新生成当前轮MC任务”。回复“拒绝”可取消。")
                    if plan != pending_mc:
                        raise ValueError("MC 状态或目录在确认后发生变化；未执行，请重新提出请求并核对新方案")
                    state.pop("pending_mc_regeneration", None)
                    upload = regenerate_mc_inputs(
                        state, approved_plan=plan, upload_root=root, config=config,
                        manager=self.workflow_kwargs["manager"],
                        phase_references=self.workflow_kwargs["phase_references"],
                        config_version=state["confirmed_config_version"],
                        allow_delete=True)
                    write_json(self.state_path, upload["state"])
                    return (f"已按原批准的完整方案重新生成 {upload['task_count']} 个 MC 输入，"
                            f"分为 {upload['batch_count']} 个批次；未提交超算作业。"
                            f"目录：`{plan['directory']}`。")
                except (KeyError, OSError, ValueError, RuntimeError) as error:
                    return f"MC 输入重生成未完成：{error}。未提交超算作业。"
            from execution_layer.local.identify_rerun_plan import (
                is_rerun_plan_request, identify_rerun_plan, format_rerun_plan,
            )
            if is_rerun_plan_request(user_message):
                plan = identify_rerun_plan(user_message, state)
                if (plan.get("status") == "identified" and plan.get("intent") == "generate"
                        and plan.get("action", {}).get("tool") == "select_dft_candidates"):
                    from execution_layer.local.regenerate_dft_files import regenerate_dft_files
                    try:
                        result = regenerate_dft_files(state, user_message,
                            manager=self.workflow_kwargs.get("manager"),
                            upload_root=self.workflow_kwargs["run_config"]["upload_batches_directory"])
                    except (ValueError, OSError, RuntimeError) as error:
                        return f"DFT 文件重生成未执行或已回滚：{error}。未提交作业。"
                    return (f"已按原批准参数重新生成 {result['task_count']} 个 DFT 任务的输入文件。"
                        f"任务身份和预算不变，未提交作业。旧输入备份：{result['backup_directory']}。")
                return format_rerun_plan(plan)
            from run.resolve_deepseek_model_request import resolve_deepseek_model_request
            requested_model = resolve_deepseek_model_request(user_message)
            if requested_model:
                if not callable(self.deepseek_model_switcher):
                    return "此运行时未配置 DeepSeek 模型切换接口；未修改任何设置。"
                try:
                    selected_model, client = self.deepseek_model_switcher(requested_model)
                except (OSError, TypeError, ValueError) as error:
                    return f"DeepSeek 模型切换失败：{type(error).__name__}: {error}。运行时设置未更改。"
                self.workflow_kwargs["agent_client"] = client
                # A search-model switch must not replace the dedicated intent role.
                # The intent client keeps its own prompt and small token budget.
                return _deepseek_switch_reply(selected_model)
            if _is_config_migration_approval(user_message):
                self.conversation_id = conversation_id
                invocation_id = f"config-migration-{uuid.uuid4().hex}"
                result = self._run(
                    invocation_id, None, user_message, approve_config_migration=True,
                )
                return format_workflow_reply(result, self.state_path)
            explicit_config_scope = any(word in user_message.lower() for word in (
                "配置", "设置文件", "json", "以后", "默认", "永久", "所有轮"))
            from decision_layer.agent.resolve_mc_budget_feedback import (
                resolve_mc_budget_feedback, resolve_mc_full_plan_steps,
            )
            pending_actions = state.get("pending_execution_policies") or {}
            pending_action = (((next(iter(pending_actions.values())).get("agent_proposal") or {})
                               .get("raw_action") or {}) if len(pending_actions) == 1 else {})
            mc_steps = (resolve_mc_budget_feedback(user_message, pending_action)
                        if not explicit_config_scope else None)
            full_plan_steps = (resolve_mc_full_plan_steps(user_message, pending_action)
                               if not explicit_config_scope else None)
            requested_mc_steps = mc_steps or full_plan_steps
            confirmed_maximum = ((state.get("confirmed_config") or {}).get("round_strategy") or {}).get(
                "maximum_mc_budget")
            if (requested_mc_steps is not None and confirmed_maximum is not None
                    and requested_mc_steps > int(confirmed_maximum)):
                if not callable(self.config_revision_factory):
                    return (f"已理解为本轮 MC 总预算 {requested_mc_steps} 步；已确认配置的单轮上限是 "
                            f"{confirmed_maximum} 步。配置修订入口未配置，不能生成超上限任务。")
                revised_state = deepcopy(state)
                revised_state["mc_budget_intent"] = {"steps": requested_mc_steps,
                    "source_config_version": state.get("confirmed_config_version")}
                try:
                    self.config_delegate = self.config_revision_factory(revised_state)
                except (OSError, TypeError, ValueError) as error:
                    return f"已理解为本轮 MC 总预算 {requested_mc_steps} 步；无法进入配置修订：{error}。"
                return self.config_delegate.revise_mc_budget_limit(
                    requested_mc_steps, conversation_id=conversation_id,
                )
            pending_generation = any(
                ((row.get("agent_proposal") or {}).get("raw_action") or {}).get("tool") == "generate_branches"
                for row in (state.get("pending_execution_policies") or {}).values())
            batch_feedback = (pending_generation and not explicit_config_scope
                              and any(word in user_message.lower() for word in ("本轮", "branch", "初态", "超胞", "det(h)")))
            batch_feedback = batch_feedback or requested_mc_steps is not None
            navigation_only = _is_navigation_command(user_message)
            if (not navigation_only and not pending_actions
                    and classify_user_decision(user_message) not in {"approve", "reject"}
                    and not batch_feedback):
                from decision_layer.agent.classify_config_edit_intent import classify_config_edit_intent
                classifier = self.config_intent_client or self.workflow_kwargs.get("agent_client")
                intent = classify_config_edit_intent(user_message, agent_client=classifier)
                from run.configuration_chat import _is_config_revision_request
                if intent == "edit" or _is_config_revision_request(user_message):
                    if not callable(self.config_revision_factory):
                        return "配置编辑入口未配置；本轮没有修改文件或执行旧建议。"
                    try:
                        self.config_delegate = self.config_revision_factory(state)
                    except (OSError, TypeError, ValueError) as error:
                        return f"无法安全进入配置修订：{error}。本轮未修改文件或执行动作。"
                    instruction = "请直接修改配置文件并保存；用户原话：" + user_message
                    return self.config_delegate(
                        [{"role": "user", "content": instruction}],
                        conversation_id=conversation_id,
                    )
            if self.history_decision is None:
                self.conversation_id = conversation_id
                decision = _classify_history_decision(user_message)
                if decision is None:
                    return format_history_prompt(state, self.workflow_kwargs.get("manager"))
                self.conversation_id = conversation_id
                if decision == "new":
                    if not callable(self.new_run_factory):
                        raise OpenWebUIRequestError("运行时未配置安全的新建运行目录工厂。")
                    replacement = self.new_run_factory()
                    if isinstance(replacement, RunWorkflowChatHandler):
                        self.workflow_kwargs = dict(replacement.workflow_kwargs)
                        self.state_path = replacement.state_path
                        self.workflow = replacement.workflow
                        self.new_run_factory = replacement.new_run_factory
                        self.deepseek_model_switcher = replacement.deepseek_model_switcher
                        self.config_revision_factory = replacement.config_revision_factory
                        self.config_intent_client = replacement.config_intent_client
                        self.execution_mode = replacement.execution_mode
                        self.config_delegate = None
                        if hasattr(replacement, "runtime_config_path"):
                            self.runtime_config_path = replacement.runtime_config_path
                        if hasattr(replacement, "knowledge_library_root"):
                            self.knowledge_library_root = replacement.knowledge_library_root
                        self.history_decision = "new"
                        return (f"已新建独立运行：`{self.state_path.parent}`。"
                                "旧 state/ledger 未修改。请发送下一条搜索指令。")
                    if "agent_client" in self.workflow_kwargs:
                        replacement["agent_client"] = self.workflow_kwargs["agent_client"]
                    self.workflow_kwargs = dict(replacement)
                    self.state_path = Path(replacement["state_path"])
                    state = {}
                    self.history_decision = "new"
                    return f"已新建独立运行：`{self.state_path.parent}`。旧 state/ledger 未修改。请发送下一条搜索指令。"
                self.history_decision = "continue"
                pending = state.get("pending_execution_policies") or {}
                if len(pending) > 1:
                    raise OpenWebUIRequestError(
                        "历史 state 含多个待审批 action，无法安全确定审批目标；"
                        "请先通过离线状态工具恢复为唯一待审批状态。"
                    )
                if pending:
                    pending_id = next(iter(pending))
                    pending_record = next(iter(pending.values()))
                    proposal = pending_record.get("agent_proposal")
                    if self.execution_mode == "interactive" and _is_model_failure_proposal(proposal):
                        refreshed = self._run(pending_id, None, user_message)
                        return format_workflow_reply(refreshed, self.state_path)
                    if self.execution_mode == "interactive" and _is_pending_relax_input_proposal(proposal):
                        # A pending Relax-input proposal can become stale after
                        # remote Relax results are recovered. Re-enter the
                        # workflow so run_tool_step can replace it with the
                        # next valid action (for example, MC allocation).
                        refreshed = self._run(
                            pending_id,
                            {"decision": "comment", "comment": "继续"},
                            user_message,
                        )
                        return format_workflow_reply(refreshed, self.state_path)
                    if self.execution_mode == "interactive" and _is_pending_mc_proposal(proposal, state):
                        # A history-resume command must re-enter the workflow
                        # so stale MC proposals cannot be echoed as a fresh plan.
                        refreshed = self._run(pending_id, None, user_message)
                        return format_workflow_reply(refreshed, self.state_path)
                    if self.execution_mode == "interactive" and _unsafe_debug_calculation(proposal, state):
                        revised = self._run(pending_id, {"decision": "comment",
                            "comment": "调试模式先准备已有结构的 Relax 输入文件"}, user_message)
                        return format_workflow_reply(revised, self.state_path)
                    # Resume must recheck the scientific stage, not echo a cached
                    # proposal that may predate DFT recovery or model assessment.
                    refreshed = self._run(pending_id, None, user_message)
                    return format_workflow_reply(refreshed, self.state_path)
                from execution_layer.remote.summarize_manual_upload_wait import (
                    summarize_manual_upload_wait,
                )
                if summarize_manual_upload_wait(state):
                    invocation_id = f"webui-{uuid.uuid4().hex}"
                    result = self._run(invocation_id, None, user_message)
                    return format_workflow_reply(result, self.state_path)
                return "已继续原运行并加载 state/ledger。请发送下一条搜索指令；本次确认不会批准任何 action。"
            pending = state.get("pending_execution_policies") or {}
            from run.configuration_chat import _is_config_json_import_command
            if _is_config_json_import_command(user_message):
                version = state.get("confirmed_config_version") or "未知"
                return (f"当前搜索仍锁定配置 {version}；这条消息不会导入 JSON，也不会修改待审批建议。"
                        "请进入配置修订并确认新快照后再生成；旧版本提案不能自动套用新参数。")
            path_clarification = _workspace_path_clarification_reply(user_message, state)
            if path_clarification is not None:
                return path_clarification
            if pending:
                if len(pending) != 1:
                    raise OpenWebUIRequestError("存在多个待审批 action；请先恢复到唯一待审批状态。")
                invocation_id = next(iter(pending))
                stored_proposal = pending[invocation_id].get("agent_proposal") or {}
                if (self.execution_mode == "interactive" and _is_model_failure_proposal(stored_proposal)
                        and str(user_message).strip().lower() in {"继续", "下一步", "continue", "next"}):
                    refreshed = self._run(invocation_id, None, user_message)
                    return format_workflow_reply(refreshed, self.state_path)
                if self.execution_mode == "interactive" and _is_pending_relax_input_proposal(stored_proposal):
                    # This is also the normal path after the chat has already
                    # selected an existing run. Refresh the pending action
                    # instead of echoing its stale Relax proposal forever.
                    refreshed = self._run(
                        invocation_id,
                        {"decision": "comment", "comment": "继续"},
                        user_message,
                    )
                    return format_workflow_reply(refreshed, self.state_path)
                if self.execution_mode == "interactive" and _unsafe_debug_calculation(stored_proposal, state):
                    revised = self._run(invocation_id, {"decision": "comment",
                        "comment": "调试模式先准备已有结构的 Relax 输入文件"}, user_message)
                    return format_workflow_reply(revised, self.state_path)
                decision = classify_user_decision(user_message)
                if decision in {"approve", "reject"}:
                    if decision == "approve" and _is_sensitive_proposal(stored_proposal):
                        return (
                            "该建议属于敏感操作。请在本机审批页确认具体影响后批准："
                            "http://127.0.0.1:8765/phase/approval"
                        )
                    from execution_layer.policy.file_approval import proposal_hash
                    state_version = build_status_summary(
                        state, config_version=state.get("confirmed_config_version")
                    )["summary_id"]
                    outcome = self.review_pending(
                        invocation_id,
                        decision,
                        expected_state_version=state_version,
                        expected_proposal_hash=proposal_hash(stored_proposal),
                        comment=user_message,
                    )
                    return format_workflow_reply(outcome.get("result") or {}, self.state_path)
                if _is_sensitive_confirmation(user_message):
                    return ("聊天消息不能批准或拒绝动作。请打开本地审批页核对计划、路径、版本和影响："
                            "http://127.0.0.1:8765/phase/approval")
                feedback = {"decision": decision, "comment": user_message}
            else:
                from decision_layer.agent.resolve_explicit_generation_request import _requested_branch_batch_size
                requested_batch = _requested_branch_batch_size(user_message)
                if requested_batch is not None and _mentions_explicit_branch_generation(user_message):
                    confirmed = state.get("confirmed_config") or {}
                    run = confirmed.get("run") or {}
                    strategy = confirmed.get("round_strategy") or {}
                    limit = max(int(run.get("total_quota", 0)),
                                int(strategy.get("generation_quota_total", 0)))
                    if requested_batch > limit:
                        return (f"要求入选 {requested_batch} 个 branch，但当前已确认的生成配额上限是 {limit}。"
                                "请先修订并确认配置；本轮没有创建或执行建议。")
                invocation_id, feedback = f"webui-{uuid.uuid4().hex}", None
            self.conversation_id = conversation_id
            result = self._run(invocation_id, feedback, user_message)
            return format_workflow_reply(result, self.state_path)

    def review_pending(self, plan_id, decision, *, expected_state_version,
                       expected_proposal_hash, comment=""):
        from run.chat_review import review_pending
        return review_pending(
            self, plan_id, decision, expected_state_version=expected_state_version,
            expected_proposal_hash=expected_proposal_hash, comment=comment,
            request_error=OpenWebUIRequestError, is_sensitive=_is_sensitive_proposal)

    def _run(self, invocation_id, human_feedback, user_message, *, approve_config_migration=False,
             dft_recovery_decision=None):
        from run.chat_execution import run_workflow_turn
        return run_workflow_turn(
            self, invocation_id, human_feedback, user_message,
            approve_config_migration=approve_config_migration,
            dft_recovery_decision=dft_recovery_decision,
            explicit_branch_request=_mentions_explicit_branch_generation(user_message))



def classify_user_decision(message: str) -> str:
    """Only a final, exact user line can approve or reject; all else is feedback."""
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    final = lines[-1].lower() if lines else ""
    if final in {"approve", "同意"}:
        return "approve"
    if final in {"reject", "拒绝"}:
        return "reject"
    return "comment"


def _is_config_migration_approval(message: str) -> bool:
    """Recognize a direct migration approval without treating questions as consent."""
    lines = [line.strip().lower() for line in str(message or "").splitlines() if line.strip()]
    final = lines[-1] if lines else ""
    for mark in ("。", "！", "!", ".", "；", ";"):
        final = final.rstrip(mark).strip()
    if any(term in final for term in ("不批准", "不允许", "不要迁移", "先别迁移", "拒绝迁移")):
        return False
    return final in {"批准迁移", "同意迁移", "确认迁移", "允许迁移", "approve migration"} or any(
        final.endswith(term) for term in ("批准迁移", "同意迁移", "确认迁移", "允许迁移")
    )


def _is_pending_relax_input_proposal(proposal):
    action = (proposal or {}).get("raw_action") or {}
    parameters = action.get("parameters") or {}
    return (action.get("tool") == "prepare_local_batch_files"
            and parameters.get("mode") == "relax_inputs")


def _is_pending_mc_proposal(proposal, state):
    action = (proposal or {}).get("raw_action") or {}
    if action.get("tool") != "allocate_mc_bohb":
        return False
    return any(row.get("stage") == "deep_search" for row in state.get("tasks") or [])


def _unsafe_debug_calculation(proposal, state):
    action = (proposal or {}).get("raw_action") or {}
    parameters = action.get("parameters") or {}
    stage = action.get("stage") or parameters.get("stage")
    return action.get("tool") == "run_calculation_stage" and stage in {"relax_screen", "relax_and_feature"}


def _workspace_path_clarification_reply(message: str, state: dict) -> str | None:
    """Answer a workspace correction without turning it into a tool action."""
    import re
    from run.configuration_chat import normalize_workspace_path

    text = str(message or "")
    if not any(label in text for label in (
        "地址是", "输出目录", "保存目录", "工作区路径", "工作区根路径", "工作区根目录"
    )):
        return None
    match = re.search(r"[A-Za-z]:[\\/][^\s，。；;]+", text)
    if not match:
        return None
    try:
        requested = normalize_workspace_path(match.group(), base_directory=Path.cwd())
    except (OSError, TypeError, ValueError):
        return "没有识别到合法的工作区路径；本轮建议未修改，也未执行动作。"
    configured = ((state.get("confirmed_config") or {}).get("storage") or {}).get("workspace_root")
    if not configured:
        return (f"你指定的输出目录是 `{requested}`。当前运行状态没有可核实的工作区配置；"
                "本轮不会生成 action 或修改路径。请检查已确认配置快照。")
    try:
        current = normalize_workspace_path(configured, base_directory=Path.cwd())
    except (OSError, TypeError, ValueError):
        return (f"你指定的输出目录是 `{requested}`，但已确认快照中的工作区路径混入了无效内容。"
                "不能用搜索 action 修复运行台账；请先备份现有运行目录并恢复配置会话，"
                "或在正确目录新建独立项目。本轮没有执行动作。")
    if requested == current:
        return (f"已核对：当前快照的工作区就是 `{current}`。"
                "若聊天中显示了别的地址，那是建议文字错误；本轮建议未修改。")
    return (f"你指定的输出目录是 `{requested}`；当前已确认快照记录的是 `{current}`。"
            "目录属于硬配置，不能由搜索 action 写入或迁移现有结果。"
            "请在正确目录新建项目，或先备份并迁移旧运行数据，再确认新配置版本；"
            "本轮未生成 action，也未执行计算。")


def _mentions_explicit_branch_generation(message: str) -> bool:
    text = "".join(str(message or "").lower().split())
    return (any(token in text for token in ("branch", "分支"))
            and any(token in text for token in ("生成", "补充", "新增", "扩展", "generate", "regenerate", "create"))
            and not any(token in text for token in ("不要生成", "不生成", "暂不生成", "先不生成", "取消生成", "别生成")))


def _drop_finished_pending(state):
    """Recover approvals left behind by a handler returning an older state copy."""
    pending = state.get("pending_execution_policies") or {}
    if not pending:
        return state, False
    finished = {"completed", "failed", "rejected", "rejected_by_user", "not_configured",
                "paused", "cancelled", "budget_exhausted"}
    records = {row.get("record_id"): row for row in state.get("action_records") or []}
    invocations = state.get("invocations") or {}
    obsolete = [key for key, item in pending.items()
                if (invocations.get(key) or records.get(item.get("record_id")) or {}).get("status") in finished]
    if not obsolete:
        return state, False
    updated = deepcopy(state)
    for key in obsolete:
        updated["pending_execution_policies"].pop(key, None)
    return updated, True


def _is_sensitive_confirmation(message):
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    return bool(lines and lines[-1] in {"确认敏感操作", "confirm sensitive action"})


def format_status_reply(state: dict) -> str:
    from run.status_presentation import format_progress
    return format_progress(state)


def format_history_prompt(state: dict, manager=None) -> str:
    summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
    ledger = getattr(manager, "data", {}) or {}
    actions = state.get("action_records") or state.get("decisions") or []
    latest = actions[-1] if actions else None
    compact = {
        "branches": len(ledger.get("branches") or {}),
        "structures": len(ledger.get("structures") or {}),
        "actions": len(actions),
        "latest_action": ({key: latest.get(key) for key in ("status", "record_id", "final_action")
                           if latest.get(key) is not None} if isinstance(latest, dict) else None),
        "tasks": len(state.get("tasks") or state.get("pending_tasks") or []),
        "pending_approvals": len(state.get("pending_execution_policies") or {}),
        "run_status": state.get("run_status") or state.get("status"),
        "config_version": summary.get("config_version"),
    }
    return ("检测到已配置的本地历史：\n```json\n" +
            json.dumps(compact, ensure_ascii=False, indent=2) +
            "\n```\n请明确回复 `继续` 或 `新建`。确认前不会提出或执行 action；`继续` 也不会批准待审批 action。")


def _latest_user_message(messages):
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text = "\n".join(item.get("text", "") for item in content
                              if isinstance(item, dict) and item.get("type") in {"text", "input_text"})
            if text.strip():
                return text.strip()
    raise OpenWebUIRequestError("messages must contain a non-empty user message")


def _open_webui_metadata_reply(message):
    """Answer Open WebUI helper requests without entering the search workflow."""
    text = str(message or "").lstrip()
    if not text.startswith("### Task:"):
        return None
    lowered = text.lower()
    if "suggest 3-5 relevant follow-up" in lowered or '"follow_ups"' in lowered:
        return '{"follow_ups": []}'
    if "broad tags categorizing" in lowered or '"tags"' in lowered:
        return '{"tags": ["材料相图搜索"]}'
    if "generate a concise" in lowered and "title" in lowered:
        return "相图搜索"
    return None


def _display(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _is_status_command(message):
    value = " ".join(str(message).strip().lower().split()).rstrip("？?。.!！")
    return value in {
        "/status", "/状态", "status", "progress", "状态", "进度", "当前状态",
        "当前进度", "查看状态", "查看进度", "查看预算", "查看相图",
        "现在什么进度", "现在是什么进度", "目前什么进度", "目前是什么进度",
        "现在进度如何", "当前进度如何", "现在到哪一步了", "做到哪一步了",
        "现在处于什么状态", "现在是什么状态", "目前处于什么状态", "现在什么状态",
        "当前是什么状态", "目前是什么状态", "现在到哪一步", "目前到哪一步了",
    }


def _phase_csv_request(message):
    value = " ".join(str(message or "").lower().split())
    export_verb = any(word in value for word in ("导出", "输出", "给我"))
    named_artifact = "相图" in value or "csv" in value
    return (export_verb and named_artifact and
            ("当前" in value or "最新" in value or "相图" in value))


def _classify_history_decision(message):
    value = " ".join(str(message).strip().lower().split())
    if value in {"继续", "continue"}:
        return "continue"
    if value in {"新建", "new", "new run"}:
        return "new"
    return None


def _is_navigation_command(message):
    """Commands that advance or inspect the saved run, never edit config."""
    value = " ".join(str(message or "").strip().lower().split())
    return value in {
        "继续", "continue", "开始", "开始搜索", "下一步", "然后呢",
        "恢复", "恢复运行", "查看状态", "查看进度", "status", "progress",
    }


def _deepseek_switch_reply(model):
    label = "DeepSeek V4.1 Flash" if model == "deepseek-flash" else "DeepSeek V4 Pro"
    return (
        f"已将本地 Agent 切换为 {label}（`{model}`），并保存到本地 Agent 运行时配置；"
        "从下一条消息起生效。搜索配置、API Key 和计算任务未修改；未调用计算后端。"
    )
