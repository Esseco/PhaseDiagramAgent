"""OpenAI-compatible local chat endpoint for Open WebUI.

The endpoint passes the actual user message to the project's interactive
workflow. It does not expose model-callable tools for approving actions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import importlib
import json
import os
from pathlib import Path
import secrets
import threading
import uuid

from copy import deepcopy

from execution_layer.step_runner.build_status_summary import build_status_summary
from execution_layer.step_runner.file_protocol import read_json, write_json


MODEL_ID = "phase-search-agent"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_CHARS = 3_200


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
            from execution_layer.local.regenerate_mc_inputs import is_mc_regeneration_request, is_mc_regeneration_confirmation
            from execution_layer.local.identify_rerun_plan import is_rerun_plan_request
            known = (_phase_csv_request(user_message) or _is_status_command(user_message)
                     or _is_navigation_command(user_message) or is_rerun_plan_request(user_message)
                     or is_mc_regeneration_request(user_message) or is_mc_regeneration_confirmation(user_message)
                     or classify_user_decision(user_message) in {"approve", "reject"}
                     or _is_config_migration_approval(user_message))
            if not known and self.config_delegate is None:
                from decision_layer.agent.resolve_chat_intent import resolve_chat_intent
                intent = resolve_chat_intent(user_message, state,
                    agent_client=self.workflow_kwargs.get("agent_client"))
                kind = intent["intent"]
                if kind == "clarify":
                    return "你想查看结果、继续下一步，还是重新准备某一轮的输入？请说明阶段和轮次。"
                if kind == "export_phase_csv":
                    user_message = "导出当前相图" + (" DFT" if intent.get("stage") == "dft" else "")
                elif kind == "status":
                    user_message = "查看状态"
                elif kind == "continue":
                    user_message = "继续"
                elif kind == "redo_plan":
                    # Use read-only generic planning, not the MC deletion confirmation path.
                    from execution_layer.local.identify_rerun_plan import identify_rerun_plan, format_rerun_plan
                    stage = {"mc": "MC", "relax": "Relax", "dft": "DFT", "branch": "branch"}[intent["stage"]]
                    return format_rerun_plan(identify_rerun_plan(f"重新准备当前轮{stage}", state))
            if _phase_csv_request(user_message):
                from analysis_layer.phase.export_current_phase_diagram import export_current_phase_diagram
                method = "dft" if "dft" in user_message.lower() else "mlip"
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
                self.config_intent_client = client
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
                    return format_workflow_reply(
                        {"status": "awaiting_approval", "agent_proposal": proposal,
                         "config_version": state.get("confirmed_config_version")}, self.state_path
                    )
                from execution_layer.remote.summarize_manual_upload_wait import (
                    summarize_manual_upload_wait,
                )
                if summarize_manual_upload_wait(state):
                    invocation_id = f"webui-{uuid.uuid4().hex}"
                    result = self._run(invocation_id, None, user_message)
                    return format_workflow_reply(result, self.state_path)
                return "已继续原运行并加载 state/ledger。请发送下一条搜索指令；本次确认不会批准任何 action。"
            pending = state.get("pending_execution_policies") or {}
            if _is_status_command(user_message):
                return format_status_reply(state)
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
        """Accept a decision only from the authenticated local approval surface."""
        from execution_layer.policy.file_approval import proposal_hash
        with self.lock:
            state = read_json(self.state_path, {}) or {}
            completed = (state.get("invocations") or {}).get(plan_id)
            pending = (state.get("pending_execution_policies") or {}).get(plan_id)
            if pending is None and completed is not None:
                return {"status": "already_processed", "result": completed}
            if pending is None:
                raise OpenWebUIRequestError("plan is not pending")
            current_version = build_status_summary(
                state, config_version=state.get("confirmed_config_version"))["summary_id"]
            if expected_state_version != current_version:
                raise OpenWebUIRequestError("state_version_stale; refresh the approval page")
            proposal = pending.get("agent_proposal") or {}
            if expected_proposal_hash != proposal_hash(proposal):
                raise OpenWebUIRequestError("proposal_hash_mismatch; refresh the approval page")
            if decision not in {"approve", "reject", "confirm_sensitive"}:
                raise OpenWebUIRequestError("invalid approval-page decision")
            if _is_sensitive_proposal(proposal) and decision == "approve":
                raise OpenWebUIRequestError("sensitive action requires confirm_sensitive")
            approved = decision in {"approve", "confirm_sensitive"}
            audit_comment = str(comment or "").strip()
            if approved:
                audit_comment = (audit_comment + "\napprove").strip()
            feedback = {"decision": "approve" if approved else "reject", "comment": audit_comment}
            result = self._run(plan_id, feedback, "local approval page")
            return {"status": result.get("status"), "result": result}

    def _run(self, invocation_id, human_feedback, user_message, *, approve_config_migration=False):
        workflow = self.workflow
        if workflow is None:
            from run.main import run_workflow
            workflow = run_workflow
        kwargs = {
            key: value for key, value in self.workflow_kwargs.items()
            if key not in {"state", "execution_mode", "human_feedback", "replay_record",
                           "max_steps", "invocation_id", "state_path", "config_session_path"}
        }
        kwargs["user_message"] = user_message
        if approve_config_migration:
            # This approval only authorizes a safe, versioned config migration;
            # it never approves a scientific action or remote submission.
            kwargs["approve_budget_extension"] = True
        base_agent_client = kwargs.get("agent_client")
        from decision_layer.agent.resolve_explicit_generation_request import (
            resolve_explicit_generation_request,
        )
        from execution_layer.local.rebuild_relax_inputs import is_relax_rebuild_request, plan_relax_rebuild
        if callable(base_agent_client) or _mentions_explicit_branch_generation(user_message) or is_relax_rebuild_request(user_message):
            def user_contextualized_agent(payload):
                if is_relax_rebuild_request(user_message):
                    plan = plan_relax_rebuild(read_json(self.state_path, {}) or {},
                        kwargs["run_config"]["upload_batches_directory"])
                    return {"tool": "prepare_local_batch_files", "task_key": f"rebuild-relax:{invocation_id}",
                            "target_ids": [], "budget": 0.0,
                            "parameters": {"mode": "relax_inputs", "rebuild_inputs": True, "cleanup_plan": plan},
                            "reason": f"请确认旧批次尚未在超算提交；批准后删除 {len(plan['directories'])} 个旧输入批次并重建，保留原结构、任务编号和预算。",
                            "expected_purpose": "重建Relax输入，每个作业最多100个结构；不提交计算。"}
                direct_action = resolve_explicit_generation_request(
                    user_message,
                    allowed_tools=payload.get("allowed_tools") or [],
                    state=payload.get("state") or {},
                )
                if direct_action is not None:
                    direct_action["task_key"] = f"generate_branches:user-request:{invocation_id}"
                    return direct_action
                if not callable(base_agent_client):
                    raise ValueError("DeepSeek Agent 未配置")
                return base_agent_client({**payload, "user_instruction": user_message})
            kwargs["agent_client"] = user_contextualized_agent
        return workflow(
            **kwargs, execution_mode=self.execution_mode, human_feedback=human_feedback,
            max_steps=1, invocation_id=invocation_id, state_path=str(self.state_path),
        )


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


from run.chat_approval_rules import is_sensitive_proposal as _is_sensitive_proposal


from run.chat_state_presentation import brief_chat_state  # Compatible public import.


def handle_chat_request(payload: dict, chat_handler, *, model_id=MODEL_ID) -> dict:
    if not isinstance(payload, dict):
        raise OpenWebUIRequestError("request body must be a JSON object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise OpenWebUIRequestError("messages must be a non-empty list")
    _latest_user_message(messages)
    metadata = payload.get("metadata") or {}
    conversation_id = str(metadata.get("chat_id") or payload.get("user") or "openwebui")[:128]
    content = _chat_content(chat_handler(messages, conversation_id=conversation_id))
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
        "created": int(datetime.now(timezone.utc).timestamp()), "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
    }


def _chat_content(value):
    """Never dump full scientific state or candidate arrays into the chat pane."""
    if isinstance(value, dict):
        fields = ("status", "tool", "action", "reason", "record_id")
        parts = [f"{key}: {str(value[key])[:180]}" for key in fields
                 if value.get(key) is not None and not isinstance(value[key], (dict, list))]
        return "结果摘要：" + ("；".join(parts) if parts else "详细数据请查看项目记录。")
    content = value if isinstance(value, str) else str(value)
    if "入选结构（Na/O₂；相；Ehull eV/atom）" in content and len(content) <= 16000:
        return content  # Bounded, intentional list of at most 100 approved candidates.
    if len(content) <= MAX_RESPONSE_CHARS:
        return content
    stripped = content.lstrip()
    if stripped.startswith(("{", "[", "```json")) or '"composition"' in content:
        return "本轮回复含大量内部候选明细，已从聊天窗口省略；请查看项目状态或审批文件。"
    return content[:MAX_RESPONSE_CHARS].rsplit("\n", 1)[0] + "\n…（详细内容请查看项目记录）"


def create_server(chat_handler, *, api_key: str, host="127.0.0.1", port=8765, model_id=MODEL_ID,
                  local_control=None, control_api_key=None, deepseek_key_setup=None,
                  deepseek_model="deepseek-flash"):
    """Create a dependency-free OpenAI-compatible server, bound locally by default."""
    if not callable(chat_handler):
        raise TypeError("chat_handler must be callable")
    if not isinstance(api_key, str) or len(api_key) < 16:
        raise ValueError("local API bearer key must contain at least 16 characters")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok", "config_protocol": 2})
            elif self.path == "/phase/setup":
                if not self._is_loopback_client():
                    self._json(403, _error("forbidden", "key setup is local-only")); return
                from run.deepseek_credentials import load_deepseek_api_key
                self._html(200, _deepseek_setup_page(
                    deepseek_model,
                    configured=bool(load_deepseek_api_key()),
                ))
            elif self.path == "/phase/chat":
                if not self._is_loopback_client():
                    self._json(403, _error("forbidden", "local chat is local-only")); return
                from run.local_chat_page import local_chat_page
                self._html(200, local_chat_page())
            elif self.path == "/v1/models" and self._authorized():
                self._json(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "created": 0, "owned_by": "local-project"}
                ]})
            elif self.path == "/v1/models":
                self._json(401, _error("unauthorized", "invalid local API key"))
            elif self.path == "/phase/approval":
                self._html(200, _approval_page())
            elif self.path in {"/phase/status", "/phase/pending", "/phase/tasks", "/phase/charts", "/phase/config", "/phase/memory", "/phase/memory/skills"} and self._control_authorized() and local_control is not None:
                name = ("domain_skill_matches" if self.path == "/phase/memory/skills"
                        else self.path.rsplit("/", 1)[-1])
                try:
                    self._json(200, getattr(local_control, name)())
                except (ValueError, KeyError) as error:
                    self._json(400, _error("invalid_request", str(error)))
            elif self.path.startswith("/phase/"):
                self._json(401 if not self._control_authorized() else 404,
                           _error("unauthorized" if not self._control_authorized() else "not_found",
                                  "invalid control API key" if not self._control_authorized() else "endpoint not found"))
            else:
                self._json(404, _error("not_found", "endpoint not found"))

        def do_POST(self):
            if self.path == "/phase/setup/key":
                if not self._is_loopback_client():
                    self._json(403, _error("forbidden", "key setup is local-only")); return
                if not callable(deepseek_key_setup):
                    self._json(503, _error("not_configured", "本地 DeepSeek 设置入口未配置")); return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 8192:
                        raise OpenWebUIRequestError("请求内容无效")
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    key = body.get("api_key") if isinstance(body, dict) else None
                    if not isinstance(key, str) or len(key.strip()) < 16:
                        raise OpenWebUIRequestError("请粘贴完整的 DeepSeek API Key")
                    response = deepseek_key_setup(key.strip())
                    self._json(200, response)
                except (UnicodeDecodeError, json.JSONDecodeError, OpenWebUIRequestError, ValueError) as error:
                    self._json(400, _error("invalid_request", str(error)))
                except Exception as error:
                    safe_message = getattr(error, "safe_message", None)
                    message = safe_message or "连接或本地安全保存失败；密钥未在网页中回显。"
                    self._json(502, _error("setup_failed", message))
                return
            if self.path in {"/phase/propose", "/phase/decision", "/phase/pause", "/phase/config/patch", "/phase/config/confirm", "/phase/memory/review", "/phase/memory/propose", "/phase/memory/skills/import", "/phase/memory/skills/publish"}:
                if not self._control_authorized():
                    self._json(401, _error("unauthorized", "invalid control API key")); return
                if local_control is None:
                    self._json(404, _error("not_found", "local control is disabled")); return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > MAX_REQUEST_BYTES: raise OpenWebUIRequestError("invalid body size")
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    conversation = str(body.get("conversation_id") or "local-control")[:128]
                    if self.path == "/phase/propose":
                        response = local_control.propose(body.get("instruction", ""), conversation_id=conversation)
                    elif self.path == "/phase/decision":
                        response = local_control.decide(
                            body.get("decision"), plan_id=body.get("plan_id"),
                            expected_state_version=body.get("state_version"),
                            expected_proposal_hash=body.get("proposal_hash"),
                            comment=body.get("comment", ""), conversation_id=conversation)
                    elif self.path == "/phase/config/patch":
                        response = local_control.patch_config(body.get("patch"), reasons=body.get("reasons"), impacts=body.get("impacts"))
                    elif self.path == "/phase/config/confirm":
                        response = local_control.confirm_config(explicit=body.get("explicit") is True)
                    elif self.path == "/phase/memory/review":
                        response = local_control.review_memory(body.get("proposal_id"), approved=body.get("approved") is True)
                    elif self.path == "/phase/memory/propose":
                        response = local_control.propose_memory(body.get("record"))
                    elif self.path == "/phase/memory/skills/import":
                        response = local_control.propose_skill_import()
                    elif self.path == "/phase/memory/skills/publish":
                        response = local_control.publish_skill(body.get("draft_directory"),
                            approved=body.get("approved") is True,
                            version=body.get("version", "1.0.0"))
                    else:
                        response = local_control.pause(body.get("reason", ""), conversation_id=conversation)
                    self._json(200, response)
                except (ValueError, KeyError, OpenWebUIRequestError) as error:
                    self._json(400, _error("invalid_request", str(error)))
                return
            if self.path != "/v1/chat/completions":
                self._json(404, _error("not_found", "endpoint not found")); return
            if not self._authorized():
                self._json(401, _error("unauthorized", "invalid local API key")); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise OpenWebUIRequestError("request body is empty or too large")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = handle_chat_request(payload, chat_handler, model_id=model_id)
            except (UnicodeDecodeError, json.JSONDecodeError, OpenWebUIRequestError) as error:
                self._json(400, _error("invalid_request", str(error))); return
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.log_error("local workflow failed (%s)", type(error).__name__)
                self._json(500, _error("workflow_error", f"local Agent workflow failed ({type(error).__name__}); inspect server stderr")); return
            if payload.get("stream"):
                self._stream(response)
            else:
                self._json(200, response)

        def _authorized(self):
            value = self.headers.get("Authorization", "")
            return value.startswith("Bearer ") and secrets.compare_digest(value[7:], api_key)

        def _control_authorized(self):
            value = self.headers.get("Authorization", "")
            return (isinstance(control_api_key, str) and len(control_api_key) >= 16 and
                    value.startswith("Bearer ") and secrets.compare_digest(value[7:], control_api_key))

        def _is_loopback_client(self):
            try:
                return ipaddress.ip_address(self.client_address[0]).is_loopback
            except (ValueError, IndexError):
                return False

        def _json(self, status, value):
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def _html(self, status, value):
            body = value.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'")
            self.end_headers(); self.wfile.write(body)

        def _stream(self, response):
            base = {key: response[key] for key in ("id", "created", "model")}
            content = response["choices"][0]["message"]["content"]
            chunks = [
                {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]},
                {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
            body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers(); self.wfile.write(encoded)

    return ThreadingHTTPServer((host, int(port)), Handler)


def serve_open_webui(chat_handler, *, api_key, host="127.0.0.1", port=8765, model_id=MODEL_ID,
                     local_control=None, control_api_key=None, deepseek_key_setup=None,
                     deepseek_model="deepseek-flash", parent_pid=None):
    server = create_server(chat_handler, api_key=api_key, host=host, port=port, model_id=model_id,
                           local_control=local_control, control_api_key=control_api_key,
                           deepseek_key_setup=deepseek_key_setup, deepseek_model=deepseek_model)
    if parent_pid is not None:
        threading.Thread(
            target=_stop_server_when_parent_exits,
            args=(server, int(parent_pid)), daemon=True,
        ).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _stop_server_when_parent_exits(server, parent_pid):
    """Bind a launcher-owned Agent to that launcher's lifetime."""
    if os.name == "nt":
        import ctypes
        synchronize = 0x00100000
        infinite = 0xFFFFFFFF
        handle = ctypes.windll.kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            server.shutdown()
            return
        try:
            ctypes.windll.kernel32.WaitForSingleObject(handle, infinite)
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
        server.shutdown()
        return
    while True:
        try:
            os.kill(parent_pid, 0)
        except OSError:
            server.shutdown()
            return
        threading.Event().wait(1.0)


from run.workflow_reply_presentation import (
    format_workflow_reply,
    _format_workflow_reply_verbose,
    _is_model_failure_proposal,
    _friendly_validation_errors,
    _friendly_workflow_rejection,
    _proposal_directory,
)  # Keep existing imports compatible while presentation owns these functions.


def format_status_reply(state: dict) -> str:
    summary = build_status_summary(state, config_version=state.get("confirmed_config_version"))
    return "当前项目状态：\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n```"


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


def _approval_page():
    return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>相图项目本地审批</title>
<style>body{font-family:system-ui;margin:2rem;max-width:1050px;color:#18212b}input,textarea,button,select{font:inherit;padding:.55rem;margin:.25rem}input{min-width:28rem}.card{border:1px solid #ccd4dd;border-radius:8px;padding:1rem;margin:1rem 0;background:#fff}pre{white-space:pre-wrap;background:#f5f7f9;padding:.8rem;overflow:auto}.charts svg{max-width:100%;height:auto;border:1px solid #eee;margin:.4rem 0}.warn{color:#a33}</style></head><body>
<h1>相图项目本地审批</h1><p>此页面只连接本机受限接口。聊天中的“同意/继续”不会执行动作。</p>
<label>本地控制令牌 <input id="token" type="password" autocomplete="off"></label><button onclick="loadAll()">读取</button>
<p id="message" class="warn"></p><section id="plans"></section><h2>项目生成图表</h2><section id="charts" class="charts"></section>
<script>
const el=id=>document.getElementById(id); const auth=()=>({'Authorization':'Bearer '+el('token').value,'Content-Type':'application/json'});
async function api(path, options={}){const r=await fetch(path,{...options,headers:auth()});const j=await r.json();if(!r.ok)throw Error(j.error?.message||r.status);return j}
function pretty(x){return JSON.stringify(x,null,2)}
async function loadAll(){try{el('message').textContent='';const p=await api('/phase/pending');const c=await api('/phase/charts');renderPlans(p);el('charts').innerHTML='';Object.values(c.charts).forEach(x=>{const d=document.createElement('div');d.className='card';d.innerHTML=x.svg;const v=document.createElement('code');v.textContent='数据版本 '+x.data_version;d.appendChild(v);el('charts').appendChild(d)})}catch(e){el('message').textContent=e.message}}
function renderPlans(value){el('plans').innerHTML='<h2>待批计划（'+value.count+'）</h2>';value.pending.forEach(p=>{const d=document.createElement('div');d.className='card';const pre=document.createElement('pre');pre.textContent=pretty(p);d.appendChild(pre);const note=document.createElement('textarea');note.placeholder='审核意见（可选）';d.appendChild(note);['approve','reject','confirm_sensitive'].forEach(decision=>{const b=document.createElement('button');b.textContent={approve:'批准',reject:'拒绝',confirm_sensitive:'确认敏感操作'}[decision];b.onclick=()=>decide(p,decision,note.value);d.appendChild(b)});el('plans').appendChild(d)})}
async function decide(p,decision,comment){try{const result=await api('/phase/decision',{method:'POST',body:JSON.stringify({plan_id:p.plan_id,decision,state_version:p.state_version,proposal_hash:p.proposal_hash,comment})});el('message').textContent='处理结果：'+result.status;await loadAll()}catch(e){el('message').textContent=e.message}}
</script></body></html>"""


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
    value = " ".join(str(message).strip().lower().split())
    return value in {
        "/status", "/状态", "status", "progress", "状态", "进度", "当前状态",
        "当前进度", "查看状态", "查看进度", "查看预算", "查看相图",
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


def _has_history(state, manager) -> bool:
    ledger = getattr(manager, "data", {}) or {}
    if ledger.get("branches") or ledger.get("structures"):
        return True
    keys = ("action_records", "decisions", "tasks", "pending_tasks", "slurm_batches",
            "event_history", "pending_execution_policies")
    return any(state.get(key) for key in keys)


def _resolve_path(value, base):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _load_json_object(path, label):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"{label}不存在：{path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label}必须是 JSON object：{path}")
    return value


def _import_reference(reference, label):
    module_name, separator, name = str(reference or "").partition(":")
    if not separator:
        raise ValueError(f"{label}必须使用 package.module:function")
    value = getattr(importlib.import_module(module_name), name)
    return value() if callable(value) else value


def _reject_secrets(config):
    forbidden = {"api_key", "token", "password", "secret", "private_key"}
    found = []

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                field = str(key).lower()
                path = f"{prefix}.{key}" if prefix else str(key)
                if field in forbidden or field.endswith("_password") or field.endswith("_secret"):
                    found.append(path)
                walk(child, path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")

    walk(config)
    if found:
        raise ValueError("运行时 JSON 不得包含密钥字段：" + ", ".join(found))


def _session_from_state(state):
    config = state.get("confirmed_config")
    version = state.get("confirmed_config_version") or state.get("config_version")
    if not isinstance(config, dict) or not version:
        return None
    return {
        "status": "confirmed", "config": deepcopy(config), "dialogue": [],
        "confirmed_snapshot": {
            "config_version": str(version), "config_hash": state.get("confirmed_config_hash"),
            "config": deepcopy(config), "audit": {"valid": True, "errors": []},
        },
    }


def create_open_webui_runtime(config_path=None):
    """Compose the built-in runtime from a local, non-secret JSON file."""
    configured = config_path or os.environ.get("PHASE_SEARCH_OPENWEBUI_CONFIG") or "run/open_webui_runtime.json"
    config_file = Path(configured).resolve()
    settings = _load_json_object(config_file, "Open WebUI 运行时配置")
    _reject_secrets(settings)
    from config_layer.session.load_config_session import load_config_session
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.deepseek_credentials import load_deepseek_api_key
    base = config_file.parent
    missing = [key for key in ("state_path", "ledger_path") if not settings.get(key)]
    if missing:
        raise ValueError(f"运行时配置缺少 {missing}；请编辑 {config_file}")

    state_path = _resolve_path(settings["state_path"], base)
    ledger_path = _resolve_path(settings["ledger_path"], base)
    phase_path = (_resolve_path(settings["phase_references_path"], base)
                  if settings.get("phase_references_path") else None)
    state = _load_json_object(state_path, "state") if state_path.is_file() else {}

    resolved_session = _resolve_path(
        settings.get("config_session_path", "local/config_session.json"), base
    )
    from config_layer.session.resolve_workspace_paths import default_workspace_storage
    path_keys = {
        "state": ("state_path", True),
        "ledger": ("ledger_path", True),
        "branch_energy_pool_ledger": ("branch_energy_pool_ledger_path", True),
        "phase_diagrams": ("phase_diagram_directory", False),
        "approvals": ("approval_directory", False),
        "approved_batches": ("local_action_directory", False),
        "upload_batches": ("manual_upload.batches_directory", False),
        "new_runs": ("new_runs_directory", False),
    }
    configured_paths = {}
    workspace_candidates = [resolved_session.parent.resolve()]
    for key, (setting_key, is_file) in path_keys.items():
        value = settings.get(setting_key)
        if setting_key == "manual_upload.batches_directory":
            value = (settings.get("manual_upload") or {}).get("batches_directory")
        if value:
            resolved_value = _resolve_path(value, base)
            configured_paths[key] = resolved_value
            workspace_candidates.append(resolved_value.parent if is_file else resolved_value)
    roots = workspace_candidates
    try:
        workspace_root = Path(os.path.commonpath([str(path) for path in roots]))
    except ValueError:
        workspace_root = resolved_session.parent
    workspace_defaults = default_workspace_storage(
        workspace_root, path_overrides=configured_paths,
    )
    if not resolved_session.is_file():
        session = _session_from_state(state)
        if session is None:
            if state:
                raise ValueError(
                    f"{state_path} 有运行数据但没有可恢复的 confirmed config；为避免覆盖，先手动恢复配置会话"
                )
            from config_layer.defaults.default_layered_search_config import default_layered_search_config
            from config_layer.session.create_config_draft import create_config_draft
            from config_layer.session.save_config_session import save_config_session
            from run.configuration_chat import BOOTSTRAP_HINTS
            session = create_config_draft(
                default_layered_search_config(), require_workspace_path=True
            )
            session["bootstrap_hints"] = deepcopy(BOOTSTRAP_HINTS)
            save_config_session(session, resolved_session)
    else:
        session = load_config_session(resolved_session)

    if session.get("status") != "confirmed" and not session.get("setup_stage"):
        # Upgrade pre-path-first drafts without discarding their edits or files.
        session["setup_stage"] = "awaiting_storage_path"
        session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
        session.setdefault("dialogue", []).append({
            "type": "workspace_path_question",
            "role": "assistant",
            "message": (
                "请先发送本地工作区根目录路径；我会展示保存位置，"
                "并在你确认后才生成设置 JSON。"
            ),
        })
        from config_layer.session.save_config_session import save_config_session
        save_config_session(session, resolved_session)

    if (session.get("status") != "confirmed"
            and "storage" not in (session.get("config") or {})
            and not session.get("setup_stage")):
        from config_layer.session.apply_config_revision import apply_config_revision
        from config_layer.session.save_config_session import save_config_session
        session = apply_config_revision(
            session, {"storage": workspace_defaults},
            reasons={"storage": "沿用当前本地配置会话目录作为统一工作区默认值；用户可在确认前调整。"},
            author="workspace_storage_default",
        )
        save_config_session(session, resolved_session)

    if session.get("status") != "confirmed":
        from run.configuration_chat import (
            BOOTSTRAP_HINTS, CONFIG_AGENT_SYSTEM_PROMPT, ConfigurationChatHandler,
            normalize_workspace_path, safe_config_filename,
        )
        editable_path_setting = settings.get("editable_config_draft_path") or "search_config.project.json"
        editable_config_filename = safe_config_filename(Path(editable_path_setting).name)
        setup_stage = session.get("setup_stage")
        if setup_stage == "json_ready":
            storage = (session.get("config") or {}).get("storage") or {}
            try:
                workspace_root = normalize_workspace_path(
                    storage.get("workspace_root"), base_directory=base,
                )
                saved_path = session.get("editable_config_json_path")
                preferred_path = workspace_root / editable_config_filename
                if preferred_path.is_file():
                    draft_candidate = preferred_path
                    if saved_path and str(saved_path) != str(preferred_path):
                        session["editable_config_json_path"] = str(preferred_path)
                        session.pop("agent_reviewed_revision", None)
                        session.pop("agent_reviewed_config_hash", None)
                        session.pop("agent_reviewed_config_digest", None)
                        session.pop("agent_reviewed_mother_digest", None)
                        session.setdefault("dialogue", []).append({
                            "type": "editable_config_source_changed",
                            "old_path": saved_path,
                            "new_path": str(preferred_path),
                            "reason": "运行时指定的设置文件已存在；以它作为下一次导入来源。",
                        })
                        from config_layer.session.save_config_session import save_config_session
                        save_config_session(session, resolved_session)
                elif saved_path:
                    draft_candidate = normalize_workspace_path(saved_path, base_directory=base)
                else:
                    draft_candidate = preferred_path
                if (not draft_candidate.is_relative_to(workspace_root)
                        or draft_candidate.suffix.lower() != ".json"):
                    raise ValueError("设置 JSON 必须位于已确认工作区内")
            except (TypeError, ValueError, OSError):
                # Recover sessions written by older versions that accepted arbitrary
                # multi-line input as a path. Keep unrelated draft fields and files.
                session.pop("editable_config_json_path", None)
                session.pop("pending_workspace_root", None)
                session["setup_stage"] = "awaiting_storage_path"
                session.setdefault("config", {}).pop("storage", None)
                session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
                session.setdefault("dialogue", []).append({
                    "type": "workspace_path_recovery",
                    "role": "assistant",
                    "message": (
                        "检测到上次保存的工作区路径格式无效，已清除该路径并恢复到路径选择。"
                        "没有删除或覆盖任何文件；请重新发送一行本地工作区目录路径。"
                    ),
                })
                from config_layer.session.save_config_session import save_config_session
                save_config_session(session, resolved_session)
                setup_stage = "awaiting_storage_path"

        elif setup_stage == "awaiting_storage_confirmation":
            pending_root = session.get("pending_workspace_root")
            try:
                if not pending_root:
                    raise ValueError("缺少待确认工作区")
                normalize_workspace_path(pending_root, base_directory=base)
            except (TypeError, ValueError, OSError):
                session.pop("pending_workspace_root", None)
                session.pop("editable_config_json_path", None)
                session["setup_stage"] = "awaiting_storage_path"
                from config_layer.session.save_config_session import save_config_session
                save_config_session(session, resolved_session)
                setup_stage = "awaiting_storage_path"

        saved_draft_path = session.get("editable_config_json_path")
        if setup_stage in {"awaiting_storage_path", "awaiting_storage_confirmation"}:
            editable_config_path = None
        elif setup_stage == "json_ready":
            editable_config_path = draft_candidate
        elif saved_draft_path:
            try:
                editable_config_path = normalize_workspace_path(saved_draft_path, base_directory=base)
            except (TypeError, ValueError, OSError):
                editable_config_path = None
        elif not setup_stage:
            if "storage" in (session.get("config") or {}) and settings.get("editable_config_draft_path"):
                editable_config_path = _resolve_path(settings["editable_config_draft_path"], base)
            else:
                session_setting = Path(settings.get("config_session_path", "local/config_session.json"))
                editable_config_path = _resolve_path(
                    session_setting.with_name(editable_config_filename), base
                )
        else:
            editable_config_path = None
        if session.get("status") == "draft" and editable_config_path is not None:
            from config_layer.session.create_editable_config_json import create_editable_config_json
            from config_layer.session.project_config_json import create_project_config_json
            # Create the template once. A user-edited file is never overwritten on restart.
            try:
                creator = create_project_config_json if editable_config_path.name.endswith(".project.json") else create_editable_config_json
                creator(
                    editable_config_path, session.get("config") or {},
                    bootstrap_hints=session.get("bootstrap_hints") or BOOTSTRAP_HINTS,
                    workspace_defaults=workspace_defaults,
                )
            except (OSError, TypeError, ValueError):
                session.pop("editable_config_json_path", None)
                session.pop("pending_workspace_root", None)
                session["setup_stage"] = "awaiting_storage_path"
                session.setdefault("config", {}).pop("storage", None)
                session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
                session.setdefault("dialogue", []).append({
                    "type": "workspace_path_recovery",
                    "role": "assistant",
                    "message": (
                        "上次工作区不可写，已恢复到路径选择。没有删除已有文件；"
                        "请重新发送一行可写的本地工作区目录路径。"
                    ),
                })
                from config_layer.session.save_config_session import save_config_session
                save_config_session(session, resolved_session)
                editable_config_path = None
        deepseek = settings.get("deepseek") or {}
        try:
            agent_client = create_deepseek_client(
                api_key=load_deepseek_api_key(),
                model=deepseek.get("model", "deepseek-v4-pro"),
                base_url=deepseek.get("base_url", "https://api.deepseek.com"),
                max_tokens=int(deepseek.get("max_tokens", 800)),
                timeout=int(deepseek.get("timeout", 60)),
                system_prompt=CONFIG_AGENT_SYSTEM_PROMPT,
                thinking=deepseek.get("configuration_thinking", "disabled"),
            )
        except ValueError:
            agent_client = None
        model_switcher = _make_deepseek_model_switcher(
            config_file, deepseek, system_prompt=CONFIG_AGENT_SYSTEM_PROMPT,
            thinking=deepseek.get("configuration_thinking", "disabled"),
        )
        workflow_kwargs = {
            "state_path": str(state_path), "config_session": session,
            "config_session_path": str(resolved_session),
        }
        return ConfigurationChatHandler(
            workflow_kwargs, config_session_path=resolved_session, base_directory=base,
            phase_references_path=phase_path, editable_config_path=editable_config_path,
                editable_config_filename=editable_config_filename,
                workspace_root_default=workspace_root,
            agent_client=agent_client,
            runtime_factory=lambda: create_open_webui_runtime(config_file),
            current_deepseek_model=deepseek.get("model", "deepseek-v4-pro"),
            deepseek_model_switcher=model_switcher,
        )

    snapshot = session.get("confirmed_snapshot") or {}
    if not snapshot.get("config") or not snapshot.get("config_version"):
        raise ValueError("配置会话不是完整的 confirmed snapshot；不会自动确认配置")

    from config_layer.session.resolve_workspace_paths import resolve_workspace_paths
    effective_config = deepcopy(snapshot["config"])
    if "storage" not in effective_config:
        # Older confirmed snapshots retain their original local workspace by default.
        effective_config["storage"] = workspace_defaults
    resolved_storage = resolve_workspace_paths(effective_config, base_directory=base)
    selected_state_path = resolved_storage["state"]
    selected_ledger_path = resolved_storage["ledger"]
    for label, previous, selected in (
        ("state", state_path, selected_state_path),
        ("ledger", ledger_path, selected_ledger_path),
    ):
        if previous.resolve() != selected.resolve() and previous.exists():
            raise ValueError(
                f"工作区路径变更检测到已有 {label} 文件：{previous}。"
                f"不会自动移动或忽略它；请在配置 JSON 中将 storage.paths.{label} 指回原位置，"
                "或先由用户手动迁移并核对后再继续。"
            )
    state_path, ledger_path = selected_state_path, selected_ledger_path
    state = _load_json_object(state_path, "state") if state_path.is_file() else {}
    if state and state.get("confirmed_config_version") != snapshot.get("config_version"):
        from config_layer.runtime.authorize_generation_policy_revision import (
            authorize_generation_policy_revision,
        )
        migration = authorize_generation_policy_revision(state, snapshot)
        if migration["status"] == "rebound":
            state = migration["state"]
            write_json(state_path, state)

    configured_references = ((effective_config.get("system") or {}).get("phase_references") or {})
    if phase_path and phase_path.is_file():
        phase_references = _load_json_object(phase_path, "相图参考")
    else:
        phase_references = deepcopy(configured_references)
    if not isinstance(phase_references, dict):
        raise ValueError("相图参考必须是映射")
    phase_references = {
        key: (str(_resolve_path(value, base)) if isinstance(value, str) else value)
        for key, value in phase_references.items()
    }

    from data_layer.ledger.phase_data_manager import PhaseDataManager
    from run.default_run_config import default_run_config
    if ledger_path.is_file():
        manager = PhaseDataManager.load(ledger_path)
        saved_system = manager.data.get("system_config") or {}
        saved_space = saved_system.get("configuration_space") or {}
        active_space = (effective_config.get("system") or {}).get("configuration_space") or {}
        if saved_space != active_space:
            if manager.data.get("branches") or manager.data.get("structures"):
                raise ValueError(
                    "现有台账的问题变量角色与已确认配置不同；不能在原台账上改变 branch 编号。"
                    "请建立新运行或明确迁移旧数据。")
            manager.data["system_config"] = deepcopy(effective_config["system"])
            if (active_space.get("roles") or {}).get("T") == "fixed":
                manager.data["system_config"]["branch_schema"]["fields"] = ["P", "H", "x"]
    else:
        boundary = ((snapshot["config"].get("system") or {}).get("boundary"))
        if not isinstance(boundary, dict):
            raise ValueError(f"台账不存在且 confirmed config 没有 system.boundary；请配置 {ledger_path}")
        manager = PhaseDataManager(boundary, system_config=effective_config.get("system"))
        manager.save(ledger_path)

    runtime_config = default_run_config()
    runtime_config["system_config"] = deepcopy(effective_config["system"])
    runtime_config.update({"state_path": str(state_path), "ledger_path": str(ledger_path)})
    for key in ("structure_directory", "phase_diagram_directory", "work_directory",
                "branch_energy_pool_ledger_path", "approval_directory",
                "local_action_directory", "mlip"):
        if key in settings:
            value = deepcopy(settings[key])
            if key != "mlip" and isinstance(value, str):
                value = str(_resolve_path(value, base))
            runtime_config[key] = value
    runtime_config.update({
        "structure_directory": str(resolved_storage["structures"]),
        "state_path": str(state_path),
        "ledger_path": str(ledger_path),
        "branch_energy_pool_ledger_path": str(resolved_storage["branch_energy_pool_ledger"]),
        "phase_diagram_directory": str(resolved_storage["phase_diagrams"]),
        "work_directory": str(resolved_storage["work"]),
        "approval_directory": str(resolved_storage["approvals"]),
        "local_action_directory": str(resolved_storage["approved_batches"]),
        "upload_batches_directory": str(resolved_storage["upload_batches"]),
    })
    runtime_config.setdefault("qbc", {})["output_path"] = str(resolved_storage["qbc_results"])
    deepseek = settings.get("deepseek") or {}
    from run.deepseek_credentials import load_deepseek_api_key
    try:
        agent_client = create_deepseek_client(
            api_key=load_deepseek_api_key(),
            model=deepseek.get("model", "deepseek-v4-pro"),
            base_url=deepseek.get("base_url", "https://api.deepseek.com"),
            max_tokens=int(deepseek.get("max_tokens", 800)), timeout=int(deepseek.get("timeout", 60)),
            thinking=deepseek.get("thinking"),
            routine_max_tokens=int(deepseek.get("routine_max_tokens", 1600)),
            reasoning_max_tokens=int(deepseek.get("reasoning_max_tokens", 8192)),
        )
    except ValueError:
        # Start the local UI without a key so the user can configure it in-browser.
        agent_client = None
    model_switcher = _make_deepseek_model_switcher(
        config_file, deepseek, system_prompt=None, thinking=deepseek.get("thinking"),
    )
    kwargs = {"manager": manager, "phase_references": phase_references,
              "run_config": runtime_config, "config_session": session,
              "state_path": str(state_path), "agent_client": agent_client}
    kwargs["config_session_path"] = str(resolved_session)
    for key in ("dispatcher", "task_runner"):
        reference = settings.get(f"{key}_factory")
        if reference:
            kwargs[key] = _import_reference(reference, f"{key}_factory")
    for key, reference in (settings.get("runtime_adapters") or {}).items():
        kwargs[key] = _import_reference(reference, f"runtime_adapters.{key}")
    manual = settings.get("manual_upload") or {}
    if manual.get("enabled"):
        if kwargs.get("task_runner") is not None:
            raise ValueError("manual_upload 与 task_runner_factory 不能同时配置")
        required = [key for key in ("batches_directory", "worker_command") if not manual.get(key)]
        if required:
            raise ValueError(f"manual_upload 缺少 {required}；不会猜测超算路径或命令")
        if manual.get("submit"):
            raise ValueError("manual_upload 只允许本地生成，submit 必须为 false")
        command = manual["worker_command"]
        if not isinstance(command, list) or not all(isinstance(item, str) and item for item in command):
            raise ValueError("manual_upload.worker_command 必须是非空字符串数组")
        from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
        task_preparer = None
        if manual.get("task_preparer_factory"):
            task_preparer = _import_reference(manual["task_preparer_factory"], "manual_upload.task_preparer_factory")
        kwargs["task_runner"] = ManualUploadBatchRunner(
            resolved_storage["upload_batches"], worker_command=command,
            dispatcher=kwargs.get("dispatcher"),
            stage_batch_sizes=manual.get("stage_batch_sizes"),
            stage_profiles=manual.get("stage_profiles"), task_preparer=task_preparer,
        )
    if kwargs.get("task_runner") is None and kwargs.get("result_collector") is None:
        # Read-only recovery for manually uploaded jobs; this never submits/prepares tasks.
        from execution_layer.remote.batch_runner import RemoteBatchRunner
        kwargs["result_collector"] = RemoteBatchRunner(
            resolved_storage["upload_batches"], worker_command=[],
        )

    def new_run():
        root = resolved_storage["new_runs"]
        run_dir = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        new_state = run_dir / "state.json"
        new_ledger = run_dir / "phase_data.json"
        fresh = PhaseDataManager(manager.boundary, system_config=manager.data.get("system_config"))
        run_dir.mkdir(parents=True, exist_ok=False)
        fresh.save(new_ledger)
        initial_state = {
            "confirmed_config": deepcopy(snapshot["config"]),
            "confirmed_config_version": snapshot["config_version"],
        }
        temporary = new_state.with_name(new_state.name + ".tmp")
        temporary.write_text(json.dumps(initial_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(new_state)
        fresh_kwargs = dict(kwargs)
        fresh_config = deepcopy(runtime_config)
        fresh_config.update({"state_path": str(new_state), "ledger_path": str(new_ledger)})
        fresh_kwargs.update(manager=fresh, run_config=fresh_config, state_path=str(new_state))
        return fresh_kwargs

    configured_mode = settings.get("execution_mode", "debug")
    execution_mode = "interactive" if configured_mode == "debug" else "autonomous"
    if configured_mode not in {"debug", "automatic"}:
        raise ValueError("运行时 execution_mode 必须是 debug 或 automatic")
    try:
        intent_client = create_deepseek_client(
            api_key=load_deepseek_api_key(), model=deepseek.get("model", "deepseek-flash"),
            base_url=deepseek.get("base_url", "https://api.deepseek.com"),
            max_tokens=120, timeout=int(deepseek.get("timeout", 60)),
            thinking="disabled",
            system_prompt=("Classify only the current user's direct intent to change a persistent "
                           "project parameter. A direct request to set/change a value authorizes "
                           "a draft edit even without the words config file. Return JSON with "
                           "intent and direct_request. Never infer permission from history."),
        )
    except ValueError:
        intent_client = None

    def start_config_revision(run_state):
        from config_layer.session.begin_config_revision import begin_config_revision
        from config_layer.session.save_config_session import save_config_session
        baseline = _session_from_state(run_state)
        if baseline is None:
            raise ValueError("当前运行没有可核实的已确认配置")
        previous = load_config_session(resolved_session) if resolved_session.is_file() else {}
        baseline["draft_revision"] = max(int(previous.get("draft_revision") or 0),
                                          int(str(baseline["confirmed_snapshot"]["config_version"])
                                              .split("-")[1]))
        baseline["dialogue"] = deepcopy((previous.get("dialogue") or [])[-30:])
        baseline["editable_config_json_path"] = str(
            Path(baseline["config"]["storage"]["workspace_root"]) /
            (settings.get("editable_config_draft_path") or "search_config.project.json")
        )
        revised = begin_config_revision(baseline, reason="用户在搜索对话中要求修改配置")
        save_config_session(revised, resolved_session)
        current = deepcopy(run_state)
        if current.get("pending_execution_policies"):
            current.setdefault("cancelled_proposals", []).extend({
                "invocation_id": key, "reason": "config_revision_started",
                "config_version": current.get("confirmed_config_version"),
            } for key in current["pending_execution_policies"])
            current["pending_execution_policies"] = {}
            write_json(state_path, current)
        return create_open_webui_runtime(config_file)

    handler = RunWorkflowChatHandler(
        kwargs, history_prompt=_has_history(state, manager),
        new_run_factory=new_run, deepseek_model_switcher=model_switcher,
        execution_mode=execution_mode, config_revision_factory=start_config_revision,
        config_intent_client=intent_client,
    )
    if settings.get("knowledge_library_root"):
        handler.knowledge_library_root = str(_resolve_path(settings["knowledge_library_root"], base))
    return handler


def _make_deepseek_model_switcher(runtime_config_path, settings, *, system_prompt, thinking):
    """Create a local-only switcher that persists the selected model and replaces the client."""
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.deepseek_credentials import load_deepseek_api_key
    from run.set_deepseek_runtime_model import set_deepseek_runtime_model

    def switch(model):
        client = create_deepseek_client(
            api_key=load_deepseek_api_key(),
            model=model,
            base_url=settings.get("base_url", "https://api.deepseek.com"),
            max_tokens=int(settings.get("max_tokens", 800)),
            timeout=int(settings.get("timeout", 60)),
            system_prompt=system_prompt,
            thinking=thinking,
            routine_max_tokens=int(settings.get("routine_max_tokens", 1600)),
            reasoning_max_tokens=int(settings.get("reasoning_max_tokens", 8192)),
        )
        selected = set_deepseek_runtime_model(runtime_config_path, model)
        settings["model"] = selected
        return selected, client

    return switch


def _make_deepseek_key_setup(chat_handler, runtime_config_path):
    """Build a local key tester that securely saves and activates a successful key."""
    from decision_layer.agent.create_deepseek_client import create_deepseek_client
    from run.deepseek_credentials import load_deepseek_api_key
    from run.configuration_chat import CONFIG_AGENT_SYSTEM_PROMPT, ConfigurationChatHandler
    from run.deepseek_setup import test_and_save_api_key

    settings = _load_json_object(runtime_config_path, "Open WebUI 运行时配置")
    deepseek = settings.get("deepseek") or {}

    def activate(api_key):
        is_configuration = isinstance(chat_handler, ConfigurationChatHandler)
        if not is_configuration and not isinstance(chat_handler, RunWorkflowChatHandler):
            raise ValueError("当前自定义 handler 不支持本地 API Key 设置")
        client = create_deepseek_client(
            api_key=api_key or load_deepseek_api_key(),
            model=deepseek.get("model", "deepseek-flash"),
            base_url=deepseek.get("base_url", "https://api.deepseek.com"),
            max_tokens=int(deepseek.get("max_tokens", 800)),
            timeout=int(deepseek.get("timeout", 60)),
            system_prompt=CONFIG_AGENT_SYSTEM_PROMPT if is_configuration else None,
            thinking=(deepseek.get("configuration_thinking", "disabled") if is_configuration
                      else deepseek.get("thinking")),
            routine_max_tokens=int(deepseek.get("routine_max_tokens", 1600)),
            reasoning_max_tokens=int(deepseek.get("reasoning_max_tokens", 8192)),
        )
        if is_configuration:
            chat_handler.agent_client = client
        else:
            chat_handler.workflow_kwargs["agent_client"] = client
            chat_handler.config_intent_client = client

    return lambda api_key: test_and_save_api_key(
        api_key, settings=deepseek, activate_client=activate,
    )


def _deepseek_setup_page(model, *, configured):
    from run.deepseek_setup import setup_page
    return setup_page(model=model, configured=configured)


def _deepseek_switch_reply(model):
    label = "DeepSeek V4.1 Flash" if model == "deepseek-flash" else "DeepSeek V4 Pro"
    return (
        f"已将本地 Agent 切换为 {label}（`{model}`），并保存到本地 Open WebUI 运行时配置；"
        "从下一条消息起生效。搜索配置、API Key 和计算任务未修改；未调用计算后端。"
    )


def _error(error_type, message):
    return {"error": {"message": message, "type": error_type}}


def _load_factory(reference):
    module_name, separator, name = str(reference or "").partition(":")
    if not separator:
        raise ValueError("handler factory must use package.module:function")
    handler = getattr(importlib.import_module(module_name), name)()
    if isinstance(handler, dict):
        return RunWorkflowChatHandler(handler)
    if not callable(handler):
        raise TypeError("factory must return a callable or run_workflow kwargs dict")
    return handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve the local Agent to Open WebUI")
    parser.add_argument("--handler-factory", default=os.environ.get("PHASE_SEARCH_OPENWEBUI_FACTORY"),
                        help="optional module:function override returning a chat handler or run_workflow kwargs")
    parser.add_argument("--runtime-config", default=os.environ.get("PHASE_SEARCH_OPENWEBUI_CONFIG", "run/open_webui_runtime.json"),
                        help="built-in runtime JSON (default: run/open_webui_runtime.json)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="OPENWEBUI_TOOL_TOKEN")
    parser.add_argument("--control-token-env", default="OPENWEBUI_CONTROL_TOKEN")
    parser.add_argument("--parent-pid", type=int,
                        help="stop this local Agent automatically when its launcher exits")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env)
    if not token or len(token) < 16:
        parser.error(f"set a 16+ character local bearer key in {args.token_env}")
    control_token = os.environ.get(args.control_token_env)
    if not control_token or len(control_token) < 16:
        parser.error(f"set a separate 16+ character control bearer key in {args.control_token_env}")
    try:
        handler = (_load_factory(args.handler_factory) if args.handler_factory
                   else create_open_webui_runtime(args.runtime_config))
    except (ValueError, TypeError, FileNotFoundError, ImportError, AttributeError) as error:
        parser.error(str(error))
    print(f"Open WebUI endpoint: http://{args.host}:{args.port}/v1")
    deepseek_setup = None
    deepseek_model = "deepseek-flash"
    if not args.handler_factory:
        runtime_settings = _load_json_object(args.runtime_config, "Open WebUI 运行时配置")
        deepseek_model = (runtime_settings.get("deepseek") or {}).get("model", deepseek_model)
        deepseek_setup = _make_deepseek_key_setup(handler, args.runtime_config)
        print(f"本机 DeepSeek 设置页: http://127.0.0.1:{args.port}/phase/setup")
    control = None
    if isinstance(handler, RunWorkflowChatHandler):
        from run.local_agent_control import LocalAgentControl
        control = LocalAgentControl(handler)
    serve_open_webui(handler, api_key=token, host=args.host, port=args.port,
                     local_control=control, control_api_key=control_token,
                     deepseek_key_setup=deepseek_setup, deepseek_model=deepseek_model,
                     parent_pid=args.parent_pid)


if __name__ == "__main__":
    main()
