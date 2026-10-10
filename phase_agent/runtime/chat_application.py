"""Chat transport; all flow control belongs to the dialogue graph."""

from pathlib import Path
from copy import deepcopy
import os
import uuid
from phase_agent.graphs.invocation_context import new_invocation
from phase_agent.tools.step_runner.file_protocol import read_json, write_json
from phase_agent.runtime.workflow_reply_presentation import (
    format_workflow_reply,
    _is_model_failure_proposal,
)
import threading
from phase_agent.graphs.dialogue.commands import (
    classify_user_decision,
    _is_config_migration_approval,
    _mentions_explicit_branch_generation,
    _drop_finished_pending,
    format_status_reply,
    format_history_prompt,
    _latest_user_message,
    _open_webui_metadata_reply,
    _is_status_command,
    _phase_csv_request,
    _classify_history_decision,
    _deepseek_switch_reply,
)
from phase_agent.graphs.dialogue.errors import OpenWebUIRequestError
from phase_agent.tools.step_runner.build_status_summary import build_status_summary
from phase_agent.runtime.chat_state_presentation import brief_chat_state
from phase_agent.runtime.chat_approval_rules import is_sensitive_proposal as _is_sensitive_proposal


class RunWorkflowChatHandler:
    def __init__(
        self,
        workflow_kwargs: dict,
        *,
        workflow=None,
        history_prompt=False,
        new_run_factory=None,
        deepseek_model_switcher=None,
        execution_mode="interactive",
        config_revision_factory=None,
    ):
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
        self.config_delegate = None
        if execution_mode not in {"interactive", "autonomous"}:
            raise ValueError("execution_mode 必须是 interactive 或 autonomous")
        # Chat always proposes first; an execution-mode preference cannot grant
        # human approval or authorize the next action after an approved one.
        self.requested_execution_mode = execution_mode
        self.execution_mode = "interactive"
        self.conversation_id = None
        self.recent_dialogue = []
        self.studio_dialogues = {}

    def start_analysis(self, *, conversation_id=None):
        """Enter the decision loop directly after configuration confirmation."""
        with self.lock:
            if self.conversation_id not in {None, conversation_id}:
                raise OpenWebUIRequestError("此运行时已绑定另一个会话。")
            self.conversation_id = conversation_id
            result = self._run(
                new_invocation(),
                None,
                "配置已由用户确认。根据当前配置、任务和已回收结果分析当前阶段，"
                "提出下一步的一项具体动作及预算，说明依据并等待人工审核。"
                "没有结果时从首轮搜索开始；不得直接执行、提交或派发计算。",
            )
            return format_workflow_reply(result, self.state_path)

    def __call__(self, messages, *, conversation_id=None):
        # Studio often transports only the newest user turn. Keep a small local
        # tail rather than repeatedly sending the full session or summarizing it.
        with self.lock:
            from phase_agent.runtime.studio_session_scope import is_project_session, thread_history

            studio_turn = is_project_session()
            if studio_turn:
                self.conversation_id = conversation_id
                persisted = thread_history()
                self.recent_dialogue = (
                    list(persisted[:-1])[-4:]
                    if persisted is not None
                    else list(self.studio_dialogues.get(conversation_id) or [])
                )
            supplied = list(messages)
            if len(supplied) == 1 and self.conversation_id in {None, conversation_id}:
                supplied = [*self.recent_dialogue, *supplied]
            self.last_turn_outcome = None
            reply = self._respond(supplied, conversation_id=conversation_id)
            read_only_reply = self.last_turn_outcome in {"answered", "configuration_view_requested"}
            if _open_webui_metadata_reply(_latest_user_message(messages)) is None:
                self.recent_dialogue = [
                    *self.recent_dialogue,
                    {"role": "user", "content": str(_latest_user_message(messages))[:450]},
                    {"role": "assistant", "content": str(reply)[:450]},
                ][-4:]
                if studio_turn:
                    self.studio_dialogues[conversation_id] = list(self.recent_dialogue)
        if _open_webui_metadata_reply(_latest_user_message(messages)) is not None:
            return reply
        if str(reply).startswith("当前项目状态："):
            return reply  # Progress already contains its stage; avoid duplicate headers.
        from phase_agent.graphs.dialogue.memory import remember_turn

        remember_turn(self.state_path, conversation_id, _latest_user_message(messages), reply)
        if read_only_reply:
            return reply
        state = read_json(self.state_path, {}) or {}
        return (
            brief_chat_state(state, configuring=self.config_delegate is not None)
            + "\n"
            + str(reply)
        )

    def review_pending(
        self, plan_id, decision, *, expected_state_version, expected_proposal_hash, comment=""
    ):
        from phase_agent.runtime.chat_review import review_pending

        return review_pending(
            self,
            plan_id,
            decision,
            expected_state_version=expected_state_version,
            expected_proposal_hash=expected_proposal_hash,
            comment=comment,
            request_error=OpenWebUIRequestError,
            is_sensitive=_is_sensitive_proposal,
        )

    def _run(
        self,
        invocation_id,
        human_feedback,
        user_message,
        *,
        approve_config_migration=False,
        dft_recovery_decision=None,
    ):
        self.unified_dialogue = callable(self.workflow_kwargs.get("agent_client"))
        from phase_agent.runtime.chat_execution import run_workflow_turn

        return run_workflow_turn(
            self,
            invocation_id,
            human_feedback,
            user_message,
            approve_config_migration=approve_config_migration,
            dft_recovery_decision=dft_recovery_decision,
            explicit_branch_request=_mentions_explicit_branch_generation(user_message),
        )

    def _respond(self, messages, *, conversation_id=None):
        from phase_agent.graphs.dialogue.graph import run_project_turn

        return run_project_turn(self, messages, conversation_id)
