"""Recover a project session without selecting or approving scientific tools."""

from phase_agent.graphs.dialogue.commands import _classify_history_decision, format_history_prompt
from phase_agent.graphs.dialogue.errors import OpenWebUIRequestError


def restore_session(handler, message, state):
    if handler.history_decision is not None:
        return None
    choice = _classify_history_decision(message)
    if choice is None:
        return format_history_prompt(state, handler.workflow_kwargs.get("manager"))
    if choice == "new":
        if not callable(handler.new_run_factory):
            raise OpenWebUIRequestError("运行时未配置安全的新建运行目录工厂。")
        replacement = handler.new_run_factory()
        from phase_agent.runtime.chat_application import RunWorkflowChatHandler

        if isinstance(replacement, RunWorkflowChatHandler):
            for name in (
                "workflow_kwargs",
                "state_path",
                "workflow",
                "new_run_factory",
                "deepseek_model_switcher",
                "config_revision_factory",
                "execution_mode",
                "runtime_config_path",
                "knowledge_library_root",
            ):
                if hasattr(replacement, name):
                    setattr(handler, name, getattr(replacement, name))
        else:
            from pathlib import Path

            replacement = dict(replacement)
            if "agent_client" in handler.workflow_kwargs:
                replacement["agent_client"] = handler.workflow_kwargs["agent_client"]
            handler.workflow_kwargs = replacement
            handler.state_path = Path(replacement["state_path"])
        handler.config_delegate = None
        handler.history_decision = "new"
        return f"已新建独立运行：`{handler.state_path.parent}`。旧 state/ledger 未修改。请发送下一条搜索指令。"
    if len(state.get("pending_execution_policies") or {}) > 1:
        raise OpenWebUIRequestError("历史 state 含多个待审批 action，请先恢复为唯一待审批状态。")
    handler.history_decision = "continue"
    from phase_agent.tools.remote.summarize_manual_upload_wait import summarize_manual_upload_wait

    if state.get("pending_execution_policies") or summarize_manual_upload_wait(state):
        return None  # Route through the same decision node with fresh facts.
    return "已继续原运行并加载 state/ledger。请发送下一条搜索指令；本次确认不会批准任何 action。"
