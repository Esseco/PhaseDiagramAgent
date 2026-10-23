"""One workflow instance shared by chat and the existing control pages."""


class DeferredControl:
    """Resolve the existing handler after a new project's config is confirmed."""

    def __init__(self, handler):
        self.handler = handler

    def __getattr__(self, name):
        def call(*args, **kwargs):
            from phase_agent.runtime.agent_api import RunWorkflowChatHandler
            from phase_agent.runtime.local_agent_control import LocalAgentControl

            handler = self.handler
            seen = set()
            while not isinstance(handler, RunWorkflowChatHandler):
                if handler is None or id(handler) in seen:
                    raise ValueError("配置尚未确认，请先在聊天中完成配置")
                seen.add(id(handler))
                handler = getattr(handler, "delegate", None)
            from phase_agent.runtime.studio_runtime import _message_lock

            with _message_lock:
                return getattr(LocalAgentControl(handler), name)(*args, **kwargs)

        return call


def configure_decision_backend(handler, backend):
    from phase_agent.runtime.agent_api import RunWorkflowChatHandler

    if isinstance(handler, RunWorkflowChatHandler):
        handler.decision_backend = backend
    for name in ("runtime_factory", "new_run_factory", "config_revision_factory"):
        factory = getattr(handler, name, None)
        if callable(factory):

            def configured_factory(*args, _factory=factory, **kwargs):
                result = _factory(*args, **kwargs)
                configure_decision_backend(result, backend)
                return result

            setattr(handler, name, configured_factory)


def create_control_server(handler, runtime_config, *, port):
    from phase_agent.runtime.local_service_tokens import local_service_tokens
    from phase_agent.runtime.agent_api import (
        create_server,
        _make_deepseek_key_setup,
        _load_json_object,
    )

    tool_token, control_token = local_service_tokens()
    settings = _load_json_object(runtime_config, "Agent runtime configuration")

    def chat(*args, **kwargs):
        from phase_agent.runtime.studio_runtime import _message_lock

        with _message_lock:
            from phase_agent.graphs.dialogue.commands import (
                _latest_user_message,
                _open_webui_metadata_reply,
            )
            from phase_agent.runtime.agent_api import _chat_content
            from phase_agent.runtime.turn_process import record_turn
            from phase_agent.runtime.interaction_reports import save_interaction
            from phase_agent.runtime.studio_runtime import project_handler
            from phase_agent.runtime.turn_process import read_presentation_state
            from pathlib import Path

            message = _latest_user_message(args[0] if args else kwargs.get("messages", []))
            if _open_webui_metadata_reply(message) is not None:
                return handler(*args, **kwargs)
            with record_turn(runtime_config, message, kwargs.get("conversation_id")) as record:
                reply = _chat_content(handler(*args, **kwargs))
                owner = project_handler(handler)
                state = read_presentation_state(owner.state_path) if owner else {}
                try:
                    return save_interaction(
                        Path(runtime_config).resolve().parent,
                        message,
                        reply,
                        reply,
                        state,
                        record=record,
                    )
                except (OSError, ValueError, TypeError):
                    return reply + "\n交互 Markdown 保存失败；保留本轮答复，未重放操作。"

    return create_server(
        chat,
        api_key=tool_token,
        port=port,
        local_control=DeferredControl(handler),
        control_api_key=control_token,
        deepseek_key_setup=_make_deepseek_key_setup(handler, runtime_config),
        deepseek_model=(settings.get("deepseek") or {}).get("model", "deepseek-flash"),
    )
