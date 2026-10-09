"""One workflow instance shared by chat and the existing control pages."""
import threading


class DeferredControl:
    """Resolve the existing handler after a new project's config is confirmed."""
    def __init__(self, handler):
        self.handler = handler

    def __getattr__(self, name):
        def call(*args, **kwargs):
            from run.agent_api import RunWorkflowChatHandler
            from run.local_agent_control import LocalAgentControl
            handler = self.handler
            seen = set()
            while not isinstance(handler, RunWorkflowChatHandler):
                if handler is None or id(handler) in seen:
                    raise ValueError("配置尚未确认，请先在聊天中完成配置")
                seen.add(id(handler))
                handler = getattr(handler, "delegate", None)
            from run.studio_runtime import _message_lock
            with _message_lock:
                return getattr(LocalAgentControl(handler), name)(*args, **kwargs)
        return call


def configure_decision_backend(handler, backend):
    from run.agent_api import RunWorkflowChatHandler
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
    from run.local_service_tokens import local_service_tokens
    from run.agent_api import (create_server,
                                   _make_deepseek_key_setup, _load_json_object)
    tool_token, control_token = local_service_tokens()
    settings = _load_json_object(runtime_config, "Agent runtime configuration")
    def chat(*args, **kwargs):
        from run.studio_runtime import _message_lock
        with _message_lock:
            return handler(*args, **kwargs)
    return create_server(
        chat, api_key=tool_token, port=port,
        local_control=DeferredControl(handler),
        control_api_key=control_token,
        deepseek_key_setup=_make_deepseek_key_setup(handler, runtime_config),
        deepseek_model=(settings.get("deepseek") or {}).get("model", "deepseek-flash"),
    )


def serve_unified(app, control_server, *, port, parent_pid=None):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port))
    stopped = threading.Event()

    def watch_parent():
        import psutil
        while not stopped.wait(1):
            if not psutil.pid_exists(parent_pid):
                server.should_exit = True
                return

    thread = threading.Thread(target=control_server.serve_forever, daemon=True)
    thread.start()
    if parent_pid is not None:
        threading.Thread(target=watch_parent, daemon=True).start()
    try:
        server.run()
    finally:
        stopped.set()
        control_server.shutdown()
        control_server.server_close()
        thread.join(timeout=5)
