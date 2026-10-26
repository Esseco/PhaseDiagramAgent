"""Install lazy checkpoint containers before the development server initializes."""


def _is_benign_transport_disconnect(context):
    """Identify client-side socket closes emitted by Windows' Proactor loop."""
    exception = context.get("exception") if isinstance(context, dict) else None
    if not isinstance(exception, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return False
    handle = context.get("handle") if isinstance(context, dict) else None
    callback = getattr(handle, "_callback", None)
    name = getattr(callback, "__qualname__", "") or getattr(callback, "__name__", "")
    return "connection_lost" in name or "_call_connection_lost" in name


def _install_transport_exception_filter():
    """Keep expected browser disconnects out of service error logs."""
    import asyncio

    policy = asyncio.get_event_loop_policy()
    if getattr(policy, "_phase_transport_filter_installed", False):
        return
    original = policy.new_event_loop

    def new_event_loop():
        loop = original()

        def handler(current_loop, context):
            if _is_benign_transport_disconnect(context):
                return
            current_loop.default_exception_handler(context)

        loop.set_exception_handler(handler)
        return loop

    policy.new_event_loop = new_event_loop
    policy._phase_transport_filter_installed = True


def main():
    import runpy
    from langgraph_runtime_inmem import checkpoint
    from phase_agent.persistence.studio_checkpoint_store import checkpoint_container

    _install_transport_exception_filter()
    checkpoint.PersistentDict = checkpoint_container
    from phase_agent.persistence.studio_storage import install_memory_adapter

    install_memory_adapter(checkpoint)
    runpy.run_module("langgraph_cli", run_name="__main__")


if __name__ == "__main__":
    main()
