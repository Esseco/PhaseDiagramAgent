"""Keep the dev server's live checkpoint stores registered after history reads."""

from functools import wraps


def register_live_stores(checkpoint, register):
    memory = checkpoint.MEMORY
    if memory is not None:
        for name in ("storage", "writes", "blobs"):
            register(getattr(memory, name))


def wrap_checkpointer(checkpoint, register):
    original = checkpoint.Checkpointer
    if getattr(original, "_phase_live_store_registration", False):
        return original

    @wraps(original)
    def preserved(*args, **kwargs):
        result = original(*args, **kwargs)
        register_live_stores(checkpoint, register)
        return result

    preserved._phase_live_store_registration = True
    checkpoint.Checkpointer = preserved
    return preserved


def install_checkpoint_persistence():
    from langgraph_runtime_inmem import checkpoint, ops
    from langgraph_runtime_inmem._persistence import register_persistent_dict

    from phase_agent.persistence.studio_storage import (
        install_memory_adapter,
        install_mutation_adapters,
    )

    from phase_agent.persistence.studio_checkpoint_store import (
        StudioCheckpointDict,
        checkpoint_container,
    )

    checkpoint.PersistentDict = checkpoint_container
    if checkpoint.MEMORY is not None:
        for name in ("storage", "writes", "blobs"):
            old = getattr(checkpoint.MEMORY, name)
            if not isinstance(old, StudioCheckpointDict):
                replacement = checkpoint_container(
                    getattr(old, "default_factory", None), filename=old.filename
                )
                if isinstance(replacement, StudioCheckpointDict):
                    replacement.load()
                    setattr(checkpoint.MEMORY, name, replacement)
    install_memory_adapter(checkpoint)
    ops.Checkpointer = wrap_checkpointer(checkpoint, register_persistent_dict)
    install_mutation_adapters(checkpoint, ops)
    register_live_stores(checkpoint, register_persistent_dict)


def flush_checkpoint_persistence():
    from langgraph_runtime_inmem import checkpoint

    if checkpoint.MEMORY is not None:
        for name in ("storage", "writes", "blobs"):
            getattr(checkpoint.MEMORY, name).sync()
