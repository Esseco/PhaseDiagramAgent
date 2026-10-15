"""Install lazy checkpoint containers before the development server initializes."""


def main():
    import runpy
    from langgraph_runtime_inmem import checkpoint
    from phase_agent.persistence.studio_checkpoint_store import checkpoint_container

    checkpoint.PersistentDict = checkpoint_container
    from phase_agent.persistence.studio_storage import install_memory_adapter

    install_memory_adapter(checkpoint)
    runpy.run_module("langgraph_cli", run_name="__main__")


if __name__ == "__main__":
    main()
