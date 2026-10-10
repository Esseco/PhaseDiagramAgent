__all__ = ["run_mlip_mc"]


def __getattr__(name):
    if name == "run_mlip_mc":
        from phase_agent.science.mc.run_mlip_mc import run_mlip_mc

        return run_mlip_mc
    raise AttributeError(name)
