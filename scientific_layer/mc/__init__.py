__all__ = ["run_mlip_mc"]


def __getattr__(name):
    if name == "run_mlip_mc":
        from scientific_layer.mc.run_mlip_mc import run_mlip_mc
        return run_mlip_mc
    raise AttributeError(name)
