"""Public runtime API."""

from run.default_run_config import default_run_config

__all__ = [
    "default_run_config",
    "run_workflow",
]


def __getattr__(name):
    """Import scientific workflow dependencies only for the public entry."""
    if name == "run_workflow":
        from run.main import run_workflow

        globals()[name] = run_workflow
        return run_workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
