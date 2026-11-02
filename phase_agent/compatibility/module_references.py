"""Resolve saved module references without restoring old application packages."""

from functools import cache
from importlib import import_module
import json
from pathlib import Path

_PREVIOUS_PACKAGES = {
    "orchestration": "phase_agent.graphs",
    "run": "phase_agent.runtime",
    "config_layer": "phase_agent.configuration",
    "data_layer": "phase_agent.persistence",
    "decision_layer": "phase_agent.decisions",
    "execution_layer": "phase_agent.tools",
    "scientific_layer": "phase_agent.science",
    "analysis_layer": "phase_agent.analysis",
}


@cache
def _module_moves():
    return json.loads(Path(__file__).with_name("module_moves.json").read_text(encoding="utf-8"))


def import_saved_module(name):
    """Translate historical top-level names, then explicit module/package moves."""
    prefix, separator, remainder = name.partition(".")
    if prefix in _PREVIOUS_PACKAGES:
        name = _PREVIOUS_PACKAGES[prefix] + (separator + remainder if separator else "")
    moves = _module_moves()
    if name in moves:
        name = moves[name]
    else:
        for old in sorted(moves, key=len, reverse=True):
            if name.startswith(old + "."):
                name = moves[old] + name[len(old) :]
                break
    return import_module(name)
