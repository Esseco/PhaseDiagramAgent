"""Resolve saved module references after the application package migration.

Only explicit old package prefixes are translated. This preserves saved adapter
references without restoring old packages, entry points or workflow controllers.
"""

from importlib import import_module

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


def import_saved_module(name):
    prefix, separator, remainder = name.partition(".")
    if prefix in _PREVIOUS_PACKAGES:
        name = _PREVIOUS_PACKAGES[prefix] + (separator + remainder if separator else "")
    return import_module(name)
