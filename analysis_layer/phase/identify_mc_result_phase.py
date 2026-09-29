"""Backward-compatible MC phase identification entry."""

from analysis_layer.phase.identify_result_phase import identify_result_phase


def identify_mc_result_phase(result, manager, *, phase_references=None, cache=None):
    if result.get("stage") != "deep_search":
        return result
    return identify_result_phase(result, manager, phase_references=phase_references, cache=cache)
