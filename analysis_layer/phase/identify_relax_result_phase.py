"""Identify the local final Relax phase while preserving original branch ownership."""
from analysis_layer.phase.identify_result_phase import identify_result_phase


def identify_relax_result_phase(result, manager, *, phase_references=None, cache=None):
    if result.get("stage") != "relax_and_feature":
        return result
    return identify_result_phase(result, manager, phase_references=phase_references, cache=cache)
