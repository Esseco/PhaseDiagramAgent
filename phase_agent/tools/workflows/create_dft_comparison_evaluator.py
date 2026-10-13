"""Validate remote same-frame predictions; never run local inference."""

from copy import deepcopy

from phase_agent.tools.workflows.comparison_model_registry import (
    original_round_model,
    remember_comparison_models,
)


def create_dft_comparison_evaluator(config, *, state=None):
    references = deepcopy(state or {})
    remember_comparison_models(references, config)

    def evaluate(*, result, structure_id, manager):
        version = result.get("model_version")
        model, expected_digest, signature = original_round_model(references, config, version)
        prediction = (result.get("outputs") or {}).get("remote_mlip_prediction")
        if prediction is not None:
            from phase_agent.tools.remote.dft_mlip_pair import validate_prediction

            return validate_prediction(result, prediction, expected_digest)
        raise ValueError(
            "remote mlip_result.json unavailable; supplement same-frame GPU prediction on HPC and return both files; local inference is disabled"
        )

    return evaluate
