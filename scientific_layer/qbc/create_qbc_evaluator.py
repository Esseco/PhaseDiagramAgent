"""Build a candidate evaluator around the existing pure QBC calculation."""

from scientific_layer.qbc.evaluate_qbc import evaluate_qbc


def create_qbc_evaluator(committee, *, predictor, structure_loader=None):
    """Return the adapter expected by ``build_qbc_candidate_metrics``.

    The adapter calculates uncertainty only. It never selects an action.
    """
    def evaluator(candidate):
        structure = candidate.get("structure")
        if structure is None and structure_loader is not None:
            structure = structure_loader(candidate)
        if structure is None:
            return {
                "status": "not_configured",
                "error": "candidate structure is unavailable",
                "interpretation": "committee_disagreement_not_true_error",
            }
        return evaluate_qbc(structure, committee, predictor=predictor)

    return evaluator
