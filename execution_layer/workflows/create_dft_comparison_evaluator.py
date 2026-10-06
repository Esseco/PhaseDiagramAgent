"""Default same-geometry comparison through the existing py-mace boundary."""
from pathlib import Path
from copy import deepcopy

from execution_layer.workflows.comparison_model_registry import (
    comparison_model_path, original_round_model, remember_comparison_models, verified_model_digest,
)


def create_dft_comparison_evaluator(config, *, state=None):
    references = deepcopy(state or {})
    remember_comparison_models(references, config)
    base = config.get("state_path")

    def evaluate(*, result, structure_id, manager):
        version = result.get("model_version")
        model, expected_digest, signature = original_round_model(references, config, version)
        prediction = (result.get("outputs") or {}).get("remote_mlip_prediction")
        if prediction is not None:
            from execution_layer.remote.dft_mlip_pair import validate_prediction
            return validate_prediction(result, prediction, expected_digest)
        if config.get("dft_comparison_location", "remote") != "local":
            raise ValueError("remote mlip_result.json unavailable; supplement same-frame prediction on HPC and return both files")
        path = comparison_model_path(model)
        if not path or not Path(path).is_file() or not base:
            raise ValueError("comparison model file or state path unavailable")
        digest = verified_model_digest(path, expected_digest, signature)
        from scientific_layer.structures.load_result_structure import load_result_structure
        from scientific_layer.mlip.run_mace_subprocess import run_mace_with_py_mace
        import hashlib
        structure, structure_digest = load_result_structure(result["outputs"])
        key = hashlib.sha256(f"{result['task_id']}:{version}:{digest}:{structure_digest}".encode()).hexdigest()[:16]
        parameters = {"device": "cpu"}
        if model.get("mace_head"):
            parameters["mace_head"] = model["mace_head"]
        elif "mh-1" in str(path).lower() or "mace-mh-1" in str(version).lower():
            parameters["mace_head"] = "omat_pbe"
        evaluated = run_mace_with_py_mace(
            structure, model_path=path,
            operation="predict", parameters=parameters,
            work_directory=Path(base).parent / "dft_comparisons" / key)
        if evaluated.get("status") != "completed":
            raise ValueError(evaluated.get("error") or "MLIP prediction failed")
        return {**evaluated, "model_version": version, "model_sha256": digest,
                "comparison_model_path": str(path), "geometry": "DFT_final_frame"}
    return evaluate
