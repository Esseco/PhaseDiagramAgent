"""Default same-geometry comparison through the existing py-mace boundary."""
from pathlib import Path


def create_dft_comparison_evaluator(config):
    model = config.get("mlip") or {}
    version = model.get("version") or model.get("name")
    path = model.get("model_path")
    if not path and model.get("model_paths"):
        path = model["model_paths"][int(model.get("main_model_index", 0))]
    base = config.get("state_path")

    def evaluate(*, result, structure_id, manager):
        if not version or result.get("model_version") != version:
            raise ValueError("DFT task model version does not match configured comparison model")
        if not path or not Path(path).is_file() or not base:
            raise ValueError("comparison model file or state path unavailable")
        from pymatgen.core import Structure
        from scientific_layer.mlip.run_mace_subprocess import run_mace_with_py_mace
        import hashlib
        key = hashlib.sha256(str(result["task_id"]).encode()).hexdigest()[:16]
        parameters = {"device": "cpu"}
        if "mh-1" in str(path).lower() or "mace-mh-1" in str(version).lower():
            parameters["mace_head"] = "omat_pbe"
        evaluated = run_mace_with_py_mace(
            Structure.from_dict(result["outputs"]["structure"]), model_path=path,
            operation="predict", parameters=parameters,
            work_directory=Path(base).parent / "dft_comparisons" / key)
        if evaluated.get("status") != "completed":
            raise ValueError(evaluated.get("error") or "MLIP prediction failed")
        return {**evaluated, "model_version": version}
    return evaluate
