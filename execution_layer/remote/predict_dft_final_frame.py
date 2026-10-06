"""Remote same-frame prediction, also usable on an already completed results folder."""
import argparse
import json
from pathlib import Path
from execution_layer.remote.integrity import file_checksum
from execution_layer.remote.dft_mlip_pair import load_pair, validate_prediction
from scientific_layer.structures.load_result_structure import load_result_structure
from scientific_layer.mlip.run_mace_subprocess import run_mace_with_py_mace


def write_prediction(root, result, model, *, predictor=None):
    root = Path(root)
    prediction = {key: result.get(key) for key in ("task_id", "task_key", "model_version")}
    try:
        if model.get("version", model.get("name")) != result.get("model_version"):
            raise ValueError("remote comparison model version mismatch")
        structure, digest = load_result_structure(result.get("outputs") or {})
        path = model.get("model_path")
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"remote comparison model unavailable: {path}")
        fingerprint = file_checksum(path)
        expected = model.get("model_sha256")
        if expected and fingerprint != expected:
            raise ValueError("remote comparison model fingerprint mismatch")
        parameters = {"device": "cpu", "mace_default_dtype": "float64"}
        if model.get("mace_head"):
            parameters["mace_head"] = model["mace_head"]
        evaluated = (predictor or run_mace_with_py_mace)(
            structure, model_path=path, operation="predict", work_directory=root / "mlip_comparison_work",
            parameters=parameters, environment=model.get("environment") or "py-mace")
        if evaluated.get("status") != "completed":
            raise ValueError(evaluated.get("error") or "remote MLIP prediction failed")
        prediction.update({key: evaluated.get(key) for key in ("energy", "forces", "energy_unit", "forces_unit")})
        prediction.update(status="completed", model_sha256=fingerprint, comparison_model_path=str(path),
                          structure_sha256=digest, final_frame_index=(result.get("outputs") or {}).get("final_frame_index"),
                          geometry="DFT_final_frame")
        validate_prediction(result, prediction, expected)
    except Exception as error:
        prediction.update(status="failed", error=f"{type(error).__name__}: {error}")
    _write(root / "mlip_result.json", prediction)
    result.setdefault("outputs", {}).update(mlip_result_file="mlip_result.json",
                                            mlip_result_checksum=file_checksum(root / "mlip_result.json"))
    return prediction


def _write(path, payload):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def supplement_results(directory, model, *, predictor=None):
    reports = []
    for path in sorted(Path(directory).rglob("result.json")):
        marker_path = path.parent / "task.finished.json"
        if not marker_path.is_file():
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("stage") not in {"dft_single_point", "dft_relax"} or result.get("status") != "completed":
            continue
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("result_checksum") != file_checksum(path):
            raise ValueError(f"existing DFT result checksum mismatch: {path}")
        old = load_pair(result, path.parent)
        if old and old.get("status") == "completed":
            validate_prediction(result, old, model.get("model_sha256"))
            if old["model_sha256"] != file_checksum(model["model_path"]):
                raise ValueError("existing prediction belongs to different model content")
            reports.append({"task_id": result["task_id"], "status": "cached"})
            continue
        prediction = write_prediction(path.parent, result, model, predictor=predictor)
        _write(path, result)
        marker["result_checksum"] = file_checksum(path)
        _write(marker_path, marker)
        reports.append({"task_id": result["task_id"], "status": prediction["status"], "error": prediction.get("error")})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--head", default="omat_pbe")
    parser.add_argument("--environment", default="py-mace")
    args = parser.parse_args()
    reports = supplement_results(args.results, {"version": args.model_version, "model_path": args.model_path,
        "mace_head": args.head, "environment": args.environment})
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 1 if any(row["status"] == "failed" for row in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
