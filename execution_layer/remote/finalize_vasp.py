"""Create the same integrity envelope for independently executed VASP tasks."""
import argparse
import json
from pathlib import Path
from execution_layer.remote.integrity import file_checksum, payload_checksum
from execution_layer.remote.export_batch_result import export_batch_result
from scientific_layer.dft.parse_vasp_result import parse_vasp_result


def resolve_calculation_directory(root):
    """Use the authoritative checkpoint, never a stale earlier attempt."""
    root = Path(root).resolve()
    checkpoint = root / "workflow_state.json"
    if checkpoint.is_file():
        config = json.loads((root / "workflow.json").read_text(encoding="utf-8"))
        stage = config.get("calculation")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        entry = (state.get("stages") or {}).get(stage) or {}
        if not entry.get("directory"):
            raise ValueError("Workflow checkpoint has no current calculation directory")
        directory = (root / entry["directory"]).resolve()
        if root not in directory.parents:
            raise ValueError("Calculation directory is outside this task")
        return directory
    outputs = {path.parent for path in (root / "runs").rglob("vasprun.xml*")
               if path.name in {"vasprun.xml", "vasprun.xml.gz"}}
    if len(outputs) > 1:
        raise ValueError("Multiple attempts without checkpoint; refusing ambiguous result")
    return next(iter(outputs)) if outputs else root


def finalize_vasp(directory, *, exit_code=0, calculation_directory=None, runtime=None):
    root = Path(directory).resolve(); task = json.loads((root / "task.json").read_text(encoding="utf-8"))
    try:
        calculation = Path(calculation_directory).resolve() if calculation_directory else resolve_calculation_directory(root)
        result = parse_vasp_result(calculation, exit_code=exit_code)
    except (ValueError, OSError) as error:
        result = {"status": "failed", "converged": False, "actual_cost": None,
                  "error": f"Result directory resolution failed: {error}", "outputs": {}}
        calculation = root
    result.update({"structure_id": task.get("structure_id"), "stage": task.get("stage")})
    outputs = result.get("outputs") or {}
    structure_path = outputs.get("structure_path")
    if structure_path:
        structure = Path(structure_path)
        if not structure.is_absolute():
            structure = calculation / structure
        result["outputs"] = {**outputs, "structure_path": str(structure)}
        if structure.is_file():
            result["outputs"]["structure_checksum"] = file_checksum(structure)
    keys = ("task_id", "task_key", "batch_id", "config_version", "model_version",
            "protocol_version", "input_file_version")
    result.update({key: task.get(key) for key in keys})
    result["task_checksum"] = task.get("task_checksum") or payload_checksum(task)
    if runtime is not None:
        result["runtime_observation"] = runtime
    else:
        import os
        import time
        from execution_layer.cost.runtime_observation import runtime_observation
        try:
            epoch = float(os.environ.get("PHASE_TASK_STARTED_EPOCH", ""))
        except ValueError:
            epoch = None
        if epoch is not None and 0 < epoch <= time.time():
            result["runtime_observation"] = runtime_observation(
                (time.monotonic() - (time.time() - epoch), epoch), task, result)
    result_path = root / "result.json"; _write(result_path, result)
    marker = {key: result.get(key) for key in (*keys, "task_checksum", "status")}
    marker.update({"result_file": result_path.name, "result_checksum": file_checksum(result_path)})
    _write(root / "task.finished.json", marker)
    export_batch_result(root, root.parent.parent / "results")
    return result


def _write(path, payload):
    temporary = Path(f"{path}.tmp"); temporary.write_text(json.dumps(payload, ensure_ascii=False,
        indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"); temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--directory", required=True)
    parser.add_argument("--exit-code", type=int, default=0); args = parser.parse_args(argv)
    result = finalize_vasp(args.directory, exit_code=args.exit_code)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__": raise SystemExit(main())
