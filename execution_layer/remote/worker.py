"""Offline worker that atomically publishes a verifiable completion marker."""
import json
from pathlib import Path
import traceback
from execution_layer.remote.integrity import file_checksum
from execution_layer.cost.runtime_observation import start_timer, runtime_observation


def run_remote_task(manifest_path, array_index, *, executor):
    manifest_path = Path(manifest_path).resolve(); root = manifest_path.parent
    entry = next(row for row in json.loads(manifest_path.read_text(encoding="utf-8"))
                 if int(row["array_index"]) == int(array_index))
    task = json.loads((root / entry["input_path"]).read_text(encoding="utf-8"))
    task["calculation_directory"] = str(root / task["calculation_directory"]); task["result_path"] = str(root / entry["result_path"])
    timer = start_timer()
    try:
        result = executor(task)
        if result.get("status") not in {"completed", "failed", "timeout", "cancelled"}:
            raise ValueError("remote executor must return a terminal status")
        payload = dict(result)
    except Exception as error:
        payload = {"status": "failed", "actual_cost": None, "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()}
    payload.update({key: entry.get(key) for key in ("task_id", "task_key", "batch_id", "config_version",
                                                     "model_version", "task_checksum", "protocol_version",
                                                     "input_file_version")})
    payload["runtime_observation"] = runtime_observation(timer, task, payload)
    result_path = Path(task["calculation_directory"]) / "result.json"
    _write(result_path, payload)
    marker = {key: payload.get(key) for key in ("task_id", "task_key", "batch_id", "config_version",
                                                "model_version", "task_checksum", "protocol_version",
                                                "input_file_version", "status")}
    marker.update({"result_file": result_path.name, "result_checksum": file_checksum(result_path)})
    _write(result_path.with_name("task.finished.json"), marker)
    published_path = (root / entry["result_path"]).resolve()
    if published_path != result_path.resolve():
        from execution_layer.remote.export_batch_result import export_batch_result
        if published_path.parent.name != result_path.parent.name or published_path.name != "result.json":
            raise ValueError("remote result destination must retain the task directory name")
        export_batch_result(result_path.parent, published_path.parent.parent)
    return payload


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True); temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
