"""Select the dependency closure needed by existing local scientific importers."""

from pathlib import Path
import json

from phase_agent.tools.relax_mc_batch.files import rebase


def compact_state(state):
    return {
        key: state[key]
        for key in (
            "tasks",
            "mc_second_round_allocations",
            "mc_second_round_allocation",
            "active_model_version",
            "confirmed_config_version",
        )
        if key in state
    }


def result_files(root, state, candidates):
    root = Path(root).resolve()
    files = set()
    result_dirs = {}
    worker_dirs = {}

    def include(value, base, *, required=False):
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        if not path.is_relative_to(root):
            raise ValueError("return_artifact_outside_batch:" + value)
        if not path.is_file():
            if required:
                raise FileNotFoundError("required_return_artifact:" + str(path))
            return None
        files.add(path)
        return path

    path_keys = {
        "structure_path",
        "final_structure_path",
        "full_na_structure_path",
        "full_na_structure",
        "source_path",
        "checkpoint",
        "training_file",
        "mlip_result_file",
    }

    def referenced(value, base, *, required=False):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in path_keys and isinstance(item, str):
                    path = Path(item)
                    if (
                        (key == "checkpoint" and path.suffix.lower() in {".json", ".gz", ".bin"})
                        or path.suffix.lower() in {".vasp", ".poscar", ".bin"}
                        or path.name
                        in {
                            "CONTCAR",
                            "checkpoint.json",
                            "checkpoint.json.gz",
                            "training.json",
                            "mlip_result.json",
                        }
                    ):
                        include(item, base, required=required)
                elif isinstance(item, (dict, list)):
                    referenced(item, base, required=required)
        elif isinstance(value, list):
            for item in value:
                referenced(item, base, required=required)

    for task in state.get("tasks") or []:
        # A controller may fail before collecting already finished worker files.
        completed = task.get("status") == "completed"
        path = Path(task.get("result_path") or "")
        path = path if path.is_absolute() else root / path
        if path.name != "result.json":
            path = path.parent / "result.json" if path.suffix else path / "result.json"
        result = include(str(path), root, required=completed)
        if result is None:
            continue
        directory = result.parent
        result_dirs[task.get("task_id")] = directory
        worker = Path(task.get("input_path") or directory / "task.json")
        worker = (worker if worker.is_absolute() else root / worker).parent
        if not worker.resolve().is_relative_to(root):
            raise ValueError("return_worker_outside_batch")
        worker_dirs[task.get("task_id")] = worker
        include("task.finished.json", directory, required=completed)
        raw = json.loads(result.read_text(encoding="utf-8"))
        # Read worker-relative pointers against that result, without modifying
        # signed result bytes. Portable task/candidate pointers are batch-relative.
        raw_outputs = raw.get("outputs") or {}
        # Canonical collector paths in task.outputs replace worker-side geometry
        # pointers. Preserve signed raw JSON, but do not copy both geometries.
        referenced(
            {
                k: v
                for k, v in raw_outputs.items()
                if k in {"training_file", "mlip_result_file", "checkpoint"}
            },
            directory,
            required=completed,
        )
        referenced({"checkpoint": raw.get("checkpoint")}, directory, required=completed)
        for name in ("mc_occupations.bin", "checkpoint.json", "checkpoint.json.gz"):
            include(name, directory)
        outputs = task.get("outputs") or {}
        for row in outputs.get("pool_summary") or raw_outputs.get("pool_summary") or []:
            name = row.get("structure_file")
            if name:
                value = include(name, directory)
                if value is None:
                    value = include(name, directory / "pool")
                if value is None:
                    include(name, worker / "pool", required=completed)
        portable = rebase(task, root, portable=True)
        referenced(portable.get("outputs") or {}, root, required=completed)
        if task.get("stage") == "deep_search" and not (directory / "mc_occupations.bin").is_file():
            referenced({"structure_path": portable.get("structure_path")}, root)

    referenced(rebase(candidates, root, portable=True), root, required=True)
    for row in candidates.get("mc_pool") or []:
        name = row.get("structure_file")
        directory = result_dirs.get(row.get("source_task_id"))
        if name and directory is not None:
            selected = include(name, directory) or include(name, directory / "pool")
            if selected is None:
                selected = include(name, worker_dirs[row["source_task_id"]] / "pool", required=True)
            row["structure_path"] = str(selected)
    # Small allocation receipts are needed for the existing GPU usage audit.
    files.update((root / "jobs").rglob("*.runtime.json"))
    return sorted(files)
