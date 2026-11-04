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


def result_files(root, state, candidates, *, remote_root=None):
    root = Path(root).resolve()
    files = set()

    def include(value, base, *, required=False):
        if not isinstance(value, str) or not value:
            return None
        if remote_root and value.startswith(remote_root.rstrip("/") + "/"):
            value = str(root / value[len(remote_root.rstrip("/")) + 1 :])
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
        worker = Path(task.get("input_path") or directory / "task.json")
        worker = (worker if worker.is_absolute() else root / worker).parent
        if not worker.resolve().is_relative_to(root):
            raise ValueError("return_worker_outside_batch")
        include("task.finished.json", directory, required=completed)
        raw = json.loads(result.read_text(encoding="utf-8"))
        # Read worker-relative pointers against that result, without modifying
        # signed result bytes. Portable task/candidate pointers are batch-relative.
        raw_outputs = raw.get("outputs") or {}
        for item in raw_outputs.get('extrapolation_failures') or []:
            from hashlib import sha256
            name = Path(str(item.get('structure_path') or '').replace('\\', '/')).name
            if not name.startswith('extrapolation_') or not name.endswith('.vasp'):
                raise ValueError('invalid_extrapolation_failure_frame')
            frame = next((folder / name for folder in (directory, worker)
                          if (folder / name).is_file()), None)
            if frame is None or sha256(frame.read_bytes()).hexdigest() != item.get('structure_checksum'):
                raise ValueError('missing_or_invalid_extrapolation_failure_frame')
            include(str(frame), root, required=True)
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
        # A pool summary is statistics, not an export selection. Uncollected
        # results still need their final geometry, including on controller failure.
        if not (outputs.get("structure_path") or outputs.get("final_structure_path")):
            geometry = raw_outputs.get("structure_path") or raw_outputs.get("final_structure_path")
            if geometry:
                name = Path(str(geometry).replace("\\", "/")).name
                selected = None
                for folder in (directory, directory / "pool", worker / "pool", worker):
                    selected = include(name, folder)
                    if selected is not None:
                        break
                if selected is None:
                    include(
                        geometry, directory, required=completed or raw.get("status") == "completed"
                    )
        portable = rebase(task, root, portable=True)
        referenced(
            {k: v for k, v in (portable.get("outputs") or {}).items() if k != "pool_summary"},
            root,
            required=completed,
        )
        if task.get("stage") == "deep_search" and not (directory / "mc_occupations.bin").is_file():
            referenced({"structure_path": portable.get("structure_path")}, root)

    referenced(
        rebase(candidates.get("branch_candidates") or [], root, portable=True),
        root,
        required=True,
    )
    # Small allocation receipts are needed for the existing GPU usage audit.
    files.update((root / "jobs").rglob("*.runtime.json"))
    return sorted(files)
