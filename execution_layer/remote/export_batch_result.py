"""Publish one finished calculation into its batch's downloadable results folder."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from execution_layer.remote.integrity import file_checksum
from execution_layer.remote.result_directory_name import result_directory_name


def export_batch_result(task_directory, results_directory):
    task_dir = Path(task_directory).resolve()
    result_path = task_dir / "result.json"
    marker_path = task_dir / "task.finished.json"
    if not result_path.is_file() or not marker_path.is_file():
        raise FileNotFoundError(f"任务终态结果不完整：{task_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("result_checksum") != file_checksum(result_path):
        raise ValueError(f"任务结果校验失败：{task_dir}")
    if (marker.get("task_id") != result.get("task_id")
            or marker.get("task_key") != result.get("task_key")
            or marker.get("status") != result.get("status")):
        raise ValueError(f"任务身份不一致：{task_dir}")

    destination = Path(results_directory) / result_directory_name(task_dir, result.get("stage"))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "task.finished.json").unlink(missing_ok=True)
    outputs = result.get("outputs") or {}
    if outputs.get("mlip_result_file"):
        from execution_layer.remote.dft_mlip_pair import load_pair
        load_pair(result, task_dir)
        _copy(task_dir / "mlip_result.json", destination / "mlip_result.json")
    training_file = outputs.get("training_file")
    if training_file:
        source = (task_dir / training_file).resolve()
        if task_dir not in source.parents or source.name != "training.json":
            raise ValueError("Training file must be training.json within task directory")
        if not source.is_file() or outputs.get("training_checksum") != file_checksum(source):
            raise ValueError("Training file missing or checksum mismatch")
        _copy(source, destination / "training.json")
    structure = outputs.get("structure_path") or outputs.get("final_structure_path")
    if result.get("stage") in {"dft_single_point", "dft_relax"} and outputs.get("structure") is not None:
        structure = None
    if structure:
        source = Path(str(structure).replace("\\", "/"))
        if not source.is_absolute():
            source = task_dir / source
        source = source.resolve()
        valid_name = source.suffix.lower() in {".vasp", ".poscar"} or source.name == "CONTCAR"
        if task_dir not in source.parents or not valid_name:
            if result.get("status") == "completed":
                raise ValueError(f"最终结构不在任务目录内：{structure}")
        else:
            if not source.is_file():
                if result.get("status") == "completed":
                    raise FileNotFoundError(source)
            else:
                expected = outputs.get("structure_checksum")
                if expected and file_checksum(source) != expected:
                    raise ValueError(f"最终结构校验失败：{source}")
                _copy(source, destination / source.name)
    checkpoint = result.get("checkpoint")
    if checkpoint:
        checkpoint_path = Path(str(checkpoint).replace("\\", "/"))
        source = checkpoint_path if checkpoint_path.is_absolute() else task_dir / checkpoint_path
        if source.is_file() and task_dir in source.resolve().parents:
            _copy(source, destination / source.name)
    # Checkpoints still participate in the existing resume contract; do not
    # discard them until remote-only restart has an explicit adapter.
    for name in ("checkpoint.json", "checkpoint.json.gz"):
        source = task_dir / name
        if source.is_file():
            _copy(source, destination / name)
    remote_artifacts = []
    for name in ("status.json.gz", "settings.json.gz", "trace.json.gz",
                 "task.stdout.log", "task.stderr.log", "pool/pool_summary.json.gz"):
        source = task_dir / name
        if source.is_file():
            remote_artifacts.append({"name": name, "remote_path": str(source.resolve()),
                                     "size_bytes": source.stat().st_size,
                                     "required_for_recovery": False})
    if remote_artifacts:
        manifest = destination / "remote_artifacts.json"
        temporary = manifest.with_name(manifest.name + ".part")
        temporary.write_text(json.dumps({"task_id": result.get("task_id"),
            "artifacts": remote_artifacts}, indent=2), encoding="utf-8")
        temporary.replace(manifest)
    _copy(result_path, destination / "result.json")
    # The completion marker is published last, after every required file.
    _copy(marker_path, destination / "task.finished.json")
    return destination


def _copy(source, destination):
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
