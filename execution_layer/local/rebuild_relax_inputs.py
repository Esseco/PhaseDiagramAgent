"""Explicit approved cleanup of unsubmitted generated Relax inputs only."""
from copy import deepcopy
from pathlib import Path
import json
import shutil


def is_relax_rebuild_request(message):
    text = str(message or "").lower().replace(" ", "")
    return any(word in text for word in ("重新", "重建", "重做")) and (
        "输入" in text or "文件" in text) and any(word in text for word in ("mlip", "relax", "弛豫"))


def plan_relax_rebuild(state, root):
    root = Path(root).resolve()
    batches = {row.get("batch_id"): row for row in state.get("slurm_batches") or []}
    tasks = [r for r in state.get("tasks") or [] if r.get("stage") == "relax_and_feature"
             and r.get("input_path") and r.get("status") != "completed"]
    ids = {r["task_id"] for r in tasks}
    directories = set()
    for row in tasks:
        if row.get("status") in {"running", "submitted"} or row.get("job_id"):
            raise ValueError("任务已提交或运行，请先取消；不删除输入")
        directory = Path(row["input_path"]).resolve().parent.parent
        batch = batches.get(row.get("batch_id") or row.get("slurm_batch_id")) or {}
        recorded = Path(batch.get("upload_directory") or directory).resolve()
        legacy_flat = directory.parent == root and directory.name.startswith("remote-")
        if root not in directory.parents or (directory != recorded and not legacy_flat):
            raise ValueError("清理目录不在本次上传根目录内")
        directories.add(directory)
    allowed = {"manifest.json", "GPU.sh", "run_mlip_batch.py", "run_mlip_task.py", "task.json",
               "initial.vasp", "full_na_structure.vasp", "SHA256SUMS", "UPLOAD_AND_SUBMIT.md", "submit.sbatch"}
    for directory in directories:
        if not directory.exists():
            continue
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if any(r.get("task_id") not in ids for r in manifest):
            raise ValueError("批次含其他或已完成任务，禁止整批删除")
        if any(r.get("job_id") for r in state.get("slurm_batches") or []
               if Path(r.get("upload_directory") or "").resolve() == directory):
            raise ValueError("批次已有远端job_id，请先取消")
        for path in directory.rglob("*"):
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or root not in path.resolve().parents:
                raise ValueError("链接或越界路径，禁止清理")
            if path.is_file() and path.name not in allowed:
                raise ValueError("存在计算结果、日志或未知文件，需单独审核；未删除")
    return {"directories": sorted(map(str, directories)), "task_ids": sorted(ids)}


def rebuild_relax_inputs(state, root, approved_plan):
    plan = plan_relax_rebuild(state, root)
    if plan != approved_plan:
        raise ValueError("清理范围变化，请重新确认")
    current = deepcopy(state)
    for directory in plan["directories"]:
        path = Path(directory)
        if path.exists():
            shutil.rmtree(path)
    for row in current.get("tasks") or []:
        if row.get("task_id") in plan["task_ids"]:
            for key in ("slurm_batch_id", "slurm_array_index", "batch_id", "input_path", "result_path", "task_checksum"):
                row.pop(key, None)
            row["status"] = "pending"
    for row in current.get("slurm_batches") or []:
        if set(row.get("task_ids") or []) & set(plan["task_ids"]):
            row["status"] = "superseded_inputs"
    current.setdefault("input_rebuild_history", []).append(plan)
    return current
