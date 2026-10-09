"""Discuss first, then replace unsubmitted training inputs with rollback."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import uuid


CONFIRM = "确认重新生成原轮微调输入"


def is_finetune_regeneration_request(message, state):
    text = str(message).lower().replace(" ", "")
    if not any(word in text for word in ("重新生成", "重新准备", "重建")):
        return False
    if any(word in text for word in ("mc", "dft", "relax", "branch", "蒙特卡洛", "分支", "弛豫")):
        return False
    if any(word in text for word in ("微调", "训练", "finetune")):
        return True
    return "原轮输入" in text and bool(_candidate_jobs(state))


def _candidate_jobs(state):
    version = (state.get("active_model") or {}).get("version") or state.get("active_model_version")
    return {key: job for key, job in (state.get("remote_finetune_jobs") or {}).items()
            if job.get("status") in {"inputs_prepared", "awaiting_remote_training"}
            and (not version or job.get("original_model_version") == version)}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _context(state, config):
    return _fingerprint([state.get("active_model"), state.get("active_model_version"),
        state.get("new_dft_records"), state.get("dft_training_records"), config])


def _files(directory):
    files = {}
    for path in directory.rglob("*"):
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError("训练目录含链接，不能自动替换")
        if path.is_dir():
            continue
        relative = path.relative_to(directory)
        if relative.parts[0] == "inputs":
            relative = Path(*relative.parts[1:])
        root_names = {"GPU.sh", "run_training.sh", "collect_training_results.py", "training_plan.json", "UPLOAD_AND_SUBMIT.md"}
        valid = (len(relative.parts) == 1 and path.name in root_names) or (
            len(relative.parts) == 2 and (relative.parts[0] == "_shared_data"
            or re.fullmatch(r"com_\d+|main_cv_\d+", relative.parts[0]))
            and path.name in {"train.xyz", "valid.xyz", "test.xyz", "mace_train.yaml", "train_bootstrap.xyz"})
        if not valid:
            raise ValueError(f"目录含结果、日志或未知文件：{relative}；不能自动替换")
        files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def plan_finetune_regeneration(state, config):
    jobs = _candidate_jobs(state)
    conflict = state.get("finetune_input_conflict") or {}
    key = conflict.get("job_key") if conflict.get("job_key") in jobs else None
    if key is None:
        if len(jobs) != 1:
            raise ValueError("未完成微调输入不唯一，请指定训练轮次；未执行")
        key = next(iter(jobs))
    job = jobs[key]
    if job.get("submitted") or job.get("activated"):
        raise ValueError("旧训练已提交或激活，不能重生成输入")
    directory = Path(job["directory"]).resolve()
    root = Path(config["upload_batches_directory"]).resolve()
    if root not in directory.parents or not re.fullmatch(r"MLIP-finetune-round-\d+", directory.name):
        raise ValueError("训练目录不在当前提交根目录内或轮次名无效")
    if directory.exists() and not directory.is_dir():
        raise ValueError("训练路径不是目录，不能替换")
    return {"job_key": key, "directory": str(directory), "exists": directory.is_dir(),
            "files": _files(directory) if directory.is_dir() else {},
            "job_fingerprint": _fingerprint(job), "context_fingerprint": _context(state, config)}


def regenerate_finetune_inputs(state, config, approved):
    if plan_finetune_regeneration(state, config) != approved:
        raise ValueError("方案、数据、配置或旧输入已改变，请重新查看方案并确认")
    current = deepcopy(state)
    directory = Path(approved["directory"])
    backup = directory.with_name(directory.name + ".inputs-backup-" + uuid.uuid4().hex[:12])
    moved = False
    if directory.exists():
        directory.rename(backup)
        moved = True
    try:
        current["remote_finetune_jobs"][approved["job_key"]]["status"] = "abandoned"
        from execution_layer.workflows.prepare_remote_finetune import prepare_remote_finetune
        result = prepare_remote_finetune(current, config, replacement_directory=directory)
        if result["status"] != "awaiting_remote_training":
            raise ValueError(result.get("reason", "微调输入未准备好"))
    except Exception:
        if directory.exists():
            directory.rename(directory.with_name(directory.name + ".failed-inputs-" + uuid.uuid4().hex[:12]))
        if moved:
            backup.rename(directory)
        raise
    result["state"].pop("pending_finetune_regeneration", None)
    result["state"].pop("finetune_input_conflict", None)
    result["state"].setdefault("finetune_regeneration_history", []).append(
        {"old_job_key": approved["job_key"], "directory": str(directory),
         "backup_directory": str(backup) if moved else None})
    return result
