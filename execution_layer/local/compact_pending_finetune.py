"""Explicitly retire missing, unsubmitted drafts and repair one pending round."""
from datetime import datetime
import json
from pathlib import Path
import shutil

from execution_layer.local.migrate_workspace_layout import require_agent_offline, rewrite_active_paths
from execution_layer.remote.migrate_legacy_upload_layout import _atomic_write


def compact_pending_finetune(workspace, *, apply=False, agent_port=8765):
    root = Path(workspace).resolve()
    from config_layer.session.workspace_active_files import workspace_active_files
    state_path = workspace_active_files(root)["state"]
    modern = state_path.parent.name == "workflow_state"
    submission_root = root / ("submissions" if modern else "upload_batches")
    original = state_path.read_bytes()
    state = json.loads(original)
    jobs = state.get("remote_finetune_jobs") or {}
    pending = [(key, job) for key, job in jobs.items()
               if job.get("status") in {"inputs_prepared", "awaiting_remote_training"}]
    existing = [(key, job) for key, job in pending if Path(job["directory"]).is_dir()]
    missing = [(key, job) for key, job in pending if not Path(job["directory"]).exists()]
    if not missing:
        return {"status": "unchanged"}
    if len(existing) != 1 or len(pending) != len(jobs):
        raise ValueError("仅支持一套有效待提交输入及未执行旧草稿；不能重编号真实训练历史")
    if any(job.get("submitted") or job.get("activated") or job.get("job_id") for _, job in pending):
        raise ValueError("存在提交或激活证据，不能重编号")
    key, job = existing[0]
    if any(old.get("original_model_version") != job.get("original_model_version") for _, old in missing):
        raise ValueError("旧草稿属于其他模型，不能自动归并")
    source = Path(job["directory"]).resolve()
    target = source.with_name("MLIP-finetune-round-0001")
    if not source.is_relative_to(submission_root) or source.is_symlink():
        raise ValueError("训练目录不在指定工作区")
    if target.exists() or any(p.is_symlink() for p in source.rglob("*")):
        raise ValueError("目标存在或目录含链接，不能覆盖")
    for _, old in missing:
        if not Path(old["directory"]).resolve().is_relative_to(submission_root):
            raise ValueError("旧草稿路径超出工作区")
    result = {"status": "planned", "source": str(source), "target": str(target),
              "retired_drafts": [k for k, _ in missing]}
    if not apply:
        return result
    require_agent_offline(agent_port)
    backup = root / ("history_backups/layout_cleanup" if modern else "backups/layout_cleanup") / datetime.now().strftime("%Y%m%dT%H%M%S%f")
    backup.mkdir(parents=True)
    shutil.copy2(state_path, backup / "state.json")
    _atomic_write(backup / "retired_finetune_drafts.json", dict(missing))
    for old_key, _ in missing:
        del jobs[old_key]
    replacements = [(str(source), str(target))]
    state = rewrite_active_paths(state, replacements)
    state["remote_finetune_jobs"][key]["report"]["training_round"] = target.name
    edits = {}
    for path in source.rglob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        updated = rewrite_active_paths(value, replacements)
        if isinstance(updated, dict) and updated.get("training_round") == source.name:
            updated["training_round"] = target.name
        if updated != value:
            edits[path.relative_to(source)] = updated
            saved = backup / "metadata" / path.relative_to(source)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, saved)
    duplicate = source / "finetune_report.json"
    plan = source / "training_plan.json"
    remove_duplicate = duplicate.is_file() and plan.is_file() and duplicate.read_bytes() == plan.read_bytes()
    if remove_duplicate:
        shutil.copy2(duplicate, backup / duplicate.name)
    require_agent_offline(agent_port)
    if state_path.read_bytes() != original:
        raise RuntimeError("state已变化，未移动目录")
    shutil.move(str(source), str(target))
    try:
        for relative, value in edits.items():
            _atomic_write(target / relative, value)
        if remove_duplicate:
            (target / "finetune_report.json").unlink()
        from analysis_layer.feedback.export_dft_products import publish_output_catalog
        publish_output_catalog(state, root / ("analysis_outputs" if modern else "outputs"))
        _atomic_write(state_path, state)
        from data_layer.memory.publish_memory_views import publish_memory_views
        publish_memory_views(state, state_path)
    except Exception:
        shutil.move(str(target), str(source))
        for relative in edits:
            shutil.copy2(backup / "metadata" / relative, source / relative)
        if remove_duplicate:
            shutil.copy2(backup / duplicate.name, duplicate)
        shutil.copy2(backup / "state.json", state_path)
        raise
    result.update(status="compacted", backup=str(backup), removed_duplicates=int(remove_duplicate))
    _atomic_write(backup / "manifest.json", result)
    return result
