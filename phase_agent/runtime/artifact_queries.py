"""Read-only answers about registered artifacts, before workflow dispatch."""

from pathlib import Path


def finetune_location_reply(message, state):
    text = str(message).lower().replace(" ", "").strip()
    if not any(word in text for word in ("微调", "训练", "finetune", "committee")):
        return None
    if not any(word in text for word in ("在哪", "哪里", "位置", "路径", "目录", "文件夹")):
        return None
    if any(
        word in text
        for word in ("生成", "重建", "删除", "修改", "移动", "提交", "运行", "批准", "同意", "拒绝")
    ):
        return None
    jobs = [
        job for job in (state.get("remote_finetune_jobs") or {}).values() if job.get("directory")
    ]
    if not jobs:
        return "尚无已登记的微调输入目录。本次仅查询，未生成文件。"
    version = (state.get("active_model") or {}).get("version") or state.get("active_model_version")
    if version:
        jobs = [job for job in jobs if job.get("original_model_version") == version]
    if not jobs:
        return "当前模型尚无已登记的微调输入目录。本次仅查询，未生成文件。"
    lines = ["已登记的微调输入："]
    conflict = state.get("finetune_input_conflict") or {}
    for job in jobs:
        path = str(job["directory"])
        suffix = "目录存在" if Path(path).is_dir() else "目录不存在"
        if path == conflict.get("directory"):
            suffix += "；参数或数据已变化，旧输入待确认重生成"
        lines.append(f"- `{path}`（{suffix}）")
    lines.append("本次仅查询，未批准、生成或提交任务。")
    return "\n".join(lines)
