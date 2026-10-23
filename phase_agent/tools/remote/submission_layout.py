"""Compact calculation paths; frozen submission payloads keep their identities."""

from pathlib import Path
import os
import re


def compact_calculation_directory(root, directory):
    root, directory = Path(root), Path(directory)
    parts = directory.relative_to(root).parts
    if len(parts) < 3 or not parts[0].startswith("epoch"):
        return directory
    labels = []
    for part in parts[1:]:
        match = re.fullmatch(r"Search-group-(\d+)", part)
        if match:
            labels.append(f"G{int(match[1]):04d}")
            continue
        match = re.fullmatch(r"DFT-round-(\d+)_.+", part)
        if match:
            labels.append(f"DFT-{int(match[1]):04d}")
            continue
        if part in {"Relax-screening", "MC-search"} and len(parts) > 3:
            continue
        labels.append(part)
    return root / parts[0] / "_".join(labels)


def submission_moves(root, state):
    root = Path(root).resolve()
    directories = set()
    for batch in state.get("slurm_batches") or []:
        upload = Path(batch["upload_directory"]).absolute()
        if upload.parent.name != "inputs":
            raise ValueError(f"Unexpected batch layout: {upload}")
        directories.add(upload.parent.parent)
    for job in (state.get("remote_finetune_jobs") or {}).values():
        if job.get("directory"):
            directories.add(Path(job["directory"]).absolute())
    moves = []
    targets = set()
    for old in sorted(directories):
        if not old.is_relative_to(root) or not old.resolve().is_relative_to(root):
            raise ValueError(f"Submission outside project: {old}")
        new = compact_calculation_directory(root, old)
        if old == new:
            continue
        if not old.is_dir():
            raise FileNotFoundError(old)
        if new.exists() or new in targets:
            raise FileExistsError(new)
        targets.add(new)
        moves.append({"from": str(old), "to": str(new)})
    return moves


def index_operation_path(root, directory):
    """Read same-project pre-migration history without relocating scientific files."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    if directory.is_relative_to(root):
        return directory
    if root.name == "submissions" and root.parent.parent.name == "diagrams":
        project = root.parent.parent.parent
        legacy = (project / "submissions").resolve()
        # An alias into another diagram must not grant cross-diagram access.
        if not legacy.is_relative_to((project / "diagrams").resolve()) and directory.is_relative_to(
            legacy
        ):
            return directory
    raise ValueError(f"Submission record belongs outside this project/diagram: {directory}")


def _index_link(path, root, label=None, *, missing="当前无目录"):
    path = Path(path).resolve()
    relative = Path(os.path.relpath(path, root)).as_posix()
    if not path.exists():
        return f"`{relative}/`（{missing}）"
    ending = "/" if path.is_dir() else ""
    return f"[{label or path.name}]({relative}{ending})"


def write_submission_index(root, state):
    root = Path(root).resolve()
    rows = {}
    for batch in state.get("slurm_batches") or []:
        upload = Path(batch["upload_directory"])
        operation = index_operation_path(root, upload.parent.parent)
        rows.setdefault(operation, []).append(batch)
    for job in (state.get("remote_finetune_jobs") or {}).values():
        if job.get("directory"):
            operation = index_operation_path(root, job["directory"])
            rows.setdefault(operation, [])
    lines = [
        "# 提交与回收",
        "",
        "按 epoch0、epoch1 分轮保存；transfers 是普通上传/回传，relax_mc 是弛豫与蒙特卡洛批次，verified_results 是已校验结果，recovery 是恢复资料，storage_archives 是完整归档。每个 epoch 的 README 说明模型与各目录用途。",
        "",
        "无需手动解压：总提交脚本自动解压输入，本地 Agent 自动校验、解压回传包并读取。",
        "",
        "上传：使用下方本次交互文件夹，整体上传后执行一次 sbatch submit_all.sh；不要重复提交历史任务。",
        "",
        "回收：把远端对应批次的 _output 文件夹整体传回下方本地回传目录（旧包仍为 return-results），保留压缩包即可，再说“继续”。旧提交仍支持原 results/transfer 路径。",
        "",
        "Agent 读取已有归档时自动恢复所需文件；工作目录和旧路径兼容入口隐藏，平时只需使用各 epoch 的配对上传/回传目录和本索引。工作目录链接可直接打开。",
        "",
        "G0001 表示搜索组；Relax=结构弛豫，MC=蒙特卡洛搜索，DFT=第一性原理计算，MLIP-finetune=模型微调。",
        "",
        "| 计算目录 / 解压目标 | 批次数 | 输入包 | 结果包 |",
        "| --- | ---: | --- | --- |",
    ]
    for operation, batches in sorted(rows.items()):
        relative = Path(os.path.relpath(operation, root)).as_posix()
        from phase_agent.tools.remote.submission_storage import storage_archive

        archive = storage_archive(operation / "results", "results")
        returned = (
            f"[{archive.name}]({Path(os.path.relpath(archive, root)).as_posix()})"
            if archive.is_file()
            else _index_link(operation / "results", root, "results", missing="暂无结果目录")
        )
        inputs = storage_archive(operation / "inputs", "inputs")
        uploaded = (
            f"[{inputs.name}]({Path(os.path.relpath(inputs, root)).as_posix()})"
            if inputs.is_file()
            else _index_link(operation / "inputs", root, "inputs", missing="暂无输入目录")
        )
        operation_link = _index_link(operation, root, relative)
        lines.append(f"| {operation_link} | {len(batches)} | {uploaded} | {returned} |")
    handoffs = {
        r["handoff"]["folder"]: r["handoff"]
        for r in state.get("slurm_batches", [])
        if r.get("handoff")
    }
    handoffs.update(
        {
            r["handoff"]["folder"]: r["handoff"]
            for r in (state.get("remote_finetune_jobs") or {}).values()
            if r.get("handoff")
        }
    )
    for job in (state.get("remote_finetune_jobs") or {}).values():
        for transfer in (job.get("transfer_handoffs") or {}).values():
            handoffs[transfer["folder"]] = transfer
    if handoffs:
        guide = [
            "## 本次交互文件夹",
            "",
            "上传整个交互文件夹（inputs.zip、dispatch_tasks.py、submit_all.sh）到超算同一个目录，执行 `sbatch submit_all.sh`。提交脚本进入计算节点后自动解压并派发本轮任务。已经手动提交的任务不要重交。",
            "",
            "| 交互文件夹 | 作业数 | 回传本地目录 |",
            "| --- | ---: | --- |",
        ]
        for row in handoffs.values():
            relative = os.path.relpath(Path(row["folder"]).resolve(), root).replace("\\", "/")
            returned = Path(row.get("return_directory") or row["results_directory"]).resolve()
            folder_link = _index_link(
                row["folder"], root, relative, missing="上传包目录缺失，先核对归档"
            )
            guide.append(f"| {folder_link} | {row['job_count']} | {returned} |")
        guide.extend(
            [
                "",
                "结束后把远端对应批次 _output 文件夹中的压缩包传到上表对应目录（旧包仍为 return-results），回复‘继续’，无需手动解压。下方完整输入包为归档，勿重复提交历史作业。",
                "",
            ]
        )
        lines[2:2] = guide
    bundles = list((state.get("relax_mc_batches") or {}).values())
    if bundles:
        guide = [
            "## Relax / MC 批次",
            "",
            "只使用 _input 文件夹上传，回传放入对应 _output；内部运行代码已封存在 inputs.zip。历史批次勿重复提交。verified 保留已校验结果，_recovery 保留恢复绑定，不作为上传入口。",
            "",
            "| 批次 | 范围 | 状态 | 上传目录 | 回传目录 |",
            "| --- | --- | --- | --- | --- |",
        ]
        from phase_agent.tools.relax_mc_batch.handoff import input_directory, output_directory

        cancelled = {}
        for task in state.get("tasks") or []:
            if task.get("recovery_cancelled") is True:
                bid = task.get("relax_mc_bundle_id")
                cancelled[bid] = cancelled.get(bid, 0) + 1
        for row in sorted(bundles, key=lambda row: row.get("package_label") or row["bundle_id"]):
            directory = index_operation_path(root, row["directory"])
            uploaded = _index_link(input_directory(directory), root, missing="无上传目录")
            returned = _index_link(output_directory(directory), root)
            scope = (
                "当前模型"
                if row.get("model_version") == state.get("active_model_version")
                else "历史"
            )
            count = cancelled.get(row["bundle_id"], 0)
            status = (
                f"停止等待 {count} 个 MC；已有结果保留"
                if count
                else {
                    "completed": "已回收",
                    "prepared": "输入已准备",
                    "failed": "回传待核对",
                    "cancelled": "已取消",
                    "superseded": "已替代",
                }.get(row.get("status"), "待核对")
            )
            guide.append(
                f"| {row.get('package_label') or directory.name} | {scope} | {status} | {uploaded} | {returned} |"
            )
        lines[2:2] = [*guide, ""]
    path = root / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    from phase_agent.analysis.feedback.workspace_guide import publish_epoch_guides

    publish_epoch_guides(root, state)
    return path
