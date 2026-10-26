"""Compact calculation paths; frozen submission payloads keep their identities."""

from pathlib import Path
from collections import Counter
import os
import json
import re


from phase_agent.tools.remote.submission_architecture import (
    EPOCH_BUCKETS,
    submission_navigation,
    navigation_markdown,
    write_epoch_manifests,
)


def _submission_navigation():
    return submission_navigation()


def _navigation_markdown():
    return navigation_markdown()


def compact_calculation_directory(root, directory):
    root, directory = Path(root), Path(directory)
    parts = directory.relative_to(root).parts
    if len(parts) < 3 or not parts[0].startswith("epoch"):
        return directory
    if parts[1] == "calculations":
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
    return root / parts[0] / "calculations" / "_".join(labels)


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


CANONICAL_EPOCH_DIRECTORIES = frozenset(
    {
        "calculations",
        "delivery",
        "evidence",
        "transfers",
        "relax_mc",
        "verified_results",
        "recovery",
        "storage_archives",
        ".transfer_receipts",
        "previously_used",
    }
)
CANONICAL_EPOCH_FILES = frozenset({"README.md", "layout_manifest.json"})


def _registered_task_directories(root, state):
    root = Path(root).resolve()
    registered = set()

    def add(value):
        if not value:
            return
        try:
            relative = Path(value).resolve().relative_to(root)
        except ValueError:
            return
        if len(relative.parts) < 2 or not re.fullmatch(r"epoch\d+", relative.parts[0]):
            return
        epoch = root / relative.parts[0]
        if len(relative.parts) >= 3 and relative.parts[1] == "calculations":
            registered.add(epoch / "calculations" / relative.parts[2])
        elif relative.parts[1] not in CANONICAL_EPOCH_DIRECTORIES:
            registered.add(epoch / relative.parts[1])

    for batch in state.get("slurm_batches") or []:
        add(batch.get("upload_directory"))
        add(batch.get("results_directory"))
    for job in (state.get("remote_finetune_jobs") or {}).values():
        add(job.get("directory"))
    for batch in (state.get("relax_mc_batches") or {}).values():
        add(batch.get("directory"))
        add(batch.get("upload_folder"))
    return registered


def submission_cleanup_moves(root, state):
    """Plan moves for unregistered epoch siblings and the old Chinese archive name."""
    root = Path(root).resolve()
    registered = _registered_task_directories(root, state)
    moves = []
    excluded = {Path(row["from"]).resolve() for row in submission_moves(root, state)}
    legacy = root / "曾用"
    previously_used = root / "previously_used"
    if legacy.exists():
        if previously_used.exists():
            raise FileExistsError(previously_used)
        moves.append({"from": str(legacy), "to": str(previously_used)})
    for epoch in sorted(root.glob("epoch*")):
        if not epoch.is_dir() or not re.fullmatch(r"epoch\d+", epoch.name):
            continue
        for child in sorted(epoch.iterdir(), key=lambda path: path.name):
            if (
                child in excluded
                or child in registered
                or child.name in CANONICAL_EPOCH_DIRECTORIES
            ):
                continue
            if child.name in CANONICAL_EPOCH_FILES:
                continue
            target = previously_used / epoch.name / child.name
            if target.exists():
                raise FileExistsError(target)
            moves.append({"from": str(child), "to": str(target)})
        calculations = epoch / "calculations"
        if calculations.is_dir():
            for child in sorted(calculations.iterdir(), key=lambda path: path.name):
                if child in registered:
                    continue
                target = previously_used / epoch.name / "calculations" / child.name
                if target.exists():
                    raise FileExistsError(target)
                moves.append({"from": str(child), "to": str(target)})
    targets = [Path(row["to"]).resolve() for row in moves]
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate submission cleanup destination")
    for row in moves:
        source, target = Path(row["from"]).resolve(), Path(row["to"]).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root):
            raise ValueError(f"Submission cleanup escapes project: {source}")
        if source.is_symlink():
            raise ValueError(f"Submission cleanup does not move links: {source}")
        if source.is_dir() and any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(f"Submission cleanup found nested link: {source}")
    return moves


def submission_bucket_moves(root, state):
    """Plan the physical epoch buckets without changing any payload bytes."""
    root = Path(root).resolve()
    registered = _registered_task_directories(root, state)
    moves = []
    old_to_new = {
        "transfers": ("delivery", "transfers"),
        "relax_mc": ("delivery", "relax_mc"),
        "recovery": ("evidence", "recovery"),
        "verified_results": ("evidence", "verified_results"),
        "storage_archives": ("evidence", "storage_archives"),
        ".transfer_receipts": ("evidence", ".transfer_receipts"),
    }
    for epoch in sorted(root.glob("epoch*")):
        if not epoch.is_dir() or not re.fullmatch(r"epoch\d+", epoch.name):
            continue
        # Move the existing fixed namespaces under their lane buckets.
        for old_name, (bucket, child_name) in old_to_new.items():
            old = epoch / old_name
            new = epoch / bucket / child_name
            if old.exists():
                if new.exists():
                    raise FileExistsError(new)
                moves.append({"from": str(old), "to": str(new)})
        # Registered calculation workspaces belong under calculations/.
        for child in sorted(epoch.iterdir(), key=lambda path: path.name):
            if child.name in {"README.md", "layout_manifest.json", *old_to_new, *EPOCH_BUCKETS}:
                continue
            if child in registered:
                new = epoch / "calculations" / child.name
                if new.exists():
                    raise FileExistsError(new)
                moves.append({"from": str(child), "to": str(new)})
    targets = [Path(row["to"]).resolve() for row in moves]
    if len(targets) != len(set(targets)):
        raise ValueError("Duplicate submission bucket destination")
    for row in moves:
        old, new = Path(row["from"]).resolve(), Path(row["to"]).resolve()
        if not old.is_relative_to(root) or not new.is_relative_to(root):
            raise ValueError(f"Submission bucket move escapes project: {old}")
        if old.is_symlink() or (old.is_dir() and any(p.is_symlink() for p in old.rglob("*"))):
            raise ValueError(f"Submission bucket move found link: {old}")
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


def _indexed_path(root, value):
    if not value:
        return {"path": None, "relative_path": None, "exists": None}
    path = Path(value)
    if not path.is_absolute():
        path = Path(root) / path
    path = path.resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = None
    return {"path": str(path), "relative_path": relative, "exists": path.exists()}


def _handoff_kind(folder, source, stage=None):
    if source in {"finetune", "recovery"}:
        return source
    text = str(folder or "").lower().replace("\\", "/")
    if "dft" in text or str(stage or "").startswith("dft_"):
        return "dft"
    if "relaxmc" in text or "relax_mc" in text:
        return "relax_mc"
    if "modelrefresh" in text or "model-refresh" in text:
        return "model_refresh"
    if stage == "deep_search":
        return "mc"
    if stage == "relax_and_feature":
        return "relax"
    return "calculation"


def _submission_index_payload(root, state):
    tasks = {row.get("task_id"): row for row in state.get("tasks") or []}
    operations = {}
    for batch in state.get("slurm_batches") or []:
        upload = Path(batch["upload_directory"])
        operation = index_operation_path(root, upload.parent.parent)
        operations.setdefault(operation, []).append(batch)
    for job in (state.get("remote_finetune_jobs") or {}).values():
        if job.get("directory"):
            operation = index_operation_path(root, job["directory"])
            operations.setdefault(operation, [])

    from phase_agent.tools.remote.submission_storage import storage_archive

    operation_rows = []
    for operation, batches in sorted(operations.items()):
        task_ids = sorted({task_id for batch in batches for task_id in batch.get("task_ids") or []})
        stages = {}
        statuses = set()
        for batch in batches:
            stage = batch.get("stage")
            if stage:
                stages[stage] = stages.get(stage, 0) + len(batch.get("task_ids") or [])
            if batch.get("status"):
                statuses.add(batch["status"])
        input_archive = storage_archive(operation / "inputs", "inputs")
        result_archive = storage_archive(operation / "results", "results")
        operation_rows.append(
            {
                "directory": _indexed_path(root, operation),
                "status": sorted(statuses),
                "task_count": len(task_ids),
                "batch_count": len(batches),
                "stages": stages,
                "input_directory": _indexed_path(root, operation / "inputs"),
                "results_directory": _indexed_path(root, operation / "results"),
                "input_archive": _indexed_path(root, input_archive)
                if input_archive.is_file()
                else _indexed_path(root, None),
                "results_archive": _indexed_path(root, result_archive)
                if result_archive.is_file()
                else _indexed_path(root, None),
            }
        )

    handoffs = {}

    def add_handoff(handoff, *, source="calculation", status=None, stage=None):
        folder = handoff.get("folder") or (
            str(Path(handoff["upload_archive"]).parent) if handoff.get("upload_archive") else None
        )
        if not folder:
            return
        key = str(Path(folder).resolve())
        row = handoffs.setdefault(
            key,
            {
                "workflow": _handoff_kind(folder, source, stage),
                "submission_path": _indexed_path(root, folder),
                "upload_archive": _indexed_path(root, None),
                "submit_script": _indexed_path(root, None),
                "submit_command": None,
                "remote_output_folder": None,
                "return_path": _indexed_path(root, None),
                "status": [],
                "task_ids": [],
                "job_count": 0,
                "layout_version": handoff.get("layout_version"),
                "verification": handoff.get("verification"),
                "collect_only": handoff.get("collect_only", False),
            },
        )
        if handoff.get("upload_archive"):
            row["upload_archive"] = _indexed_path(root, handoff["upload_archive"])
        if handoff.get("submit_script"):
            row["submit_script"] = _indexed_path(root, handoff["submit_script"])
            row["submit_command"] = (
                handoff.get("command") or f"sbatch {Path(handoff['submit_script']).name}"
            )
        else:
            row["submit_command"] = handoff.get("command") or row["submit_command"]
        return_path = handoff.get("return_directory") or handoff.get("results_directory")
        if return_path:
            row["return_path"] = _indexed_path(root, return_path)
        if status and status not in row["status"]:
            row["status"].append(status)
        row["task_ids"] = sorted(set(row["task_ids"]) | set(handoff.get("task_ids") or []))
        row["task_status_counts"] = dict(
            Counter(
                tasks.get(task_id, {}).get("status") or "unregistered"
                for task_id in row["task_ids"]
            )
        )
        row["job_count"] = max(
            row["job_count"], int(handoff.get("job_count") or len(row["task_ids"]))
        )
        version = int(handoff.get("layout_version") or row.get("layout_version") or 0)
        if version >= 4:
            path = Path(folder)
            row["remote_output_folder"] = (
                path.name[:-6] + "_output" if path.name.endswith("_input") else None
            )
        elif version >= 2:
            row["remote_output_folder"] = "return-results"
        else:
            row["remote_output_folder"] = "results/transfer/all.results.zip"

    for batch in state.get("slurm_batches") or []:
        # Calculation batches are deliberately absent from the delivery table
        # until an explicit handoff is prepared.  Their upload/results paths
        # remain visible in `operations` and must not look submit-ready.
        if not batch.get("handoff"):
            continue
        handoff = dict(batch["handoff"])
        script = handoff.get("submit_script") or batch.get("script_path")
        if script:
            handoff["submit_script"] = script
            handoff["command"] = handoff.get("command") or f"sbatch {Path(script).name}"
        handoff["task_ids"] = batch.get("task_ids") or []
        handoff["job_count"] = handoff.get("job_count") or len(handoff["task_ids"])
        add_handoff(
            handoff,
            status=batch.get("status"),
            stage=batch.get("stage"),
        )
    for job in (state.get("remote_finetune_jobs") or {}).values():
        handoff = job.get("handoff") or {}
        add_handoff(handoff, source="finetune", status=job.get("status"))
        for transfer in (job.get("transfer_handoffs") or {}).values():
            add_handoff(transfer, source="finetune", status=job.get("status"))
    from phase_agent.tools.remote.legacy_recovery import recovery_handoffs

    for handoff in recovery_handoffs(state):
        add_handoff(handoff, source="recovery", status="recovery_prepared")
    from phase_agent.tools.relax_mc_batch.handoff import input_directory, output_directory

    relax_mc_batches = []
    for bundle_id, batch in (state.get("relax_mc_batches") or {}).items():
        directory = (
            index_operation_path(root, batch["directory"]) if batch.get("directory") else None
        )
        upload = batch.get("upload_folder") or (input_directory(directory) if directory else None)
        expected = batch.get("expected_results_archive")
        expected = Path(_indexed_path(root, expected)["path"]) if expected else None
        returned = (
            expected.parent if expected else output_directory(directory) if directory else None
        )
        compact_upload = Path(_indexed_path(root, upload)["path"]) if upload else None
        if not batch.get("recovery_original_directory"):
            # The compact driver publishes beside its registered input, even after relocation.
            if (
                compact_upload
                and compact_upload.name.endswith("_input")
                and (compact_upload / "run_upload.py").is_file()
            ):
                returned = compact_upload.with_name(compact_upload.name[:-6] + "_output")
            elif (
                directory
                and expected == directory / "returned/results.zip"
                and (output_directory(directory) / "results.zip").is_file()
            ):
                returned = output_directory(directory)
        legacy_returns = []
        if directory and output_directory(directory) != returned:
            legacy_returns.append(_indexed_path(root, output_directory(directory)))
        script = batch.get("script_path") or (
            str(Path(upload) / "submit_all.sh") if upload else None
        )
        if upload and Path(upload).name.endswith("_input"):
            paired_script = Path(upload) / "submit_all.sh"
            if paired_script.is_file():
                # A legacy state record can still point at the old full package;
                # the paired input directory is the only current submit entry.
                script = str(paired_script)
        relax_mc_batches.append(
            {
                "bundle_id": batch.get("bundle_id") or bundle_id,
                "package_label": batch.get("package_label")
                or (directory.name if directory else bundle_id),
                "model_version": batch.get("model_version"),
                "status": batch.get("status"),
                "branch_count": len(batch.get("branch_ids") or []),
                "submission_path": _indexed_path(root, upload),
                "submit_script": _indexed_path(root, script),
                "submit_command": f"sbatch {Path(script).name}" if script else None,
                "return_path": _indexed_path(root, returned),
                "legacy_return_paths": legacy_returns,
                "return_error": batch.get("return_error"),
                "input_archive": _indexed_path(root, batch.get("upload_archive")),
            }
        )

    return {
        "schema_version": 1,
        "project": root.parent.name,
        "submission_root": str(root),
        "active_model_version": state.get("active_model_version"),
        "model_rounds": (state.get("upload_layout") or {}).get("model_rounds") or {},
        "path_format": {
            "type": "object",
            "fields": {
                "path": "absolute local filesystem path, or null when unregistered",
                "relative_path": "path relative to submission_root, or null when outside it",
                "exists": "filesystem existence at index generation time; recheck before reading or submitting",
            },
        },
        "navigation": _submission_navigation(),
        "layout": {
            "root_control_files": "README.md, submission_index.json, and optional hidden archive/migration receipts",
            "epoch_directory": "epochN/",
            "calculations": "epochN/calculations/<operation>/inputs/ and results/",
            "transfers": "epochN/delivery/transfers/<batch>_input/ and <batch>_output/",
            "relax_mc": "epochN/delivery/relax_mc/RelaxMC_BatchNN_input/ and RelaxMC_BatchNN_output/",
            "recovery": "epochN/evidence/recovery/ contains collection and diagnostic material only",
            "verified_results": "epochN/evidence/verified_results/ contains validated returned evidence; never submit it",
            "storage_archives": "epochN/evidence/storage_archives/ contains retained input/result archives",
            "previously_used": "previously_used/ contains unregistered or legacy material; never submit from it",
            "legacy": "Historical or unregistered material is kept under previously_used/; use recorded task paths above.",
            "legacy_delivery_alias": "Old epochN/transfers/ and epochN/relax_mc/ names are historical aliases; current paths include delivery/.",
            "legacy_return_paths": "Historical inferred paths for reference only, never the current return destination",
        },
        "operations": operation_rows,
        "handoffs": sorted(handoffs.values(), key=lambda row: row["submission_path"]["path"] or ""),
        "relax_mc_batches": sorted(
            relax_mc_batches, key=lambda row: row["submission_path"]["path"] or ""
        ),
    }


def _submission_markdown(index, state):
    from phase_agent.tools.state.task_waiting import awaiting_task_result

    root = Path(index["submission_root"])
    active_model = index["active_model_version"]
    tasks = {task.get("task_id"): task for task in state.get("tasks") or []}
    current_paths, unfinished_paths = set(), set()

    def register(handoff, *, current, unfinished):
        folder = handoff.get("folder")
        if folder:
            path = _indexed_path(root, folder)["path"]
            if current:
                current_paths.add(path)
            if current and unfinished:
                unfinished_paths.add(path)

    def waiting(task):
        return awaiting_task_result(task) and not task.get("recovery_cancelled")

    task_models = {
        task_id: batch.get("model_version")
        for batch in state.get("slurm_batches") or []
        for task_id in batch.get("task_ids") or []
    }
    recovered_ids = set()
    recovery_rows = sorted(
        (row for row in index["handoffs"] if row["workflow"] == "recovery"),
        key=lambda row: row["submission_path"]["path"] or "",
        reverse=True,
    )
    for row in recovery_rows:
        members = [tasks[task_id] for task_id in row["task_ids"] if task_id in tasks]
        current_members = [
            task
            for task in members
            if active_model
            and (task.get("model_version") or task_models.get(task.get("task_id"))) == active_model
        ]
        pending = {task["task_id"] for task in current_members if waiting(task)} - recovered_ids
        register(
            {"folder": row["submission_path"]["path"]},
            current=bool(current_members),
            unfinished=bool(pending),
        )
        recovered_ids.update(pending)

    for batch in state.get("slurm_batches") or []:
        members = [tasks[task_id] for task_id in batch.get("task_ids") or [] if task_id in tasks]
        current = active_model and (
            batch.get("model_version") == active_model
            or any(task.get("model_version") == active_model for task in members)
        )
        unfinished = any(
            waiting(task)
            and task.get("task_id") not in recovered_ids
            and (task.get("model_version") or batch.get("model_version")) == active_model
            for task in members
        )
        register(
            batch.get("handoff") or {"folder": batch.get("upload_directory")},
            current=current,
            unfinished=unfinished,
        )
    for job in (state.get("remote_finetune_jobs") or {}).values():
        current = active_model and job.get("original_model_version") == active_model
        unfinished = job.get("status") in {
            "inputs_prepared",
            "submitted",
            "running",
            "awaiting_results",
        }
        register(job.get("handoff") or {}, current=current, unfinished=unfinished)
        for handoff in (job.get("transfer_handoffs") or {}).values():
            register(handoff, current=current, unfinished=False)
        followup = job.get("training_handoff") or {}
        if followup.get("stage") in {"awaiting_manifest", "awaiting_validation"}:
            register(followup.get("handoff") or {}, current=current, unfinished=True)

    deliveries = []
    for row in index["handoffs"]:
        path = row["submission_path"]["path"]
        status = ", ".join(row["status"]) or "未登记"
        if row["workflow"] == "recovery":
            status = (
                "仅收集已有结果；原计算勿重投"
                if row["collect_only"]
                else "回收修复包已准备；先核对原提交"
            )
        if row.get("task_status_counts"):
            status += "；" + ", ".join(
                f"{name}: {count}" for name, count in sorted(row["task_status_counts"].items())
            )
        deliveries.append(
            {
                **row,
                "label": row["workflow"],
                "display_status": status,
                "scope": "当前模型" if path in current_paths else "历史",
                "unfinished": path in unfinished_paths,
            }
        )
    for row in index["relax_mc_batches"]:
        members = [
            task
            for task in state.get("tasks") or []
            if task.get("relax_mc_bundle_id") == row["bundle_id"]
        ]
        cancelled = sum(task.get("recovery_cancelled") is True for task in members)
        current = bool(active_model and row["model_version"] == active_model)
        status = (
            f"停止等待 {cancelled} 个 MC；已有结果保留"
            if cancelled
            else {
                "completed": "已回收",
                "prepared": "输入已准备",
                "failed": "回传待核对",
                "cancelled": "已取消",
                "superseded": "已替代",
            }.get(row["status"], row["status"] or "待核对")
        )
        deliveries.append(
            {
                **row,
                "label": row["package_label"],
                "display_status": status,
                "scope": "当前模型" if current else "历史",
                "unfinished": current and any(waiting(task) for task in members),
            }
        )

    def link(value, label=None, *, missing="路径缺失，请核对", unregistered="未登记"):
        return (
            _index_link(value["path"], root, label, missing=missing)
            if value.get("path")
            else unregistered
        )

    def table(rows):
        result = [
            "| 批次 | 范围 | 状态 | 登记上传目录 | 登记命令（核对） | 回传目录 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            upload = link(
                row["submission_path"],
                missing="上传包目录缺失，先核对归档",
                unregistered="无上传目录",
            )
            command = f"`{row['submit_command']}`" if row.get("submit_command") else "未登记"
            result.append(
                f"| {row['label']} | {row['scope']} | {row['display_status']} | "
                f"{upload} | {command} | {link(row['return_path'])} |"
            )
        return result

    lines = ["# 提交与回收", ""]
    lines.extend(_navigation_markdown())
    lines.extend(
        [
            "本地未回收不代表尚未提交或远端未运行；先核对原提交记录，勿重复提交。"
            "以下路径与命令仅供核对，不构成提交授权。回传放入对应目录后说“已回传”，无需手动解压。",
            "",
            "## 当前待回传/核对",
            "",
        ]
    )
    current = [row for row in deliveries if row["unfinished"]]
    lines.extend(table(current) if current else ["当前没有登记的本模型未完交付。"])
    historical = [row for row in deliveries if not row["unfinished"]]
    if historical:
        lines.extend(["", "<details>", "<summary>历史交付与其他登记包</summary>", ""])
        lines.extend(table(historical))
        lines.extend(["", "</details>"])
    lines.extend(
        [
            "",
            "<details>",
            "<summary>内部计算档案与目录说明</summary>",
            "",
            "机器登记见 `submission_index.json`；状态以 workflow_state 中的任务记录为准。"
            "目录存在性只是生成时快照，使用前核对。历史记录不构成重投授权。",
            "",
            "`epochN/delivery/transfers/` 与 `epochN/delivery/relax_mc/` 保存配对上传/回传；"
            "`epochN/calculations/` 是本地计算，`epochN/evidence/` 保存已校验结果、恢复资料和归档。"
            "未登记的旧目录和 return-results 已移入 previously_used/，冻结 manifest 保留原始登记。",
            "旧版 `epochN/transfers/`、`epochN/relax_mc/` 路径只作为历史别名保留；当前提交入口必须使用 `delivery/` 下的登记目录。",
            "",
            "| 计算目录 / 解压目标 | 批次数 | 输入包 | 结果包 |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in index["operations"]:
        directory = row["directory"]
        relative = Path(os.path.relpath(directory["path"], root)).as_posix()
        inputs = row["input_archive"] if row["input_archive"]["path"] else row["input_directory"]
        results = (
            row["results_archive"] if row["results_archive"]["path"] else row["results_directory"]
        )
        lines.append(
            f"| {link(directory, relative)} | {row['batch_count']} | "
            f"{link(inputs)} | {link(results)} |"
        )
    legacy_returns = [
        (row["package_label"], path)
        for row in index["relax_mc_batches"]
        for path in row.get("legacy_return_paths") or []
    ]
    if legacy_returns:
        lines.extend(
            [
                "",
                "历史回传路径仅供核对，不是本轮人工回传入口；以主表登记路径为准。",
                "",
                "| 批次 | 历史回传路径 |",
                "| --- | --- |",
            ]
        )
        lines.extend(f"| {label} | {link(path)} |" for label, path in legacy_returns)
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def write_submission_index(root, state):
    root = Path(root).resolve()
    machine_index = _submission_index_payload(root, state)
    text = _submission_markdown(machine_index, state)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "README.md"
    path.write_text(text, encoding="utf-8")
    (root / "submission_index.json").write_text(
        json.dumps(machine_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    from phase_agent.analysis.feedback.workspace_guide import publish_epoch_guides

    write_epoch_manifests(
        root,
        (state.get("upload_layout") or {}).get("model_rounds") or {},
        _registered_task_directories(root, state),
    )
    publish_epoch_guides(root, state)
    return path
