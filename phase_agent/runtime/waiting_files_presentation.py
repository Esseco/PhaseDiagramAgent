"""Read-only, actionable file handoffs for waiting calculations."""

from pathlib import Path

from phase_agent.tools.state.task_waiting import awaiting_task_result
from phase_agent.runtime.response_preferences import detailed_response


def waiting_file_lines(state):
    if not detailed_response():
        return _brief_waiting_file_lines(state)
    return _detailed_waiting_file_lines(state)


def _brief_waiting_file_lines(state):
    refresh = state.get("model_refresh") or {}
    refresh_ids = (
        set(refresh.get("task_ids") or []) if refresh.get("status") == "waiting_results" else None
    )
    tasks = [
        row
        for row in state.get("tasks", [])
        if awaiting_task_result(row) and (refresh_ids is None or row.get("task_id") in refresh_ids)
    ]
    if not tasks:
        return []
    ids = {row.get("task_id") for row in tasks}
    batches = [
        row for row in state.get("slurm_batches", []) if ids.intersection(row.get("task_ids") or [])
    ]
    folders = []
    for row in batches:
        handoff = row.get("handoff") or {}
        path = (
            handoff.get("folder") or row.get("operation_directory") or row.get("upload_directory")
        )
        if path:
            directory = Path(path)
            # The recorded model-round directory contains all refresh operations.
            folder = (
                directory
                if handoff.get("folder")
                else next(
                    (
                        p
                        for p in [directory, *directory.parents]
                        if p.name.startswith("epoch") and "_" in p.name
                    ),
                    directory,
                )
            )
            folders.append(folder)
    if not folders:
        for row in tasks:
            if row.get("input_path"):
                directory = Path(row["input_path"]).parent
                folders.append(
                    next(
                        (
                            p
                            for p in [directory, *directory.parents]
                            if p.name.startswith("epoch") and "_" in p.name
                        ),
                        directory,
                    )
                )
    folders = list(dict.fromkeys(folders))
    if not folders:
        return ["本轮提交文件夹未记录，需先核对文件位置。"]
    # Do not broaden unrelated rounds into an arbitrary common disk directory.
    links = [str(folder) for folder in folders]
    lines = ["本轮提交文件夹：" + "、".join(links)]
    returns = list(
        dict.fromkeys(
            (row.get("handoff") or {}).get("return_directory")
            for row in batches
            if (row.get("handoff") or {}).get("return_directory")
        )
    )
    if returns:
        lines.append("结果回传文件夹：" + "、".join(returns))
        lines.append(
            "下一步：上传提交文件夹，执行 `bash submit_all.sh`；完成后把远端 return-results 文件夹传回上述位置，再说“继续”。压缩包自动读取，无需手动解压；已提交的不要重复提交。"
        )
    else:
        lines.append(
            "下一步：上传本轮文件夹完成计算；把结果放回本轮对应 results 后说“继续”。已提交的无需重复提交。"
        )
    return lines


def _detailed_waiting_file_lines(state):
    refresh = state.get("model_refresh") or {}
    refresh_ids = (
        set(refresh.get("task_ids") or []) if refresh.get("status") == "waiting_results" else None
    )
    tasks = [
        row
        for row in state.get("tasks", [])
        if awaiting_task_result(row) and (refresh_ids is None or row.get("task_id") in refresh_ids)
    ]
    if not tasks:
        return []
    ids = {row.get("task_id") for row in tasks}
    batches = [
        row for row in state.get("slurm_batches", []) if ids.intersection(row.get("task_ids") or [])
    ]
    if not batches:
        inputs = list(dict.fromkeys(row["input_path"] for row in tasks if row.get("input_path")))
        results = list(
            dict.fromkeys(
                str(Path(row["result_path"]).parent) for row in tasks if row.get("result_path")
            )
        )
        lines = [f"输入文件：`{path}`" for path in inputs]
        lines += [f"结果回传目录：`{path}`" for path in results]
        if not inputs:
            lines.append("输入位置未记录；需先核对任务文件，不能直接提交。")
        if not results:
            lines.append("结果回传位置未记录；需先核对任务记录。")
        lines.append(
            "下一步：已提交的任务等待完成，将结果回传到上述对应位置后说“继续”；未提交的任务先核对提交脚本。"
        )
        return lines
    lines = [f"本次待回传 {len(tasks)} 个任务，涉及 {len(batches)} 个提交批次。"]
    if refresh_ids is not None:
        predict = sum(
            len(ids.intersection(row.get("task_ids") or []))
            for row in batches
            if row.get("refresh_operation") == "predict"
        )
        relax = sum(
            len(ids.intersection(row.get("task_ids") or []))
            for row in batches
            if row.get("refresh_operation") == "relax"
        )
        if predict or relax:
            lines.append(
                f"计算内容：MLIP 单点 {predict} 个、MLIP 弛豫 {relax} 个，使用新模型更新能量和结构。"
            )
    shown_handoffs, destinations = set(), []
    for batch in batches:
        handoff = batch.get("handoff") or {}
        if handoff.get("upload_archive") and handoff.get("submit_script"):
            key = (handoff["upload_archive"], handoff["submit_script"])
            if key not in shown_handoffs:
                shown_handoffs.add(key)
                lines.append(
                    f"上传文件：`{key[0]}` 和 `{key[1]}`；上传到超算同一新目录，在该目录执行 `bash submit_all.sh`。"
                )
                lines.append(
                    "完成后下载该目录的 `return-results` 文件夹，无需手动解压。"
                    if handoff.get("layout_version") == 2
                    else "完成后下载该目录的 `results/transfer/all.results.zip`。"
                )
        else:
            directory = batch.get("upload_directory")
            script = batch.get("script_path")
            archive = Path(str(directory) + ".upload.zip") if directory else None
            if archive is not None and archive.is_file():
                lines.append(f"上传包：`{archive}`")
            elif directory:
                lines.append(f"上传输入目录：`{directory}`")
            else:
                lines.append("输入位置未记录；需先核对任务文件。")
            if script:
                lines.append(
                    f"提交脚本：`{script}`；在超算解压后的对应目录执行 `sbatch {Path(script).name}`。"
                )
            else:
                lines.append("提交脚本未记录；先核对，不猜测命令。")
        destination = (
            handoff.get("return_directory")
            or handoff.get("results_directory")
            or batch.get("results_directory")
        )
        if destination and destination not in destinations:
            destinations.append(destination)
    lines += [f"结果回传目录：`{path}`" for path in destinations]
    if not destinations:
        lines.append("结果回传位置未记录；需先核对批次记录。")
    lines.append(
        "下一步：若尚未提交，按上述文件和脚本操作；已提交的不要重复提交。计算完成后回传对应结果，保留任务子目录，再说“继续”。"
    )
    return lines
