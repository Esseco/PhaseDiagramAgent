"""Render registered delivery facts; never prepare files or infer a new handoff."""

from pathlib import Path

from phase_agent.runtime.response_preferences import detailed_response


def handoff_lines(handoffs, *, collection=None, submission=True):
    lines = []
    for row in handoffs:
        folder = row.get("folder") or (
            str(Path(row["upload_archive"]).parent) if row.get("upload_archive") else None
        )
        destination = row.get("return_directory") or row.get("results_directory")
        paired = row.get("layout_version", 0) >= 4
        if folder:
            lines.append(f"提交路径：`{folder}`（本地上传文件夹）")
        if destination:
            lines.append(f"回收路径：`{destination}`（本地接收文件夹）")
        else:
            lines.append("回收路径未登记；需先核对已有交付记录。")
        if detailed_response() and row.get("upload_archive"):
            lines.append(
                f"压缩包：`{row['upload_archive']}`；作业数：{row.get('job_count', '未登记')}。"
            )
        if submission and folder and row.get("submit_script"):
            script = row["submit_script"]
            command = row.get("command") or f"sbatch {Path(script).name}"
            lines.append(f"提交脚本：`{script}`；提交命令：`{command}`。")
        elif submission:
            lines.append("提交脚本或文件夹未登记；先核对，不猜测命令。")
        if destination:
            returned = (
                "对应批次的 `_output` 文件夹"
                if paired
                else "`return-results` 文件夹"
                if row.get("layout_version", 0) >= 2
                else "`results/transfer/all.results.zip`"
            )
            lines.append(
                f"超算端结果目录：{returned}。任务完成后将其整体回传到上面的本地回收路径，再说“已回传”；压缩包自动读取，无需手动解压。"
            )
    errors = (collection or {}).get("transfer_errors") or []
    if errors:
        lines.append("结果包导入失败：" + "；".join(row["reason"] for row in errors[:3]))
    return list(dict.fromkeys(lines))


def planned_handoff_lines(proposal, state, state_path, *, config=None):
    """Show the expected paired paths before approval without creating files."""
    action = (proposal or {}).get("raw_action") or {}
    params = action.get("parameters") or (proposal or {}).get("action_parameters") or {}
    tool = action.get("tool") or (proposal or {}).get("recommended_action")
    mode = params.get("mode")
    if tool == "update_mlip":
        if params.get("action", "RETRAIN_MLIP") != "RETRAIN_MLIP":
            return []
        kind, directory, name = "微调", "delivery/transfers", "<微调批次>"
    elif tool == "select_dft_candidates" or mode == "dft_inputs":
        kind, directory, name = "DFT", "delivery/transfers", "<DFT批次>"
    elif (
        mode == "relax_mc_inputs"
        or tool == "prepare_relax_mc_batch"
        or (tool == "generate_branches" and params.get("module_execution_scope"))
    ):
        kind, directory, name = "Relax+MC", "delivery/relax_mc", None
    elif tool == "allocate_mc_bohb":
        kind, directory, name = "MC", "delivery/transfers", "<MC批次>"
    elif mode in {"relax_inputs", "mc_inputs", "model_refresh_inputs"}:
        kind = {"relax_inputs": "Relax", "mc_inputs": "MC", "model_refresh_inputs": "模型刷新"}[
            mode
        ]
        directory, name = "delivery/transfers", f"<{kind}批次>"
    else:
        return []

    root = (config or {}).get("upload_batches_directory")
    if root:
        root = Path(root).resolve()
    elif state_path:
        state_file = Path(state_path).resolve()
        project_root = next(
            (
                parent.parent
                for parent in state_file.parents
                if parent.name in {"workflow_state", "runtime"}
            ),
            None,
        )
        root = project_root / "submissions" if project_root else None
    if not root:
        return ["提交路径、提交脚本与回收路径尚未登记；先核对项目输入目录配置。"]
    preview = params.get("relax_mc_preview") or params.get("dft_input_preview") or {}
    version = (
        preview.get("model_version")
        or params.get("model_version")
        or state.get("active_model_version")
    )
    rounds = (state.get("upload_layout") or {}).get("model_rounds") or {}
    try:
        epoch_number = (
            int(rounds[version]) - 1
            if version in rounds
            else max((int(value) for value in rounds.values()), default=0)
        )
    except (TypeError, ValueError):
        return []
    if epoch_number < 0:
        return []

    parent = root / f"epoch{epoch_number}" / directory
    if kind == "Relax+MC":
        if version:
            sequence = 1 + sum(
                row.get("model_version") == version
                for row in (state.get("relax_mc_batches") or {}).values()
            )
            name = f"RelaxMC_Batch{sequence:02d}"
        else:
            name = "RelaxMC_Batch<NN>"
    upload = parent / f"{name}_input"
    returned = parent / f"{name}_output"
    lines = [
        f"预计提交路径（{kind}）：`{upload}`",
        f"预计提交脚本：`{upload / 'submit_all.sh'}`；命令：`sbatch submit_all.sh`。",
        f"预计回收路径：`{returned}`（超算端回传配对 `_output` 文件夹）。",
        "本次只准备本地输入，不提交超算；后续提交由你手动执行。",
    ]
    if "<" in name:
        lines.insert(3, "以上为目录命名规则；具体批次文件夹在输入准备完成后登记并给出。")
    return lines
