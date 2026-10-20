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
            lines.append(f"本轮提交文件夹：`{folder}`")
        if destination:
            lines.append(f"结果回传文件夹：`{destination}`")
        else:
            lines.append("结果回传位置未登记；需先核对已有交付记录。")
        if detailed_response() and row.get("upload_archive"):
            lines.append(
                f"压缩包：`{row['upload_archive']}`；作业数：{row.get('job_count', '未登记')}。"
            )
        if submission and folder and row.get("submit_script"):
            lines.append(
                "提交：上传整个本轮文件夹，在该目录执行 `sbatch submit_all.sh`；已提交的不要重复提交。"
            )
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
                f"下一步：等待任务完成，将远端{returned}整体回传到 `{destination}`，再说“已回传”。压缩包自动读取，无需手动解压。"
            )
    errors = (collection or {}).get("transfer_errors") or []
    if errors:
        lines.append("结果包导入失败：" + "；".join(row["reason"] for row in errors[:3]))
    return list(dict.fromkeys(lines))
