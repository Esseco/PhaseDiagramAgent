"""Readable per-interaction Markdown and a short chat receipt; no scientific effects."""

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from uuid import uuid4


def _markdown_table(lines):
    output, table = [], False
    for line in lines:
        if " | " in line:
            cells = line.split(" | ")
            if not table:
                output.extend(
                    [
                        "",
                        "| " + " | ".join(cells) + " |",
                        "| " + " | ".join("---" for _ in cells) + " |",
                    ]
                )
            else:
                output.append("| " + " | ".join(cells) + " |")
            table = True
        else:
            output.extend(["", line])
            table = False
    return "\n".join(output)


ORIGINS = {
    "remote_calculation": "超算计算",
    "remote_job": "Slurm作业",
    "remote_controller": "远端控制器",
    "remote_collection": "远端回收",
    "local_import": "本地导入",
    "return_transfer": "传输回传",
}


def _cell(value):
    return str(value if value is not None else "未记录").replace("|", "\\|").replace("\n", "<br>")


def _trace(value):
    fence = "`" * max(3, max((len(m[0]) for m in re.finditer(r"`+", value)), default=0) + 1)
    return f"\n{fence}text\n{value}\n{fence}\n"


def short_reply(reply):
    """Keep concrete findings and actions; this never selects a scientific action."""
    filler = {"本次：已检查当前项目状态。", "预算：本次未提供新的计算预算。"}
    lines = [
        line.strip()
        for line in str(reply).splitlines()
        if line.strip() not in filler and line.strip()
    ]
    selected = [lines[0]] if lines and "·" in lines[0] else []

    def first(prefixes):
        return next(
            (line for prefix in prefixes for line in lines if line.startswith(prefix)), None
        )

    current = first(("当前状态：", "当前：", "当前项目状态："))
    if current and current not in selected:
        selected.append(current)
    finding = first(("本轮失败：", "本轮未执行：", "结果：", "待确认：", "本次："))
    if finding:
        if finding not in selected:
            selected.append(finding)
    else:
        # An ordinary answer may have no fixed prefix; keep the answer itself.
        plain = next(
            (
                line
                for line in lines
                if line not in selected
                and not line.startswith(
                    ("原因：", "阻碍：", "需注意：", "下一步：", "建议：", "预算：")
                )
            ),
            None,
        )
        if plain:
            selected.append(plain)
    for prefixes in (
        ("原因：", "阻碍：", "回传核对失败：", "需注意："),
        ("下一步：", "建议：", "需要你："),
        ("预算：",),
    ):
        line = first(prefixes)
        if line and line not in selected:
            selected.append(line)
    handoff_prefixes = (
        "Relax+MC 输入地址：",
        "Relax+MC 提交脚本：",
        "Relax+MC 回收地址：",
        "DFT 输入地址：",
        "DFT 提交脚本：",
        "DFT 回传地址：",
        "DFT 回收地址：",
        "提交脚本：",
    )
    handoff_lines = [
        line
        for line in lines
        if any(line.startswith(prefix) for prefix in handoff_prefixes)
        and line not in selected
    ]
    if handoff_lines:
        next_index = next(
            (
                index
                for index, line in enumerate(selected)
                if line.startswith(("下一步：", "建议：", "需要你："))
            ),
            len(selected),
        )
        selected[next_index:next_index] = handoff_lines
    if not first(("下一步：", "建议：", "需要你：")):
        # Input handoffs must keep the real folder, command and return location.
        for prefix in ("上传整个文件夹：", "提交：", "回传："):
            line = first((prefix,))
            if line and line not in selected:
                selected.append(line)
    return "  \n".join(line if len(line) <= 220 else line[:217] + "…" for line in selected)


def save_interaction(root, message, full_reply, display_reply, state, *, record=None):
    from phase_agent.text_safety import redact_text
    from phase_agent.runtime.result_recovery_status import result_recovery_facts

    root = Path(root).resolve()
    from phase_agent.runtime.interaction_report_paths import report_directory

    directory = report_directory(root, state)
    directory.mkdir(parents=True, exist_ok=True)
    identity = (record or {}).get("id") or uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{8,64}", identity):
        raise ValueError("invalid_interaction_identity")
    now = datetime.now(timezone.utc)
    summary = short_reply(display_reply)
    text = [
        f"# 本轮交互记录\n\n时间：{now.isoformat()}\n",
        f"记录标识：`{identity}`；项目：`{root}`。",
        "## 用户消息\n\n" + str(message),
        "## 本轮结论\n\n" + summary,
    ]
    if str(full_reply).strip() != summary.strip():
        text.append("## 详细说明\n\n" + str(full_reply))
    # Only embed artifacts actually referenced by this turn, not old proposals.
    seen, dft_keys = set(), set()
    for value in re.findall(r"`([^`\n]+\.(?:txt|json|md))`", str(full_reply)):
        artifact = Path(value)
        if not artifact.is_absolute():
            artifact = root / artifact
        artifact = artifact.resolve()
        if artifact in seen or not artifact.is_relative_to(root) or not artifact.is_file():
            continue
        if not any(part in {"branch_plans", "dft_plans"} for part in artifact.parts):
            continue
        seen.add(artifact)
        source = artifact.with_suffix(".json")
        if source.is_file() and source.stat().st_size <= 8 * 1024 * 1024:
            from phase_agent.tools.workflows.branch_review_view import raw_record
            from phase_agent.tools.workflows.compact_review import branch_lines
            from phase_agent.tools.workflows.dft_review_view import markdown_view

            try:
                saved = json.loads(source.read_text(encoding="utf-8"))
                if not isinstance(saved, dict):
                    raise ValueError("展示记录必须是 JSON 对象")
                payload = raw_record(saved)
            except (OSError, ValueError, TypeError) as error:
                text.append(
                    f"展示文件读取受阻：`{source}`；{type(error).__name__}。保留本轮答复，以已保存方案为准。"
                )
                continue
            kind = "DFT" if "dft_plans" in source.parts else "Branch"
            from phase_agent.tools.workflows.branch_review_view import markdown_rationale

            if kind == "Branch" and not payload.get("reason") and saved.get("本次选择理由"):
                payload = {
                    **payload,
                    "reason": saved["本次选择理由"],
                    "branch_region_review_notes": saved.get("本轮背景与复用说明"),
                }
            rows = (
                markdown_view(payload)
                if kind == "DFT"
                else markdown_rationale(payload) + _markdown_table(branch_lines(payload))
            )
            if kind == "DFT":
                dft_keys.add(payload.get("task_key"))
            text.append(f"## {kind} 选点分布\n\n" + rows + f"\n\n原始记录：`{source}`")
    # Status/plan queries can show a saved proposal without any artifact link.
    for pending in (state.get("pending_execution_policies") or {}).values():
        proposal = pending.get("agent_proposal") or {}
        action = proposal.get("raw_action") or {}
        if (action.get("tool") or proposal.get("recommended_action")) != "select_dft_candidates":
            continue
        binding = pending.get("project_identity") or {}
        if any(
            binding.get(k) not in {None, state.get(target)}
            for k, target in (
                ("model_version", "active_model_version"),
                ("config_version", "confirmed_config_version"),
            )
        ):
            text.append(
                "## 当前待审批 DFT 方案\n\n方案版本与当前项目不同，请核对原方案；不使用当前候选补造旧结构信息。"
            )
            continue
        if action.get("task_key") in dft_keys:
            continue
        from phase_agent.tools.workflows.dft_selection_artifacts import selection_record
        from phase_agent.tools.workflows.dft_review_view import markdown_view

        text.append(
            "## 当前待审批 DFT 选点分布\n\n" + markdown_view(selection_record(proposal, state))
        )
        dft_keys.add(action.get("task_key"))
    recoveries = result_recovery_facts(state)
    if recoveries:
        text.append(
            "## 回收核对与错误\n\n计算失败、远端回收和本地导入分开记录；缺少文件本身不证明计算失败。"
        )
        for batch in recoveries:
            text.append(
                f"### 批次 `{batch['bundle_id']}`\n\n{batch['label']}；待核对 {batch['pending_count']} 项。"
            )
            local = batch.get("return_diagnostic") or {}
            remote = batch.get("remote_diagnostics") or {}
            issues = [*remote.get("issue_groups", [])]
            if local:
                issues.append(local)
            driver = batch.get("driver_diagnostic") or {}
            if driver.get("error") and driver != local:
                issues.append(driver)
            if issues:
                text.append(
                    "| 来源 | 任务或作业 | 错误信息 |\n| --- | --- | --- |\n"
                    + "\n".join(
                        f"| {_cell(ORIGINS.get(row.get('origin'), row.get('origin')))} | {_cell(row.get('task_ids') or row.get('job_id'))} | {_cell(row.get('error') or row.get('code'))} |"
                        for row in issues
                    )
                )
            controller = remote.get("controller") or {}
            if controller:
                text.append("原控制器诊断：" + str(controller.get("error") or ""))
            elif batch.get("historical_controller_diagnostic"):
                text.append(
                    "已保存的历史控制器诊断（不等同于计算失败）："
                    + str(batch["historical_controller_diagnostic"])
                )
            if batch["pending_task_ids"]:
                text.append(
                    "尚缺结果核验凭据的任务："
                    + "、".join(f"`{task_id}`" for task_id in batch["pending_task_ids"])
                )
            for row in [*issues, controller]:
                trace = row.get("traceback") or row.get("log_tail")
                if trace:
                    text.append(_trace(str(trace)))
            for job in remote.get("jobs") or []:
                text.append(
                    f"作业 `{job.get('job_id')}`：{job.get('state')}；退出码 `{job.get('exit_code')}`。"
                )
                for log in job.get("logs") or []:
                    text.append(
                        f"日志 `{log.get('path')}`" + _trace(str(log.get("log_tail") or ""))
                    )
            if remote.get("evidence_path"):
                text.append(f"完整错误证据：`{remote['evidence_path']}`")
    model = state.get("active_model") or {}
    references = model.get("model_paths") or list(
        dict.fromkeys(
            row.get("remote_model_path") or row.get("model_path")
            for row in [model, *(model.get("members") or [])]
            if row.get("remote_model_path") or row.get("model_path")
        )
    )
    if references:
        text.append(
            "## 当前模型的远程引用\n\n下表来自已登记模型；本地文件存在性不能证明超算路径有效。历史失败里的路径也不能替代当前输入及远端预检证据。\n\n| 成员 | 远程路径 |\n| --- | --- |\n"
            + "\n".join(f"| {i} | `{_cell(value)}` |" for i, value in enumerate(references, 1))
        )
    if record:
        failure = record.get("failure_summary") or {}
        if failure:
            text.append(
                "## 故障详情\n\n" + _trace(json.dumps(failure, ensure_ascii=False, indent=2))
            )
            text.append(
                "后续核查：保留原方案与任务记录，先核对本轮执行回执；此报告不授权重放计算或修改科学数据。"
            )
        failed = [row for row in record.get("events", []) if row.get("title") == "本轮异常"]
        for row in failed:
            text.append(
                "## 本轮异常\n\n"
                + "\n\n".join(f"**{k}**：{v}" for k, v in (row.get("data") or {}).items())
            )
    body = redact_text("\n\n".join(text)) + "\n"
    from phase_agent.runtime.interaction_report_paths import report_stem, reserve_report

    path, reserved = reserve_report(directory, report_stem(state, record, now), record)
    temporary = path.with_suffix("." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(body, encoding="utf-8")
        temporary.replace(path)
    except BaseException:
        if reserved:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    if record is not None:
        record["interaction_report"] = str(path)
    return summary + f"  \n交互记录地址：{path}"
