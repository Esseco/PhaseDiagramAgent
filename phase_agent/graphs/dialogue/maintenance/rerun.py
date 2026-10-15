"""Confirm existing DFT regeneration plans; ordinary requests use Agent decisions."""

from phase_agent.tools.step_runner.file_protocol import write_json

CONFIRM = "确认重新生成当前轮DFT输入"


def handle(self, user_message, state, messages, conversation_id):
    from phase_agent.tools.local.identify_rerun_plan import (
        identify_rerun_plan,
        format_rerun_plan,
    )

    pending = state.get("pending_dft_regeneration")
    if (
        pending
        and user_message.strip() in {"拒绝", "reject"}
        and not state.get("pending_execution_policies")
    ):
        state.pop("pending_dft_regeneration", None)
        write_json(self.state_path, state)
        return "已取消 DFT 输入重生成方案；文件未修改。"
    confirmed = user_message.strip() == CONFIRM
    if confirmed and not pending:
        return "尚无待确认的 DFT 输入重生成方案；请先提出请求并核对范围。"
    # Natural-language requests belong to the current-model decision pipeline.
    # Only an existing, explicit DFT regeneration confirmation is handled here.
    if not confirmed:
        return None
    request = pending["request"]
    plan = identify_rerun_plan(request, state)
    if (
        plan.get("status") != "identified"
        or plan.get("intent") != "generate"
        or plan.get("action", {}).get("tool") != "select_dft_candidates"
    ):
        return format_rerun_plan(plan)
    if plan != pending["plan"]:
        return "DFT 任务或输入状态已变化；请重新提出请求并核对方案。未执行。"
    from phase_agent.tools.local.regenerate_dft_files import regenerate_dft_files

    try:
        result = regenerate_dft_files(
            state,
            request,
            manager=self.workflow_kwargs.get("manager"),
            upload_root=self.workflow_kwargs["run_config"]["upload_batches_directory"],
        )
    except (ValueError, OSError, RuntimeError) as error:
        return f"DFT 文件重生成未执行或已回滚：{error}。未提交作业。"
    state.pop("pending_dft_regeneration", None)
    write_json(self.state_path, state)
    return (
        f"已按原批准参数重新生成 {result['task_count']} 个 DFT 任务输入；未提交作业。"
        f"旧输入备份：{result['backup_directory']}。"
    )
