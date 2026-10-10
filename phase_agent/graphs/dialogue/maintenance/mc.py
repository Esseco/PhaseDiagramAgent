"""mc operation adapter; invoked as a named dialogue graph node."""

from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.graphs.dialogue.commands import classify_user_decision


def handle(self, user_message, state, messages, conversation_id):
    from phase_agent.tools.local.regenerate_mc_inputs import (
        is_mc_regeneration_request,
        is_mc_regeneration_confirmation,
        plan_mc_regeneration,
        regenerate_mc_inputs,
    )

    pending_mc = state.get("pending_mc_regeneration")
    reject_mc = (
        pending_mc
        and not state.get("pending_execution_policies")
        and classify_user_decision(user_message) == "reject"
    )
    mc_request = is_mc_regeneration_request(user_message)
    mc_confirmed = is_mc_regeneration_confirmation(user_message)
    if mc_request or mc_confirmed or reject_mc:
        if reject_mc:
            state.pop("pending_mc_regeneration", None)
            write_json(self.state_path, state)
            return "已取消 MC 重生成计划；任务状态和文件未改变。"
        if mc_confirmed and not pending_mc:
            return "尚无待确认的 MC 重生成方案。请先说“重新生成当前轮次 MC 任务”，查看范围和影响。"
        from phase_agent.configuration.runtime.build_effective_run_config import (
            build_effective_run_config,
        )

        config = build_effective_run_config(
            self.workflow_kwargs["config_session"], self.workflow_kwargs["run_config"]
        )
        root = config["upload_batches_directory"]
        try:
            plan = plan_mc_regeneration(state, root)
            if not mc_confirmed:
                state["pending_mc_regeneration"] = plan
                write_json(self.state_path, state)
                existing = (
                    "该目录存在；确认后会删除其中已生成的 MC 输入。"
                    if plan["exists"]
                    else "该目录不存在；无需删除文件，但会废弃旧 MC 状态。"
                )
                stage_limit = ((config.get("budgets") or {}).get("stage_limits") or {}).get(
                    "deep_search"
                ) or {}
                cost_limit = stage_limit.get("max_cost")
                over_limit = (
                    f"；高于当前 MC 阶段成本上限 {float(cost_limit):.2f}"
                    if cost_limit is not None
                    and float(plan["estimated_relative_cost"]) > float(cost_limit)
                    else ""
                )
                strata = "、".join(
                    f"{label} {count} 个"
                    for label, count in plan["allocation_by_tier_phase"].items()
                )
                return (
                    "当前只提出方案，尚未删除文件或生成任务。\n"
                    f"当前轮 MC 目录：`{plan['directory']}`。{existing}\n"
                    f"将废弃 {len(plan['old_task_ids'])} 个旧 MC 任务及 "
                    f"{plan['old_batch_count']} 条旧批次记录，保留其他阶段和已批准的分配方案。\n"
                    f"重新生成 {plan['allocation_count']} 个独立 MC 算例、"
                    f"共 {plan['total_mc_steps']} 步；估计相对成本 "
                    f"{float(plan['estimated_relative_cost']):.2f}{over_limit}。\n"
                    f"每个 GPU 批次最多 {plan['mc_batch_size']} 个算例，"
                    f"按当前兼容条件预计 {plan['estimated_batch_count']} 个批次；"
                    "每个算例仍有独立结果。\n"
                    f"分层/分相数量：{strata}。\n"
                    f"新 MC-sampling 从 {plan['mc_sampling_start']:04d} 编号，"
                    f"远端批次从 remote-{plan['remote_batch_start']:06d} 编号；"
                    "仅生成本地输入，不提交超算作业。\n"
                    "请核对方案和旧任务是否已在超算提交；确定废弃并执行时，"
                    "单独回复“确认重新生成当前轮MC任务”。回复“拒绝”可取消。"
                )
            if plan != pending_mc:
                raise ValueError(
                    "MC 状态或目录在确认后发生变化；未执行，请重新提出请求并核对新方案"
                )
            state.pop("pending_mc_regeneration", None)
            upload = regenerate_mc_inputs(
                state,
                approved_plan=plan,
                upload_root=root,
                config=config,
                manager=self.workflow_kwargs["manager"],
                phase_references=self.workflow_kwargs["phase_references"],
                config_version=state["confirmed_config_version"],
                allow_delete=True,
            )
            write_json(self.state_path, upload["state"])
            return (
                f"已按原批准的完整方案重新生成 {upload['task_count']} 个 MC 输入，"
                f"分为 {upload['batch_count']} 个批次；未提交超算作业。"
                f"目录：`{plan['directory']}`。"
            )
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            return f"MC 输入重生成未完成：{error}。未提交超算作业。"
    return None
