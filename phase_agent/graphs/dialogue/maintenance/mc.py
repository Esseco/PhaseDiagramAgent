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
    from phase_agent.graphs.dialogue.support import _load_binding, _save_binding
    from phase_agent.tools.remote.integrity import payload_checksum
    from phase_agent.runtime.review_requests import pending_reviews

    if (
        is_mc_regeneration_request(user_message)
        or is_mc_regeneration_confirmation(user_message)
        or (pending_mc and classify_user_decision(user_message) in {"approve", "reject"})
    ):
        self.conversation_id = conversation_id
    binding = _load_binding(self) or {}
    simple_approval = (
        bool(pending_mc)
        and classify_user_decision(user_message) == "approve"
        and not pending_reviews(state)
    )
    bound_approval = (
        simple_approval
        and binding.get("kind") == "mc_regeneration"
        and binding.get("proposal_hash") == payload_checksum(pending_mc)
    )
    reject_mc = (
        pending_mc
        and not state.get("pending_execution_policies")
        and classify_user_decision(user_message) == "reject"
    )
    mc_request = is_mc_regeneration_request(user_message) or (
        simple_approval and not bound_approval
    )
    mc_confirmed = is_mc_regeneration_confirmation(user_message) or bound_approval
    if mc_request or mc_confirmed or reject_mc:
        if reject_mc:
            state.pop("pending_mc_regeneration", None)
            _save_binding(self, None)
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
            from phase_agent.tools.local.replan_mc_allocation import replan_mc_allocation

            preview = replan_mc_allocation(state, self.workflow_kwargs["manager"], config)
            plan = plan_mc_regeneration(state, root, replacement_preview=preview)
            if not mc_confirmed:
                state["pending_mc_regeneration"] = plan
                write_json(self.state_path, state)
                checksum = payload_checksum(plan)
                _save_binding(
                    self,
                    {
                        "kind": "mc_regeneration",
                        "plan_id": "mc-regeneration:" + checksum,
                        "proposal_hash": checksum,
                    },
                )
                existing = (
                    "该目录存在；将先备份再替换当前 MC 输入，全部 results 保留。"
                    if plan["exists"]
                    else "该目录不存在；无需移动旧输入，但会更新旧 MC 状态。"
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
                    f"将替换 {len(plan['old_task_ids'])} 个旧 MC 任务及 "
                    f"{plan['old_batch_count']} 条旧批次记录，保留其他阶段。\n"
                    "已根据当前 Relax 凸包和确认的 MC 筛选规则重新规划，不复用旧截断清单。\n"
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
                    "核对后回复“同意”即可按此方案生成；“拒绝”取消。"
                )
            if plan != pending_mc:
                return "MC 依据或输入已变化，本次未执行；已更新方案供核对。\n" + handle(
                    self, "重新生成MC任务", state, messages, conversation_id
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
                replacement_preview=preview,
            )
            write_json(self.state_path, upload["state"])
            _save_binding(self, None)
            return (
                f"已按当前 Relax 凸包重新规划并生成 {upload['task_count']} 个 MC 输入，"
                f"分为 {upload['batch_count']} 个批次；未提交超算作业。"
                f"目录：`{plan['directory']}`。旧输入已备份，results 保留。"
            )
        except (KeyError, OSError, ValueError, RuntimeError) as error:
            return f"MC 输入重生成未完成：{error}。未提交超算作业。"
    return None
