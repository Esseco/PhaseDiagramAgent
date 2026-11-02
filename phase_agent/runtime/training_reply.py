"""Plain-language training decisions, including older saved handoffs."""


def training_delivery_lines(job):
    from phase_agent.runtime.handoff_presentation import handoff_lines

    transfer = job.get("handoff")
    if not transfer:
        return []
    status = job.get("status")
    delivery = handoff_lines([transfer], submission=status == "inputs_prepared")
    if transfer.get("training_job_count") and transfer.get("max_parallel_training_jobs"):
        delivery.insert(
            0,
            f"训练安排：{transfer['training_job_count']} 个独立单 GPU 训练，最多 {transfer['max_parallel_training_jobs']} 个同时运行；"
            "全部结束后另用 1 个单 GPU 作业评估汇总，另有 1 个入口作业。",
        )
    if status == "inputs_prepared":
        return [
            "结果：微调输入已生成；本地尚未登记超算提交状态。",
            *delivery,
            "下一步：若尚未提交，将整个输入文件夹上传超算，按上方脚本和命令提交；"
            "若已提交，等待训练完成后将对应输出文件夹回传到上方回收路径，再告知“已回收”（也支持“已回传”），系统直接分析并生成本轮微调结果报告，再请你批准是否进入下一 epoch。"
            "已生成的输入无需再次批准或生成，已提交的任务请勿重复提交。",
        ]
    if status in {"submitted", "running"}:
        return [
            "结果：微调任务已提交，等待训练结果。"
            if status == "submitted"
            else "结果：微调任务运行中，等待训练结果。",
            *delivery,
            "下一步：等待超算训练完成，将输出回传到上方回收路径，再告知“已回收”（也支持“已回传”），系统直接分析并生成本轮微调结果报告，再请你批准是否进入下一 epoch；"
            "请勿再次提交本批。",
        ]
    return delivery


def concise_training_reply(handoff, state):
    if handoff.get("stage") != "awaiting_activation_approval":
        return None
    candidate = (state.get("candidate_models") or {}).get(
        handoff.get("candidate_model_version")
    ) or {}
    review = candidate.get("agent_review") or {}
    if review.get("choice") == "activate":
        return (
            "结果：本轮微调分析已完成。\n"
            + (
                f"本轮微调结果报告：`{handoff['training_report_path']}`\n"
                if handoff.get("training_report_path")
                else ""
            )
            + "Agent建议：换用新模型。\n原因："
            + str(review["reason"])[:180]
            + f"\n下一步：是否进入 {handoff.get('next_epoch') or '下一 epoch'}？回复‘同意’批准切换，或回复‘拒绝’。"
            + "本次批准仅切换模型；切换后再审阅已有结构的能量和相图刷新方案，刷新输入另行审批。"
        )
    return "微调已完成，新模型测试误差更小。\n下一步：由Agent比较换模型、补DFT等方案。回复‘继续’获取判断；无需填写模型编号。"
