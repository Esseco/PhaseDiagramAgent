"""finetune operation adapter; invoked as a named dialogue graph node."""

from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.graphs.dialogue.commands import classify_user_decision, _drop_finished_pending


def handle(self, user_message, state, messages, conversation_id):
    state, stale_pending_removed = _drop_finished_pending(state)
    if stale_pending_removed:
        write_json(self.state_path, state)
    from phase_agent.tools.local.regenerate_finetune_inputs import (
        CONFIRM,
        is_finetune_regeneration_request,
        plan_finetune_regeneration,
        regenerate_finetune_inputs,
    )

    pending_training = state.get("pending_finetune_regeneration")
    training_confirm = str(user_message).strip() == CONFIRM
    training_reject = (
        pending_training
        and not state.get("pending_execution_policies")
        and classify_user_decision(user_message) == "reject"
    )
    if training_reject:
        state.pop("pending_finetune_regeneration", None)
        write_json(self.state_path, state)
        return "已取消微调输入重生成方案；文件与训练记录未改变。"
    if training_confirm or is_finetune_regeneration_request(user_message, state):
        from phase_agent.configuration.runtime.build_effective_run_config import (
            build_effective_run_config,
        )

        config = build_effective_run_config(
            self.workflow_kwargs["config_session"], self.workflow_kwargs["run_config"]
        )
        try:
            if training_confirm:
                if not pending_training:
                    return "尚无微调输入重生成方案，请先说“重新生成原轮微调输入”。未执行。"
                result = regenerate_finetune_inputs(state, config, pending_training)
                write_json(self.state_path, result["state"])
                return result["reason"] + "\n原轮编号不变；旧输入已备份，未提交或训练。"
            plan = plan_finetune_regeneration(state, config)
            state["pending_finetune_regeneration"] = plan
            write_json(self.state_path, state)
            operation = "备份旧输入后重新生成" if plan["exists"] else "目录不存在，直接重新生成"
            return (
                f"微调原轮输入：`{plan['directory']}`。\n方案：{operation}，使用当前有效训练数据和参数；保留原轮编号，不追加轮次。\n"
                f"请确认旧输入是否已在超算提交；已提交则不要重生成。执行请单独回复“{CONFIRM}”；“拒绝”取消。当前未移动或生成文件。"
            )
        except (ValueError, OSError, RuntimeError) as error:
            return f"微调输入重生成未执行或已回滚：{error}。未提交或训练。"
    return None
