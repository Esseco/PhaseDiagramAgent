"""Plain-language training decisions, including older saved handoffs."""


def concise_training_reply(handoff, state):
    if handoff.get("stage") != "awaiting_activation_approval":
        return None
    candidate = (state.get("candidate_models") or {}).get(
        handoff.get("candidate_model_version")
    ) or {}
    review = candidate.get("agent_review") or {}
    if review.get("choice") == "activate":
        return (
            "方向判断已完成。Agent建议：换用新模型。\n原因："
            + str(review["reason"])[:180]
            + "\n下一步：更新已有结构的能量和相图，再判断是否收敛或需要补DFT。\n回复‘同意’仅批准切换模型，‘拒绝’取消；刷新输入另行审批。"
        )
    return "微调已完成，新模型测试误差更小。\n下一步：由Agent比较换模型、补DFT等方案。回复‘继续’获取判断；无需填写模型编号。"
