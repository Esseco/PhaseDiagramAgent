"""Plain-language training decisions, including older saved handoffs."""


def concise_training_reply(handoff, state):
    if handoff.get("stage") != "awaiting_activation_approval":
        return None
    import os

    approval_url = f"http://127.0.0.1:{os.environ.get('PHASE_CONTROL_PORT', '8765')}/phase/approval"
    candidate = (state.get("candidate_models") or {}).get(
        handoff.get("candidate_model_version")
    ) or {}
    review = candidate.get("agent_review") or {}
    if review.get("choice") == "activate":
        return (
            "方向判断已完成。Agent建议：换用新模型。\n原因："
            + str(review["reason"])[:180]
            + "\n下一步：在本机审批页核对并批准切换模型："
            + approval_url
            + "\n模型激活须在审批页确认；聊天回复‘同意’会转到该页面。切换后再审阅已有结构的能量和相图刷新方案，刷新输入另行审批。"
        )
    return "微调已完成，新模型测试误差更小。\n下一步：由Agent比较换模型、补DFT等方案。回复‘继续’获取判断；无需填写模型编号。"
