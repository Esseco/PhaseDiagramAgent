"""Classify whether the user's current message authorizes editing configuration."""

from __future__ import annotations


def classify_config_edit_intent(message: str, *, agent_client=None) -> str:
    """Return edit, discuss, or uncertain; never infer permission from prior turns."""
    if not str(message or "").strip():
        return "discuss"
    if agent_client is None:
        return "uncertain"
    try:
        response = agent_client({
            "mode": "classify_config_edit_intent",
            "instruction": (
                "只判断下面这条用户原话是否要求现在修改并保存项目长期配置文件。"
                "返回 JSON：intent 为 edit、discuss 或 uncertain；direct_request 为布尔值。"
                "用户不必说‘配置文件’：明确要求把参数设为某值、增减预算或修改上限，"
                "且没有限定为本轮/当前动作时，按 edit 处理。"
                "要求给建议、解释参数、只改本轮待审批动作、询问能否修改，都不是 edit。"
                "例如‘入选上限改为500’是 edit；‘这轮先用500’是 action_feedback；"
                "‘入选上限是什么意思’是 discuss。"
                "不要从历史对话推断授权，不要提出或执行科学计算。"
            ),
            "user_message": str(message),
        })
    except Exception:
        return "uncertain"
    if not isinstance(response, dict) or response.get("direct_request") is not True:
        return "discuss" if isinstance(response, dict) and response.get("intent") == "discuss" else "uncertain"
    return "edit" if response.get("intent") == "edit" else "uncertain"
