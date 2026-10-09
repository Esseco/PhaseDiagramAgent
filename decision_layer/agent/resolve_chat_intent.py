"""Interpret paraphrases without granting execution or deletion authority."""


def resolve_chat_intent(message, state, *, agent_client=None, conversation=None):
    if not callable(agent_client):
        return {"intent": "other"}
    pending = state.get("pending_execution_policies") or {}
    try:
        response = agent_client({
            "mode": "resolve_chat_intent",
            "instruction": (
                "识别用户当前原话的意图，不执行。返回 JSON: intent, direct_request, "
                "confidence, stage, scope, clarification。intent 只允许 export_phase_csv、"
                "status、continue、redo_plan、view_plan、read_only、other、clarify。stage 只允许 mc、relax、dft、"
                "branch、unknown；scope 只允许 current、unknown。"
                "识别同义词、口语、否定与问句。询问如何/能否、讨论原理不等于执行请求。"
                "查看/展示已经提出的方案或计划=view_plan，只读，不表示修改、重新决策或批准。"
                "文件位置、参数解释、为什么和能力询问优先read_only；answer直接回答，默认1至3句。"
                "结合context.conversation中的登记事实与最近对话；缺失信息明确说明，不编造路径或已执行动作。"
                "给相图数据表/最新凸包CSV=export_phase_csv；跑到哪/进展=status；"
                "接着做/往下走=continue；之前不要重新准备输入=redo_plan（只规划）。"
                "仅有明确当前轮次语义才能 scope=current；指定历史轮次用 other 保留原文。"
                "代词仅能从唯一待审批动作识别对象，否则 clarify。混合多个操作用 clarify。"
                "同意/拒绝/批准迁移/确认删除/确认重生成等审批一律 other，"
                "绝不能从同义词或上下文补出审批授权。confidence 是0到1的数。"
            ),
            "user_message": str(message),
            "context": {"pending_tools": [
                (row.get("agent_proposal") or {}).get("recommended_action") for row in pending.values()],
                "has_mc_regeneration_plan": bool(state.get("pending_mc_regeneration")),
                "conversation": conversation or {}},
        })
    except Exception as error:
        return {"intent": "unavailable", "reason": "intent_client_failed",
                "error_type": type(error).__name__}
    allowed = {"export_phase_csv", "status", "continue", "redo_plan", "view_plan", "read_only", "other", "clarify"}
    if not isinstance(response, dict) or response.get("intent") not in allowed:
        return {"intent": "unavailable", "reason": "invalid_intent_response"}
    if response["intent"] in {"other", "clarify"}:
        return {"intent": response["intent"]}
    if response["intent"] == "view_plan":
        return {"intent": "view_plan"}
    if response["intent"] == "read_only":
        answer = response.get("answer")
        if isinstance(answer, str) and answer.strip():
            return {"intent": "read_only", "answer": answer.strip()[:3500]}
        return {"intent": "other"}
    confidence = response.get("confidence")
    if response.get("direct_request") is not True:
        return {"intent": "other"}
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or confidence < 0.9:
        return {"intent": "clarify"}
    if response["intent"] == "redo_plan" and (
        response.get("scope") != "current" or response.get("stage") not in {"mc", "relax", "dft", "branch"}):
        return {"intent": "clarify"}
    return {"intent": response["intent"], "stage": response.get("stage")}
