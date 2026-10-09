"""Semantic intent normalization; never approves or runs scientific actions."""


def normalize_chat_intent(user_message, state, *, agent_client, messages=(), enable_context=False, single_pass=False):
    from decision_layer.agent.resolve_chat_intent import resolve_chat_intent
    from run.conversation_context import conversation_context, answer_conversation
    intent = resolve_chat_intent(user_message, state,
        agent_client=agent_client,
        conversation=conversation_context(state, messages) if enable_context else {})
    kind = intent["intent"]
    if kind == "view_plan":
        from run.plan_queries import pending_plan_reply
        return {"reply": pending_plan_reply(state)}
    if kind == "unavailable":
        return {"reply": "本轮未推进：意图识别暂时失败，请重试；也可明确说‘查看状态’或‘继续’。未生成或执行任务。"}
    if kind == "clarify":
        return {"reply": intent.get("clarification") or "你想查看结果、继续下一步，还是重新准备某一轮的输入？请说明阶段和轮次。"}
    if kind == "read_only":
        return {"reply": intent["answer"]}
    if kind == 'read_config':
        return {'message': '读取配置 JSON 并继续' if intent['continue_if_ready'] else '读取配置 JSON'}
    if kind == 'candidate_review':
        command = '激活候选' if intent['decision'] == 'activate' else '拒绝候选'
        return {'candidate_command': f"{command} {intent['candidate_model_version']} 原因：{intent['reason']}"}
    if kind == "workflow_feedback":
        return {"message": user_message}
    if kind == "other":
        if intent.get("answer"):
            return {"reply": intent["answer"]}
        if single_pass and callable(agent_client):
            return {"reply": intent.get("clarification") or "当前请求的对象还不明确，请补充你要处理的结果或方案。"}
        if not callable(agent_client):
            return {"message": user_message}
        if not enable_context:
            return {"reply": "请明确是询问信息还是修改方案；本次未推进任务。"}
        return answer_conversation(user_message, state, agent_client=agent_client, messages=messages)
    if kind == "export_phase_csv":
        user_message = "导出当前相图" + (" DFT" if intent.get("stage") == "dft" else "")
    elif kind == "status":
        user_message = "查看状态"
    elif kind == "continue":
        user_message = "继续"
    elif kind == "redo_plan":
        # Use read-only generic planning, not the MC deletion confirmation path.
        from execution_layer.local.identify_rerun_plan import identify_rerun_plan, format_rerun_plan
        stage = {"mc": "MC", "relax": "Relax", "dft": "DFT", "branch": "branch"}[intent["stage"]]
        return {"reply": format_rerun_plan(identify_rerun_plan(f"重新准备当前轮{stage}", state))}
    return {"message": user_message}
