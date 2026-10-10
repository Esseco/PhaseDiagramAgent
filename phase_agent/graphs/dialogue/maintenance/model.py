"""model operation adapter; invoked as a named dialogue graph node."""

from phase_agent.graphs.dialogue.commands import _deepseek_switch_reply


def handle(self, user_message, state, messages, conversation_id):
    from phase_agent.runtime.resolve_deepseek_model_request import resolve_deepseek_model_request

    requested_model = resolve_deepseek_model_request(user_message)
    if requested_model:
        if not callable(self.deepseek_model_switcher):
            return "此运行时未配置 DeepSeek 模型切换接口；未修改任何设置。"
        try:
            selected_model, client = self.deepseek_model_switcher(requested_model)
        except (OSError, TypeError, ValueError) as error:
            return f"DeepSeek 模型切换失败：{type(error).__name__}: {error}。运行时设置未更改。"
        self.workflow_kwargs["agent_client"] = client
        return _deepseek_switch_reply(selected_model)
    return None
