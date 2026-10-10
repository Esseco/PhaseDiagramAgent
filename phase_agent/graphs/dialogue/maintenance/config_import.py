"""config_import operation adapter; invoked as a named dialogue graph node."""


def handle(self, user_message, state, messages, conversation_id):
    from phase_agent.runtime.configuration_chat import _is_config_json_import_command

    if self.config_delegate is None and _is_config_json_import_command(user_message):
        if not callable(self.config_revision_factory):
            return "配置重新读取入口未配置；仍保留当前已确认版本，未执行计算。"
        try:
            self.config_delegate = self.config_revision_factory(state)
        except (OSError, TypeError, ValueError) as error:
            return f"无法进入配置检查：{error}。未执行计算。"
        return self.config_delegate(messages, conversation_id=conversation_id)
    return None
