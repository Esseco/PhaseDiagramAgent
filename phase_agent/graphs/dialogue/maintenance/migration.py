"""migration operation adapter; invoked as a named dialogue graph node."""

import uuid
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply
from phase_agent.graphs.dialogue.commands import _is_config_migration_approval


def handle(self, user_message, state, messages, conversation_id):
    if _is_config_migration_approval(user_message):
        self.conversation_id = conversation_id
        invocation_id = f"config-migration-{uuid.uuid4().hex}"
        result = self._run(
            invocation_id,
            None,
            user_message,
            approve_config_migration=True,
        )
        return format_workflow_reply(result, self.state_path)
    return None
