"""inspection operation adapter; invoked as a named dialogue graph node."""

from phase_agent.tools.step_runner.file_protocol import write_json
from phase_agent.graphs.dialogue.commands import format_status_reply, _is_status_command


def handle(self, user_message, state, messages, conversation_id):
    # Read-only inspection must precede history recovery and all mutations.
    if self.config_delegate is None:
        from phase_agent.runtime.plan_queries import is_plan_query, pending_plan_reply
        from phase_agent.graphs.dialogue.support import bind_presented_proposal

        if is_plan_query(user_message):
            bind_presented_proposal(self, state)
            return pending_plan_reply(state)
    if self.config_delegate is None:
        from phase_agent.runtime.artifact_queries import finetune_location_reply

        location = finetune_location_reply(user_message, state)
        if location is not None:
            return location
    if self.config_delegate is None and _is_status_command(user_message):
        return format_status_reply(state)
    if self.config_delegate is None:
        from phase_agent.tools.local.review_candidate_command import review_candidate_command
        from phase_agent.tools.local.review_direction_command import review_direction_command

        direction = review_direction_command(user_message, state, self.state_path)
        if direction is not None:
            write_json(self.state_path, direction["state"])
            return direction["reason"]
        review = review_candidate_command(user_message, state, state_path=self.state_path)
        if review is not None:
            write_json(self.state_path, review["state"])
            return review["reason"]
    from phase_agent.runtime.configuration_chat import _is_config_json_import_command

    return None
