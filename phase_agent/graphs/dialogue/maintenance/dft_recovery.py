"""dft_recovery operation adapter; invoked as a named dialogue graph node."""

from phase_agent.graphs.invocation_context import new_invocation
from phase_agent.runtime.workflow_reply_presentation import format_workflow_reply


def handle(self, user_message, state, messages, conversation_id):
    from phase_agent.tools.state.dft_recovery_decision import classify_dft_recovery_reply

    recovery_decision = classify_dft_recovery_reply(user_message, state)
    if recovery_decision is not None and self.config_delegate is None:
        self.conversation_id = conversation_id
        result = self._run(
            new_invocation(), None, user_message, dft_recovery_decision=recovery_decision
        )
        prefix = (
            "已记录：本轮剩余 DFT 不再等待；未取消超算任务，下一步仍需正常审批。\n"
            if recovery_decision["decision"] == "close"
            else "已记录：继续等待本轮剩余 DFT。\n"
        )
        if not any(
            row.get("question_id") == recovery_decision["question_id"]
            and row.get("decision") == recovery_decision["decision"]
            for row in (result.get("state") or {}).get("dft_recovery_decisions") or []
        ):
            prefix = ""
        return prefix + format_workflow_reply(result, self.state_path)
    return None
