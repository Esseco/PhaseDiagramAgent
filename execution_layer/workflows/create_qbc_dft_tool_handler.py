"""Adapt the QBC-agent flow to autonomous_platform's tool registry."""

from execution_layer.workflows.run_qbc_dft_decision_flow import run_qbc_dft_decision_flow


def create_qbc_dft_tool_handler(*, candidates_provider, config_provider, submitter=None, qbc_evaluator=None, agent_client=None):
    def handler(*, action, context):
        confirmed = context["confirmed_config"]
        return run_qbc_dft_decision_flow(candidates_provider(action, context), context.get("search_state", {}), config=config_provider(confirmed), config_version=context["config_version"], dft_parameters=(confirmed.get("dft") or {}).get("parameters") or {}, context=context, agent_client=agent_client, qbc_evaluator=qbc_evaluator, dft_submitter=submitter, invocation_id=action.get("invocation_id") or action.get("task_key"))
    return handler
