"""Route configuration changes to a draft, never silently enable training."""

from copy import deepcopy
from phase_agent.configuration.session.create_config_draft import create_config_draft
from phase_agent.configuration.session.apply_config_revision import apply_config_revision
from phase_agent.decisions.strategy.apply_strategy_adjustment import apply_strategy_adjustment


def request_strategy_revision(*, action, context):
    state = deepcopy(context.get("event_state") or {})
    config = context.get("effective_config") or {}
    params = action.get("parameters") or {}
    patch = params.get("patch") or {}
    if params.get("request_configuration_revision") is True:
        if not patch or any(not _known_path(config, key) for key in patch):
            return {
                "status": "rejected",
                "state": state,
                "reason": "配置草稿必须指定现有配置字段及新值",
            }
        if "mlip_finetune.enabled" in patch and type(patch["mlip_finetune.enabled"]) is not bool:
            return {
                "status": "rejected",
                "state": state,
                "reason": "mlip_finetune.enabled必须为布尔值",
            }
        if patch == {"mlip_finetune.enabled": True}:
            # Approval authorizes input preparation only, not training or activation.
            from phase_agent.tools.workflows.create_workflow_handlers import _update_mlip

            return _update_mlip(
                action={
                    **action,
                    "parameters": {"action": "RETRAIN_MLIP", "prepare_inputs_only": True},
                },
                context={
                    **context,
                    "effective_config": config,
                    "mlip_trainer": None,
                    "model_update_handler": None,
                },
            )
        draft = apply_config_revision(
            create_config_draft(config),
            patch,
            reasons={key: action.get("reason") for key in patch},
            author="agent_proposal",
        )
        state["requested_config_revision"] = {
            "draft_session": draft,
            "patch": patch,
            "reason": action.get("reason"),
            "task_key": action.get("task_key"),
            "source_config_version": state.get("confirmed_config_version"),
        }
        return {
            "status": "configuration_revision_required",
            "state": state,
            "reason": "配置修订草稿已保存到state.requested_config_revision；请在配置对话复核并确认patch，再批准迁移。确认前不启用微调、不训练。",
        }
    return apply_strategy_adjustment(state, patch, config, bounds=params.get("bounds"))


def _known_path(config, path):
    node = config
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True
