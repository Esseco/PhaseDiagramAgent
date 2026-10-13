"""Read the current saved configuration session for each conversation turn."""

from pathlib import Path
from phase_agent.tools.step_runner.file_protocol import read_json
from phase_agent.analysis.state.project_authority import current_project


def current_config_session(handler):
    kwargs = getattr(handler, "workflow_kwargs", None) or {}
    supplied = kwargs.get("config_session") or {}
    path = getattr(handler, "config_session_path", None) or kwargs.get("config_session_path")
    if path is not None and Path(path).is_file():
        return read_json(path, {}) or {}
    return supplied


def handler_project_context(handler, state):
    from phase_agent.analysis.state.project_authority import (
        confirmed_configuration,
        configuration_changes,
    )
    from phase_agent.configuration.session.load_editable_config_json import (
        load_editable_config_json,
    )

    session = current_config_session(handler)
    card = current_project(state, session, state_path=handler.state_path)
    root = Path(card["project_path"])
    source = getattr(handler, "editable_config_path", None) or session.get(
        "editable_config_json_path"
    )
    source = Path(source) if source else root / "parameters/search_config.project.json"
    if source.is_file():
        try:
            source.resolve().relative_to(root.resolve())
            editable = load_editable_config_json(source, confirmed_configuration(state, session))
            from phase_agent.runtime.editable_config_baseline import (
                editable_comparison_baseline,
            )

            expected = editable_comparison_baseline(
                confirmed_configuration(state, session),
                source,
                getattr(handler, "base_directory", None) or root,
            )
            changed = configuration_changes(expected, editable)
            card["editable_file"] = {
                "path": str(source),
                "effective": False,
                "changed_paths": changed[:20],
            }
            if changed and card["draft"]["status"] == "no_changes":
                card["draft"]["status"] = "file_changes_not_effective"
        except (OSError, ValueError, TypeError, KeyError) as error:
            card["editable_file"] = {
                "path": str(source),
                "effective": False,
                "error": str(error),
            }
    return card


def project_context_line(card):
    parts = [card.get("project") or "当前项目"]
    if card.get("epoch"):
        parts.append(card["epoch"])
    version = str((card.get("active_model") or {}).get("version") or "未登记")
    label = "微调模型" if version.startswith("remote-") and card.get("epoch") else version
    parts.append("当前模型 " + label)
    draft = card.get("draft") or {}
    if draft.get("status") == "file_changes_not_effective":
        parts.append("设置文件修改未生效")
    elif draft.get("status") == "confirmed_not_applied":
        parts.append("新配置已确认但尚未应用")
    elif draft.get("status") == "pending_confirmation":
        parts.append("设置修改待确认")
    return " · ".join(parts)
