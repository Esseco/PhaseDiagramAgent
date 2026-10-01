"""Load or initialize configuration sessions without inventing run configuration."""

from copy import deepcopy

from config_layer.session.load_config_session import load_config_session
from run.runtime_config_io import _session_from_state


def load_or_initialize_runtime_session(state, *, state_path, resolved_session, workspace_defaults):
    """Recover confirmed state or initialize a draft; preserve existing session files."""
    if not resolved_session.is_file():
        session = _session_from_state(state)
        if session is None:
            if state:
                raise ValueError(
                    f"{state_path} 有运行数据但没有可恢复的 confirmed config；为避免覆盖，先手动恢复配置会话"
                )
            from config_layer.defaults.default_layered_search_config import default_layered_search_config
            from config_layer.session.create_config_draft import create_config_draft
            from config_layer.session.save_config_session import save_config_session
            from run.configuration_chat import BOOTSTRAP_HINTS
            session = create_config_draft(
                default_layered_search_config(), require_workspace_path=True
            )
            session["bootstrap_hints"] = deepcopy(BOOTSTRAP_HINTS)
            save_config_session(session, resolved_session)
    else:
        session = load_config_session(resolved_session)

    if session.get("status") != "confirmed" and not session.get("setup_stage"):
        # Upgrade pre-path-first drafts without discarding their edits or files.
        session["setup_stage"] = "awaiting_storage_path"
        session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
        session.setdefault("dialogue", []).append({
            "type": "workspace_path_question",
            "role": "assistant",
            "message": (
                "请先发送本地工作区根目录路径；我会展示保存位置，"
                "并在你确认后才生成设置 JSON。"
            ),
        })
        from config_layer.session.save_config_session import save_config_session
        save_config_session(session, resolved_session)

    if (session.get("status") != "confirmed"
            and "storage" not in (session.get("config") or {})
            and not session.get("setup_stage")):
        from config_layer.session.apply_config_revision import apply_config_revision
        from config_layer.session.save_config_session import save_config_session
        session = apply_config_revision(
            session, {"storage": workspace_defaults},
            reasons={"storage": "沿用当前本地配置会话目录作为统一工作区默认值；用户可在确认前调整。"},
            author="workspace_storage_default",
        )
        save_config_session(session, resolved_session)
    return session

