"""Resolve editable draft paths and recover invalid workspace settings."""

from dataclasses import dataclass
from pathlib import Path

from phase_agent.runtime.configuration_chat import (
    BOOTSTRAP_HINTS,
    normalize_workspace_path,
    safe_config_filename,
)
from phase_agent.runtime.runtime_config_io import _resolve_path
from phase_agent.configuration.session.editable_config_location import editable_config_location


@dataclass(frozen=True)
class DraftWorkspacePaths:
    editable_config_path: Path | None
    editable_config_filename: str
    workspace_root: Path
    editable_config_directory: str = ""


def prepare_draft_workspace_paths(
    session, settings, *, base, resolved_session, workspace_root, workspace_defaults
):
    """Preserve existing recovery writes to session; never overwrite an editable file."""
    editable_path_setting = (
        settings.get("editable_config_draft_path") or "search_config.project.json"
    )
    editable_config_filename = safe_config_filename(Path(editable_path_setting).name)
    # Absolute legacy settings continue to use their existing source; new
    # workspace-relative locations retain the configured config/ subdirectory.
    configured = Path(editable_path_setting)
    editable_directory = "" if configured.is_absolute() else configured.parent.as_posix()
    if editable_directory == ".":
        editable_directory = ""
    editable_config_location(workspace_root, Path(editable_directory) / editable_config_filename)
    setup_stage = session.get("setup_stage")
    if setup_stage == "json_ready":
        storage = (session.get("config") or {}).get("storage") or {}
        try:
            workspace_root = normalize_workspace_path(
                storage.get("workspace_root"),
                base_directory=base,
            )
            saved_path = session.get("editable_config_json_path")
            preferred_path = editable_config_location(
                workspace_root, Path(editable_directory) / editable_config_filename
            )
            if preferred_path.is_file():
                draft_candidate = preferred_path
                if saved_path and str(saved_path) != str(preferred_path):
                    session["editable_config_json_path"] = str(preferred_path)
                    session.pop("agent_reviewed_revision", None)
                    session.pop("agent_reviewed_config_hash", None)
                    session.pop("agent_reviewed_config_digest", None)
                    session.pop("agent_reviewed_mother_digest", None)
                    session.setdefault("dialogue", []).append(
                        {
                            "type": "editable_config_source_changed",
                            "old_path": saved_path,
                            "new_path": str(preferred_path),
                            "reason": "运行时指定的设置文件已存在；以它作为下一次导入来源。",
                        }
                    )
                    from phase_agent.configuration.session.save_config_session import (
                        save_config_session,
                    )

                    save_config_session(session, resolved_session)
            elif saved_path:
                draft_candidate = normalize_workspace_path(saved_path, base_directory=base)
            else:
                draft_candidate = preferred_path
            if (
                not draft_candidate.is_relative_to(workspace_root)
                or draft_candidate.suffix.lower() != ".json"
            ):
                raise ValueError("设置 JSON 必须位于已确认工作区内")
        except (TypeError, ValueError, OSError):
            # Recover sessions written by older versions that accepted arbitrary
            # multi-line input as a path. Keep unrelated draft fields and files.
            session.pop("editable_config_json_path", None)
            session.pop("pending_workspace_root", None)
            session["setup_stage"] = "awaiting_storage_path"
            session.setdefault("config", {}).pop("storage", None)
            session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
            session.setdefault("dialogue", []).append(
                {
                    "type": "workspace_path_recovery",
                    "role": "assistant",
                    "message": (
                        "检测到上次保存的工作区路径格式无效，已清除该路径并恢复到路径选择。"
                        "没有删除或覆盖任何文件；请重新发送一行本地工作区目录路径。"
                    ),
                }
            )
            from phase_agent.configuration.session.save_config_session import save_config_session

            save_config_session(session, resolved_session)
            setup_stage = "awaiting_storage_path"

    elif setup_stage == "awaiting_storage_confirmation":
        pending_root = session.get("pending_workspace_root")
        try:
            if not pending_root:
                raise ValueError("缺少待确认工作区")
            normalize_workspace_path(pending_root, base_directory=base)
        except (TypeError, ValueError, OSError):
            session.pop("pending_workspace_root", None)
            session.pop("editable_config_json_path", None)
            session["setup_stage"] = "awaiting_storage_path"
            from phase_agent.configuration.session.save_config_session import save_config_session

            save_config_session(session, resolved_session)
            setup_stage = "awaiting_storage_path"

    saved_draft_path = session.get("editable_config_json_path")
    if setup_stage in {"awaiting_storage_path", "awaiting_storage_confirmation"}:
        editable_config_path = None
    elif setup_stage == "json_ready":
        editable_config_path = draft_candidate
    elif saved_draft_path:
        try:
            editable_config_path = normalize_workspace_path(saved_draft_path, base_directory=base)
        except (TypeError, ValueError, OSError):
            editable_config_path = None
    elif not setup_stage:
        if "storage" in (session.get("config") or {}) and settings.get(
            "editable_config_draft_path"
        ):
            editable_config_path = _resolve_path(settings["editable_config_draft_path"], base)
        else:
            session_setting = Path(settings.get("config_session_path", "local/config_session.json"))
            editable_config_path = _resolve_path(
                session_setting.with_name(editable_config_filename), base
            )
    else:
        editable_config_path = None
    if session.get("status") == "draft" and editable_config_path is not None:
        from phase_agent.configuration.session.create_editable_config_json import (
            create_editable_config_json,
        )
        from phase_agent.configuration.session.project_config_json import create_project_config_json

        # Create the template once. A user-edited file is never overwritten on restart.
        try:
            creator = (
                create_project_config_json
                if editable_config_path.name.endswith(".project.json")
                else create_editable_config_json
            )
            creator(
                editable_config_path,
                session.get("config") or {},
                bootstrap_hints=session.get("bootstrap_hints") or BOOTSTRAP_HINTS,
                workspace_defaults=workspace_defaults,
            )
        except (OSError, TypeError, ValueError):
            session.pop("editable_config_json_path", None)
            session.pop("pending_workspace_root", None)
            session["setup_stage"] = "awaiting_storage_path"
            session.setdefault("config", {}).pop("storage", None)
            session.setdefault("default_parameter_prompt", {})["status"] = "skipped"
            session.setdefault("dialogue", []).append(
                {
                    "type": "workspace_path_recovery",
                    "role": "assistant",
                    "message": (
                        "上次工作区不可写，已恢复到路径选择。没有删除已有文件；"
                        "请重新发送一行可写的本地工作区目录路径。"
                    ),
                }
            )
            from phase_agent.configuration.session.save_config_session import save_config_session

            save_config_session(session, resolved_session)
            editable_config_path = None
    return DraftWorkspacePaths(
        editable_config_path, editable_config_filename, workspace_root, editable_directory
    )
