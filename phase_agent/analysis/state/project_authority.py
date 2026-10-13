"""Resolve current project facts without promoting drafts or historical models."""

from copy import deepcopy
from pathlib import Path

from phase_agent.tools.step_runner.file_protocol import content_id
from phase_agent.analysis.state.model_epoch import model_epoch

AUTHORITY_RULE = (
    "current_project 是程序核实的本轮唯一版本与设置来源。active_model 是当前运行模型；"
    "initial_model_version 是确认配置中的初始模型，模型激活后允许与当前模型不同；"
    "candidate_models 尚未激活，draft 尚未生效。历史聊天、旧方案及旧计算只能作背景，"
    "不得覆盖本轮事实或跨项目引用。设置修改、确认和模型激活的结果以程序回执为准。"
    "display_name 是当前项目的交互名称，称呼项目时优先使用它；project 和 project_path 是内部身份。"
)


def confirmed_configuration(state, session=None):
    """Existing runs use their bound snapshot; unconfirmed config is never a fallback."""
    if state.get("confirmed_config") is not None:
        return deepcopy(state["confirmed_config"] or {})
    snapshot = (session or {}).get("confirmed_snapshot") or {}
    if (
        state.get("confirmed_config_version")
        and snapshot.get("config_version") != state["confirmed_config_version"]
    ):
        return {}
    return deepcopy(snapshot.get("config") or {})


def project_identity(state):
    """Content changes under an unchanged version label must invalidate approval."""
    config = state.get("confirmed_config") or {}
    active = state.get("active_model") or {}
    return {
        "config_version": state.get("confirmed_config_version"),
        "config_fingerprint": content_id(config, "config-content"),
        "model_version": state.get("active_model_version") or active.get("version"),
        "model_fingerprint": content_id(active, "active-model"),
        "model_consistent": not (
            state.get("active_model_version")
            and active.get("version")
            and state["active_model_version"] != active["version"]
        ),
    }


def current_project(state, session=None, *, state_path=None):
    session = session or {}
    config = confirmed_configuration(state, session)
    snapshot = session.get("confirmed_snapshot") or {}
    version = state.get("confirmed_config_version") or snapshot.get("config_version")
    root = (config.get("storage") or {}).get("workspace_root")
    if state_path is not None:
        # workflow_state can be a junction to diagrams/<name>/workflow_state.
        # Its lexical project path remains the workspace identity shown to users.
        path = Path(state_path).absolute()
        root = str(
            path.parent.parent if path.parent.name == "workflow_state" else path.parent
        )
    initial = (config.get("calculation") or {}).get("mlip_version")
    active = deepcopy(state.get("active_model") or {})
    active_version = state.get("active_model_version") or active.get("version")
    issues = []
    if active.get("version") and active_version != active["version"]:
        issues.append("当前模型版本与模型文件登记不一致，不能推断实际使用的模型文件。")
        active = {"version": active_version, "model_path": None}
    elif active_version and active_version == initial and not active:
        active = deepcopy(config.get("mlip") or {})
    active = {
        key: value
        for key, value in active.items()
        if key
        in {
            "version",
            "model_path",
            "model_sha256",
            "sha256",
            "checksum",
            "backend",
            "device",
            "cutoff",
            "model_family",
        }
    }
    active["version"] = active_version
    if (
        state.get("confirmed_config_version")
        and snapshot.get("config_version")
        and (
            state["confirmed_config_version"] != snapshot["config_version"]
            or state.get("confirmed_config") != snapshot.get("config")
        )
    ):
        issues.append(
            "运行绑定配置与配置会话快照不同；本轮以运行绑定配置为准，不能混用。"
        )
    draft_config = (
        (snapshot.get("config") or {})
        if session.get("status") == "confirmed"
        else session.get("config") or {}
    )
    draft_pending = bool(session) and session.get("status") != "confirmed"
    draft_changed = bool(session) and draft_config != config
    changed_paths = configuration_changes(config, draft_config) if draft_changed else []
    settings = {
        key: deepcopy(config[key])
        for key in (
            "run",
            "calculation",
            "budgets",
            "generation_actions",
            "round_strategy",
            "convergence",
            "python_environments",
            "dft",
            "mlip_finetune",
            "bohb",
            "qbc",
            "frozen_parameters",
        )
        if key in config
    }
    system = config.get("system") or {}
    settings["system"] = {
        "boundary": {
            key: deepcopy(value)
            for key, value in (system.get("boundary") or {}).items()
            if key in {"P", "TM_ratio", "x_range"}
        },
        "H_generation": {
            key: deepcopy(value)
            for key, value in (system.get("H_generation") or {}).items()
            if key
            in {"enabled", "size_min", "size_max", "size_step", "first_round_max_det_H"}
        },
    }
    if active_version:
        settings.setdefault("calculation", {})["mlip_version"] = active_version
    settings["mlip"] = {**deepcopy(config.get("mlip") or {}), **deepcopy(active)}
    if isinstance((settings.get("bohb") or {}).get("scope"), dict):
        settings["bohb"]["scope"].pop("H", None)
    from phase_agent.runtime.project_display_name import display_name
    return {
        "project": Path(root).name if root else None,
        "display_name": display_name(Path(root).resolve()) if root else None,
        "project_path": root,
        "epoch": model_epoch(state, active_version),
        "active_model": active,
        "initial_model_version": initial,
        "candidate_models": [
            {"version": key, "status": row.get("status"), "activated": False}
            for key, row in (state.get("candidate_models") or {}).items()
            if key != active_version
            and row.get("status") not in {"activated", "rejected"}
        ],
        "configuration": {
            "version": version,
            "source": "state.confirmed_config"
            if state.get("confirmed_config") is not None
            else "session.confirmed_snapshot"
            if snapshot
            else "unconfirmed",
            "fingerprint": content_id(config, "config-content"),
            "effective_settings": settings,
        },
        "draft": {
            "revision": session.get("draft_revision"),
            "status": "confirmed_not_applied"
            if draft_changed and session.get("status") == "confirmed"
            else "pending_confirmation"
            if draft_pending or draft_changed
            else "no_changes",
            "effective": False,
            "changed_paths": changed_paths[:20],
        },
        "pending_plans": [
            {
                "plan_id": key,
                "revision": row.get("revision", 0),
                "tool": (row.get("agent_proposal") or {}).get("recommended_action"),
                "bound_identity": deepcopy(row.get("project_identity")),
                "context_changed": bool(row.get("project_identity"))
                and row["project_identity"] != project_identity(state),
            }
            for key, row in (state.get("pending_execution_policies") or {}).items()
        ],
        "consistency_issues": issues,
        "authority": AUTHORITY_RULE,
    }


def _changed_paths(before, after, prefix=""):
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        return [
            path
            for key in sorted(set(before) | set(after))
            for path in _changed_paths(
                before.get(key), after.get(key), f"{prefix}.{key}" if prefix else key
            )
        ]
    return [prefix]


def configuration_changes(before, after):
    """Generated H/reference caches are outputs, not edits to their source settings."""

    def settings(config):
        value = deepcopy(config)
        system = value.get("system") or {}
        if (system.get("H_generation") or {}).get("enabled"):
            (system.get("boundary") or {}).pop("H", None)
        if system.get("phase_reference_directory"):
            system.pop("phase_references", None)
        return value

    return _changed_paths(settings(before), settings(after))
