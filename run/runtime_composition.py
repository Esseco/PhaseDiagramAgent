"""Compose configured scientific dependencies without owning chat or HTTP."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from execution_layer.step_runner.file_protocol import write_json
from run.runtime_config_io import (
    _has_history, _resolve_path, _load_json_object, _import_reference,
    _reject_secrets, _session_from_state,
)


def compose_runtime(config_path=None, *, chat_handler_factory,
                    model_switcher_factory, runtime_factory):
    """Compose the built-in runtime from a local, non-secret JSON file."""
    configured = config_path or os.environ.get("PHASE_AGENT_RUNTIME_CONFIG") or "run/agent_runtime.json"
    config_file = Path(configured).resolve()
    settings = _load_json_object(config_file, "Agent 运行时配置")
    _reject_secrets(settings)
    from config_layer.session.load_config_session import load_config_session
    from run.runtime_clients import create_optional_runtime_client
    from run.runtime_client_settings import (
        configuration_client_settings, search_client_settings, intent_client_settings, CONFIG_INTENT_PROMPT,
    )
    base = config_file.parent
    missing = [key for key in ("state_path", "ledger_path") if not settings.get(key)]
    if missing:
        raise ValueError(f"运行时配置缺少 {missing}；请编辑 {config_file}")

    state_path = _resolve_path(settings["state_path"], base)
    ledger_path = _resolve_path(settings["ledger_path"], base)
    phase_path = (_resolve_path(settings["phase_references_path"], base)
                  if settings.get("phase_references_path") else None)
    state = _load_json_object(state_path, "state") if state_path.is_file() else {}

    resolved_session = _resolve_path(
        settings.get("config_session_path", "local/config_session.json"), base
    )
    from config_layer.session.resolve_workspace_paths import default_workspace_storage
    path_keys = {
        "state": ("state_path", True),
        "ledger": ("ledger_path", True),
        "branch_energy_pool_ledger": ("branch_energy_pool_ledger_path", True),
        "phase_diagrams": ("phase_diagram_directory", False),
        "approvals": ("approval_directory", False),
        "approved_batches": ("local_action_directory", False),
        "upload_batches": ("manual_upload.batches_directory", False),
        "new_runs": ("new_runs_directory", False),
    }
    configured_paths = {}
    workspace_candidates = [resolved_session.parent.resolve()]
    for key, (setting_key, is_file) in path_keys.items():
        value = settings.get(setting_key)
        if setting_key == "manual_upload.batches_directory":
            value = (settings.get("manual_upload") or {}).get("batches_directory")
        if value:
            resolved_value = _resolve_path(value, base)
            configured_paths[key] = resolved_value
            workspace_candidates.append(resolved_value.parent if is_file else resolved_value)
    roots = workspace_candidates
    try:
        workspace_root = Path(os.path.commonpath([str(path) for path in roots]))
    except ValueError:
        workspace_root = resolved_session.parent
    workspace_defaults = default_workspace_storage(
        workspace_root, path_overrides=configured_paths,
    )
    from run.runtime_session import load_or_initialize_runtime_session
    session = load_or_initialize_runtime_session(
        state, state_path=state_path, resolved_session=resolved_session,
        workspace_defaults=workspace_defaults,
    )

    if session.get("status") != "confirmed":
        from run.configuration_chat import (
            CONFIG_AGENT_SYSTEM_PROMPT, ConfigurationChatHandler,
        )
        from run.draft_workspace_paths import prepare_draft_workspace_paths
        draft_paths = prepare_draft_workspace_paths(
            session, settings, base=base, resolved_session=resolved_session,
            workspace_root=workspace_root, workspace_defaults=workspace_defaults,
        )
        editable_config_path = draft_paths.editable_config_path
        editable_config_filename = draft_paths.editable_config_filename
        workspace_root = draft_paths.workspace_root
        deepseek = settings.get("deepseek") or {}
        client_parameters = configuration_client_settings(deepseek, system_prompt=CONFIG_AGENT_SYSTEM_PROMPT)
        agent_client = create_optional_runtime_client(client_parameters)
        model_switcher = model_switcher_factory(
            config_file, deepseek, system_prompt=CONFIG_AGENT_SYSTEM_PROMPT,
            thinking=deepseek.get("configuration_thinking", "disabled"),
        )
        workflow_kwargs = {
            "state_path": str(state_path), "config_session": session,
            "config_session_path": str(resolved_session),
        }
        handler = ConfigurationChatHandler(
            workflow_kwargs, config_session_path=resolved_session, base_directory=base,
            phase_references_path=phase_path, editable_config_path=editable_config_path,
                editable_config_filename=editable_config_filename,
                editable_config_directory=draft_paths.editable_config_directory,
                workspace_root_default=workspace_root,
            agent_client=agent_client,
            runtime_factory=lambda: runtime_factory(config_file),
            current_deepseek_model=deepseek.get("model", "deepseek-v4-pro"),
            deepseek_model_switcher=model_switcher,
        )
        handler.runtime_config_path = config_file
        return handler

    snapshot = session.get("confirmed_snapshot") or {}
    if not snapshot.get("config") or not snapshot.get("config_version"):
        raise ValueError("配置会话不是完整的 confirmed snapshot；不会自动确认配置")

    from config_layer.session.resolve_workspace_paths import resolve_workspace_paths
    effective_config = deepcopy(snapshot["config"])
    if settings.get("local_path_relocations"):
        from config_layer.session.relocate_workspace_paths import relocate_workspace_paths
        effective_config = relocate_workspace_paths(effective_config, settings["local_path_relocations"], base)
    if settings.get("runtime_storage_override") is not None:
        effective_config["storage"] = deepcopy(settings["runtime_storage_override"])
    if "storage" not in effective_config:
        # Older confirmed snapshots retain their original local workspace by default.
        effective_config["storage"] = workspace_defaults
    resolved_storage = resolve_workspace_paths(effective_config, base_directory=base)
    selected_state_path = resolved_storage["state"]
    selected_ledger_path = resolved_storage["ledger"]
    for label, previous, selected in (
        ("state", state_path, selected_state_path),
        ("ledger", ledger_path, selected_ledger_path),
    ):
        if previous.resolve() != selected.resolve() and previous.exists():
            raise ValueError(
                f"工作区路径变更检测到已有 {label} 文件：{previous}。"
                f"不会自动移动或忽略它；请在配置 JSON 中将 storage.paths.{label} 指回原位置，"
                "或先由用户手动迁移并核对后再继续。"
            )
    state_path, ledger_path = selected_state_path, selected_ledger_path
    state = _load_json_object(state_path, "state") if state_path.is_file() else {}
    if state and state.get("confirmed_config_version") != snapshot.get("config_version"):
        from config_layer.runtime.authorize_generation_policy_revision import (
            authorize_generation_policy_revision,
        )
        migration = authorize_generation_policy_revision(state, snapshot)
        if migration["status"] == "rebound":
            state = migration["state"]
            write_json(state_path, state)

    configured_references = ((effective_config.get("system") or {}).get("phase_references") or {})
    if phase_path and phase_path.is_file():
        phase_references = _load_json_object(phase_path, "相图参考")
    else:
        phase_references = deepcopy(configured_references)
    if not isinstance(phase_references, dict):
        raise ValueError("相图参考必须是映射")
    phase_references = {
        key: (str(_resolve_path(value, base)) if isinstance(value, str) else value)
        for key, value in phase_references.items()
    }

    from data_layer.ledger.phase_data_manager import PhaseDataManager
    from run.default_run_config import default_run_config
    if ledger_path.is_file():
        manager = PhaseDataManager.load(ledger_path)
        saved_system = manager.data.get("system_config") or {}
        saved_space = saved_system.get("configuration_space") or {}
        active_space = (effective_config.get("system") or {}).get("configuration_space") or {}
        if saved_space != active_space:
            if manager.data.get("branches") or manager.data.get("structures"):
                raise ValueError(
                    "现有台账的问题变量角色与已确认配置不同；不能在原台账上改变 branch 编号。"
                    "请建立新运行或明确迁移旧数据。")
            manager.data["system_config"] = deepcopy(effective_config["system"])
            if (active_space.get("roles") or {}).get("T") == "fixed":
                manager.data["system_config"]["branch_schema"]["fields"] = ["P", "H", "x"]
    else:
        boundary = ((snapshot["config"].get("system") or {}).get("boundary"))
        if not isinstance(boundary, dict):
            raise ValueError(f"台账不存在且 confirmed config 没有 system.boundary；请配置 {ledger_path}")
        manager = PhaseDataManager(boundary, system_config=effective_config.get("system"))
        manager.save(ledger_path)

    runtime_config = default_run_config()
    runtime_config["system_config"] = deepcopy(effective_config["system"])
    runtime_config.update({"state_path": str(state_path), "ledger_path": str(ledger_path)})
    for key in ("structure_directory", "phase_diagram_directory", "work_directory",
                "branch_energy_pool_ledger_path", "approval_directory",
                "local_action_directory", "mlip"):
        if key in settings:
            value = deepcopy(settings[key])
            if key != "mlip" and isinstance(value, str):
                value = str(_resolve_path(value, base))
            runtime_config[key] = value
    runtime_config.update({
        "structure_directory": str(resolved_storage["structures"]),
        "state_path": str(state_path),
        "ledger_path": str(ledger_path),
        "branch_energy_pool_ledger_path": str(resolved_storage["branch_energy_pool_ledger"]),
        "phase_diagram_directory": str(resolved_storage["phase_diagrams"]),
        "work_directory": str(resolved_storage["work"]),
        "approval_directory": str(resolved_storage["approvals"]),
        "local_action_directory": str(resolved_storage["approved_batches"]),
        "upload_batches_directory": str(resolved_storage["upload_batches"]),
    })
    runtime_config.setdefault("qbc", {})["output_path"] = str(resolved_storage["qbc_results"])
    deepseek = settings.get("deepseek") or {}
    client_parameters = search_client_settings(deepseek)
    agent_client = create_optional_runtime_client(client_parameters)
    model_switcher = model_switcher_factory(
        config_file, deepseek, system_prompt=None, thinking=deepseek.get("thinking"),
    )
    kwargs = {"manager": manager, "phase_references": phase_references,
              "run_config": runtime_config, "config_session": session,
              "state_path": str(state_path), "agent_client": agent_client}
    kwargs["config_session_path"] = str(resolved_session)
    from run.runtime_backends import compose_runtime_backends
    kwargs = compose_runtime_backends(kwargs, settings, resolved_storage)

    def new_run():
        if (settings.get("dispatcher_factory") or settings.get("task_runner_factory")
                or settings.get("runtime_adapters") or (settings.get("manual_upload") or {}).get("task_preparer_factory")):
            raise ValueError("自定义后端尚无新运行路径隔离协议；请使用独立项目配置，未新建或修改运行。")
        root = resolved_storage["new_runs"]
        run_dir = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        new_state = run_dir / "state.json"
        new_ledger = run_dir / "phase_data.json"
        fresh = PhaseDataManager(manager.boundary, system_config=manager.data.get("system_config"))
        run_dir.mkdir(parents=True, exist_ok=False)
        fresh.save(new_ledger)
        initial_state = {
            "confirmed_config": deepcopy(snapshot["config"]),
            "confirmed_config_version": snapshot["config_version"],
        }
        temporary = new_state.with_name(new_state.name + ".tmp")
        temporary.write_text(json.dumps(initial_state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        temporary.replace(new_state)
        from run.isolated_run_paths import isolate_runtime_outputs, publish_run_descriptor
        _, fresh_storage = isolate_runtime_outputs(runtime_config, run_dir)
        descriptor = publish_run_descriptor(settings, session, fresh_storage, source_config=config_file)
        return runtime_factory(descriptor)

    configured_mode = settings.get("execution_mode", "debug")
    execution_mode = "interactive" if configured_mode == "debug" else "autonomous"
    if configured_mode not in {"debug", "automatic"}:
        raise ValueError("运行时 execution_mode 必须是 debug 或 automatic")
    intent_parameters = intent_client_settings(deepseek,
            system_prompt=CONFIG_INTENT_PROMPT,
        )
    intent_client = create_optional_runtime_client(intent_parameters)

    def start_config_revision(run_state):
        from config_layer.session.begin_config_revision import begin_config_revision
        from config_layer.session.save_config_session import save_config_session
        baseline = _session_from_state(run_state)
        if baseline is None:
            raise ValueError("当前运行没有可核实的已确认配置")
        baseline["config"]["storage"] = deepcopy(effective_config["storage"])
        previous = load_config_session(resolved_session) if resolved_session.is_file() else {}
        baseline["draft_revision"] = max(int(previous.get("draft_revision") or 0),
                                          int(str(baseline["confirmed_snapshot"]["config_version"])
                                              .split("-")[1]))
        baseline["dialogue"] = deepcopy((previous.get("dialogue") or [])[-30:])
        baseline["editable_config_json_path"] = str(
            Path(baseline["config"]["storage"]["workspace_root"]) /
            (settings.get("editable_config_draft_path") or "search_config.project.json")
        )
        revised = begin_config_revision(baseline, reason="用户在搜索对话中要求修改配置")
        save_config_session(revised, resolved_session)
        current = deepcopy(run_state)
        if current.get("pending_execution_policies"):
            current.setdefault("cancelled_proposals", []).extend({
                "invocation_id": key, "reason": "config_revision_started",
                "config_version": current.get("confirmed_config_version"),
            } for key in current["pending_execution_policies"])
            current["pending_execution_policies"] = {}
            write_json(state_path, current)
        return runtime_factory(config_file)

    handler = chat_handler_factory(
        kwargs, history_prompt=_has_history(state, manager),
        new_run_factory=new_run, deepseek_model_switcher=model_switcher,
        execution_mode=execution_mode, config_revision_factory=start_config_revision,
        config_intent_client=intent_client,
    )
    if settings.get("knowledge_library_root"):
        handler.knowledge_library_root = str(_resolve_path(settings["knowledge_library_root"], base))
    handler.runtime_config_path = config_file
    return handler

