"""Create independent diagrams and migrate legacy projects without losing history."""

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil

from phase_agent.persistence.shared_dft import identity, protocol, publish


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def options(root, name):
    return {
        "directory": str(root / "shared_dft"),
        "diagram_id": name,
        "enabled": False,
        "source_projects": [],
    }


def migrate(root, *, backup_directory):
    """Caller stops the idle service and holds an external migration lock."""
    root = Path(root).resolve()
    descriptor = root / "system.json"
    if descriptor.exists():
        return read(descriptor)
    default = root / "diagrams" / "diagram_default"
    if default.exists():
        raise FileExistsError(default)
    state_path = root / "workflow_state/state.json"
    state = read(state_path)
    config = state["confirmed_config"]
    runtime = read(root / "agent_runtime.json")
    backup = Path(backup_directory)
    backup.mkdir(parents=True, exist_ok=True)
    for relative in [
        "agent_runtime.json",
        "README.md",
        "workflow_state/state.json",
        "parameters/config_session.json",
    ]:
        path = root / relative
        if path.is_file():
            destination = backup / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    default.mkdir(parents=True)
    moved = []
    try:
        for name in [
            "agent_memory",
            "analysis_outputs",
            "documentation",
            "history_backups",
            "logs",
            "parameters",
            "structures",
            "submissions",
            "workflow_state",
            "open_webui_runs",
        ]:
            old = root / name
            if not old.exists():
                continue
            target = default / name
            assert old.resolve().is_relative_to(root) and target.resolve().is_relative_to(default)
            old.rename(target)
            moved.append((old, target, False))
        from phase_agent.configuration.session.relocate_workspace_paths import (
            relocate_workspace_paths,
        )

        relocations = [{"from": str(old), "to": str(target)} for old, target, _ in moved]
        operational = {
            key: value for key, value in state.items() if key not in {"confirmed_config"}
        }
        state.update(relocate_workspace_paths(operational, relocations, root))
        runtime["local_path_relocations"] = [
            *(runtime.get("local_path_relocations") or []),
            *relocations,
        ]
        runtime["editable_config_draft_path"] = str(
            default / "parameters/search_config.project.json"
        )
        state["shared_dft"] = options(root, "diagram_default")
        write(default / "workflow_state/state.json", state)
        runtime["shared_dft"] = options(root, "diagram_default")
        from phase_agent.configuration.session.resolve_workspace_paths import (
            default_workspace_storage,
        )

        storage = deepcopy(
            runtime.get("runtime_storage_override")
            or config.get("storage")
            or default_workspace_storage(default)
        )
        storage["workspace_root"] = str(default)
        runtime["runtime_storage_override"] = storage
        for field, key in {
            "state_path": "state",
            "ledger_path": "ledger",
            "new_runs_directory": "new_runs",
            "phase_diagram_directory": "phase_diagrams",
            "approval_directory": "approvals",
            "local_action_directory": "approved_batches",
            "structure_directory": "structures",
            "work_directory": "work",
            "branch_energy_pool_ledger_path": "branch_energy_pool_ledger",
        }.items():
            runtime[field] = str(Path(storage["workspace_root"]) / storage["paths"][key])
        runtime["config_session_path"] = str(default / "parameters/config_session.json")
        if runtime.get("manual_upload"):
            runtime["manual_upload"]["batches_directory"] = str(
                Path(storage["workspace_root"]) / storage["paths"]["upload_batches"]
            )
        write(default / "agent_runtime.json", runtime)
        write(root / "agent_runtime.json", runtime)  # existing launcher is a default-diagram entry
        from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager

        manager = PhaseDataManager.load(default / "workflow_state/ledgers/phase_data.json")
        published = publish(state, manager, config)
        metadata = {
            "schema": 1,
            "system_id": identity(config),
            "dft_protocol": protocol(config),
            "default_diagram": "diagram_default",
            "diagrams": ["diagram_default"],
            "base_config": deepcopy(config),
            "dft_records_imported": published,
        }
        (root / "README.md").write_text(
            '# 同体系独立相图\n\nshared_dft/ 只共享 DFT 数据。diagrams/diagram_default/ 保存原项目全部进度；其他相图的模型、配置、预算、状态与提交文件独立。\n\n原 agent_runtime.json 默认打开 diagram_default；全部数据位于真实目录，不创建旧路径链接。\n\n创建相图：python -m phase_agent.runtime.diagram_projects create --system-root "'
            + str(root)
            + '" --name diagram_B --boundary boundary.json\n\n查看相图：python -m phase_agent.runtime.diagram_projects list --system-root "'
            + str(root)
            + '"\n',
            encoding="utf-8",
        )
        write(descriptor, metadata)
        return metadata
    except BaseException:
        # Restore physical directories without creating compatibility links.
        for old, target, _ in reversed(moved):
            target.rename(old)
        for relative in [
            "agent_runtime.json",
            "README.md",
            "workflow_state/state.json",
            "parameters/config_session.json",
        ]:
            source = backup / relative
            if source.is_file():
                shutil.copy2(source, root / relative)
        (default / "agent_runtime.json").unlink(missing_ok=True)
        default.rmdir()
        raise


def create(root, name, boundary):
    from phase_agent.runtime.project_service import project_lock

    root = Path(root).resolve()
    with project_lock(root / "agent_runtime.json", lock_name="../../diagrams.registry.lock"):
        return _create(root, name, boundary)


def _create(root, name, boundary):
    root = Path(root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise ValueError("相图名称只能包含字母、数字、下划线和短横线")
    descriptor = read(root / "system.json")
    config = deepcopy(descriptor["base_config"])
    config["system"]["boundary"] = deepcopy(boundary)
    if identity(config) != descriptor["system_id"]:
        raise ValueError("只能修改同体系的相图边界，不能改变 TM 配比")
    directory = root / "diagrams" / name
    if directory.exists():
        raise FileExistsError(directory)
    from phase_agent.configuration.session.resolve_workspace_paths import (
        default_workspace_storage,
    )
    from phase_agent.configuration.session.create_config_draft import (
        create_config_draft,
    )
    from phase_agent.configuration.session.confirm_config_snapshot import (
        confirm_config_snapshot,
    )

    config["storage"] = default_workspace_storage(directory)
    session = confirm_config_snapshot(create_config_draft(config), user_confirmed=True)
    if session["status"] != "confirmed":
        raise ValueError("新边界未通过配置校验：" + str(session.get("audit")))
    runtime = read(root / "diagrams" / descriptor["default_diagram"] / "agent_runtime.json")
    runtime_paths = {
        "state_path": "workflow_state/state.json",
        "ledger_path": "workflow_state/ledgers/phase_data.json",
        "config_session_path": "parameters/config_session.json",
        "editable_config_draft_path": "parameters/search_config.project.json",
        "new_runs_directory": "open_webui_runs",
        "phase_diagram_directory": "analysis_outputs",
        "approval_directory": "workflow_state/approvals",
        "local_action_directory": "workflow_state/approved_batches",
        "structure_directory": "structures/candidate_structures",
        "work_directory": "workflow_state/work",
        "branch_energy_pool_ledger_path": "workflow_state/ledgers/branch_energy_pools.json",
    }
    runtime.update(runtime_paths)
    if runtime.get("manual_upload"):
        runtime["manual_upload"]["batches_directory"] = "submissions"
    runtime.pop("phase_references_path", None)
    runtime.pop("agent_web_port", None)
    runtime["shared_dft"] = options(root, name)
    # New diagrams always initialize their own physical workspaces and UI state.
    for key in ("runtime_storage_override", "local_path_relocations"):
        runtime.pop(key, None)
    runtime["resume_existing_project"] = True
    write(directory / "agent_runtime.json", runtime)
    write(directory / "parameters/config_session.json", session)
    write(directory / "parameters/search_config.project.json", config)
    from phase_agent.tools.workflows.reference_dft_startup import prepare_choice

    state = {
        "confirmed_config": config,
        "confirmed_config_version": session["confirmed_snapshot"]["config_version"],
        "shared_dft": options(root, name),
        "tasks": [],
        "pending_tasks": [],
        "budget_usage": {"total_relative_cost": 0.0, "stages": {}},
        "reserved_relative_cost": 0.0,
    }
    state["reference_dft_startup"] = prepare_choice(state, config)
    write(directory / "workflow_state/state.json", state)
    from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager

    manager = PhaseDataManager(boundary, system_config=config["system"])
    manager.save(directory / "workflow_state/ledgers/phase_data.json")
    # No task, progress, approvals, committee weights, or fine-tuning state is copied.
    descriptor["diagrams"].append(name)
    write(root / "system.json", descriptor)
    (directory / "README.md").write_text(
        "# 独立相图 "
        + name
        + "\n\n只连接体系 shared_dft；MLIP、预算、状态、审批与提交目录归本相图独立管理。打开本目录 agent_runtime.json。\n",
        encoding="utf-8",
    )
    return directory


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["create", "list"])
    parser.add_argument("--system-root", required=True)
    parser.add_argument("--name")
    parser.add_argument("--boundary")
    args = parser.parse_args()
    if args.action == "list":
        print("\n".join(read(Path(args.system_root) / "system.json")["diagrams"]))
    elif not args.name or not args.boundary:
        parser.error("create requires --name and --boundary")
    else:
        print(create(args.system_root, args.name, read(args.boundary)))


if __name__ == "__main__":
    main()
