"""Isolate generated output paths; shared immutable scientific inputs stay unchanged."""

from copy import deepcopy
from config_layer.session.resolve_workspace_paths import default_workspace_storage, resolve_workspace_paths


def isolate_runtime_outputs(runtime_config, run_directory):
    storage = default_workspace_storage(run_directory)
    storage["paths"].update(state="state.json", ledger="phase_data.json")
    paths = resolve_workspace_paths({"storage": storage}, base_directory=run_directory)
    config = deepcopy(runtime_config)
    mapping = {"state_path": "state", "ledger_path": "ledger",
        "structure_directory": "structures", "phase_diagram_directory": "phase_diagrams",
        "work_directory": "work", "branch_energy_pool_ledger_path": "branch_energy_pool_ledger",
        "approval_directory": "approvals", "local_action_directory": "approved_batches",
        "upload_batches_directory": "upload_batches"}
    config.update({key: str(paths[value]) for key, value in mapping.items()})
    config.setdefault("qbc", {})["output_path"] = str(paths["qbc_results"])
    return config, paths


def publish_run_descriptor(settings, session, paths, *, source_config):
    """Persist execution-path binding without changing the confirmed scientific snapshot."""
    from execution_layer.step_runner.file_protocol import write_json
    from config_layer.session.resolve_workspace_paths import DEFAULT_WORKSPACE_PATHS
    root = paths["workspace_root"]
    storage = {"workspace_root": str(root), "paths": {
        key: paths[key].relative_to(root).as_posix() for key in DEFAULT_WORKSPACE_PATHS}}
    descriptor = deepcopy(settings)
    from pathlib import Path
    for key in ("phase_references_path", "knowledge_library_root"):
        if descriptor.get(key):
            value = Path(descriptor[key])
            descriptor[key] = str(value if value.is_absolute() else
                                  (Path(source_config).parent / value).resolve())
    descriptor.update(state_path=str(paths["state"]), ledger_path=str(paths["ledger"]),
        config_session_path=str(root / "parameters/config_session.json"),
        runtime_storage_override=storage, source_runtime_config=str(source_config))
    local_session = deepcopy(session)
    local_session["editable_config_json_path"] = str(root / "parameters/search_config.project.json")
    write_json(root / "parameters/config_session.json", local_session)
    write_json(root / "agent_runtime.json", descriptor)
    return root / "agent_runtime.json"
