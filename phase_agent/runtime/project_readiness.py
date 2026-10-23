"""Read-only project continuation diagnostics using actual runtime path mappings."""

from copy import deepcopy
from pathlib import Path


def project_readiness(state, config, settings, *, workspace):
    from phase_agent.configuration.session.relocate_workspace_paths import relocate_workspace_paths
    from phase_agent.configuration.schema.validate_search_config import validate_search_config
    from phase_agent.analysis.convergence.training_provenance import model_with_training_provenance
    from phase_agent.tools.policy.workflow_policy import active_tasks

    effective = deepcopy(config)
    if settings.get("local_path_relocations"):
        effective = relocate_workspace_paths(
            effective, settings["local_path_relocations"], workspace
        )
    references = (effective.get("system") or {}).get("phase_references") or {}
    paths = {}
    for phase, value in references.items():
        value = value.get("path") if isinstance(value, dict) else value
        if not isinstance(value, str):
            paths[phase] = None
            continue
        path = Path(value)
        paths[phase] = path if path.is_absolute() else Path(workspace) / path
    active = active_tasks(state)
    model = state.get("active_model") or effective.get("mlip") or {}
    derived = model_with_training_provenance(state, model)
    trained = bool(model.get("dataset_version") or state.get("last_trained_dataset_version"))
    known = not trained or (
        derived.get("training_provenance_complete") is True
        and "training_data_ids" in derived
        and "training_structure_hashes" in derived
    )
    return {
        "configuration": validate_search_config(effective, stage="startup"),
        "references": {
            "missing_phases": [p for p, v in paths.items() if v is None or not v.is_file()],
            "resolved_paths": {p: str(v) if v else None for p, v in paths.items()},
        },
        "training_barrier_task_ids": [t.get("task_id") for t in active],
        "cancelled_dft_task_ids": [
            t.get("task_id")
            for t in state.get("tasks") or []
            if t.get("recovery_cancelled") is True
            and t.get("stage") in {"dft_single_point", "dft_relax"}
        ],
        "cancelled_mc_task_ids": [
            t.get("task_id")
            for t in state.get("tasks") or []
            if t.get("recovery_cancelled") is True and t.get("stage") == "deep_search"
        ],
        "training_provenance": {
            "known": known,
            "source": derived.get("training_provenance_source"),
            "training_identity_count": len(derived.get("training_data_ids") or []),
        },
        "pending_proposal_count": len(state.get("pending_execution_policies") or {}),
        "services_started": False,
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, help="Existing agent_runtime.json; read only")
    args = parser.parse_args()
    runtime_path = Path(args.runtime).resolve()
    settings = json.loads(runtime_path.read_text(encoding="utf-8-sig"))

    def load(name):
        path = Path(settings[name])
        if not path.is_absolute():
            path = runtime_path.parent / path
        return json.loads(path.read_text(encoding="utf-8-sig"))

    state = load("state_path")
    session = load("config_session_path")
    config = (
        (session.get("confirmed_snapshot") or {}).get("config")
        or state.get("confirmed_config")
        or {}
    )
    report = project_readiness(state, config, settings, workspace=runtime_path.parent)
    report["configuration_version_matches"] = state.get("confirmed_config_version") == (
        session.get("confirmed_snapshot") or {}
    ).get("config_version")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
