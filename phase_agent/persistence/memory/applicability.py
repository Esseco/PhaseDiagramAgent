"""Explicit knowledge applicability, distinct from source provenance."""

KEYS = {"system_id", "phase", "action", "model_version", "project_id"}


def project_scope_matches(first, second):
    def project(row):
        return (
            row.get("project_id")
            or (row.get("applicability") or {}).get("project_id")
            or row.get("source_project")
        )

    return first.get("scope") != "project" or project(first) == project(second)


def memory_scope(state):
    config = state.get("confirmed_config") or {}
    project = state.get("current_project") or {}
    return {
        "project_id": state.get("project_id")
        or project.get("project_path")
        or project.get("project")
        or (config.get("storage") or {}).get("workspace_root"),
        "phase": state.get("decision_phase")
        or state.get("current_phase")
        or ((config.get("system") or {}).get("constraints") or {}).get("phases"),
    }


def applicability_errors(value):
    if not isinstance(value, dict):
        return ["applicability must be an object"]
    errors = []
    for key, item in value.items():
        if key not in KEYS or (
            item is not None and (not isinstance(item, str) or not item.strip())
        ):
            errors.append(f"invalid applicability: {key}")
    return errors
