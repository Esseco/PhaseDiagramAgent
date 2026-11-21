"""Migrate a user-approved NELM-only config change without altering old tasks."""
from copy import deepcopy


def migrate_state(state, snapshot, expected_old, expected_new, value):
    from phase_agent.configuration.runtime.authorize_budget_extension import _has_active_work

    if type(value) is not int or value < 1:
        raise ValueError("invalid_nelm")
    if (state.get("confirmed_config_version") != expected_old
        or snapshot.get("config_version") != expected_new or expected_old == expected_new):
        raise ValueError("nelm_config_version_mismatch")
    if _has_active_work(state) or state.get("pending_execution_policies"):
        raise ValueError("nelm_config_active_work")
    expected = deepcopy(state.get("confirmed_config") or {})
    expected.setdefault("dft", {}).setdefault("parameters", {}).setdefault("incar_settings", {}).setdefault("static", {})["NELM"] = value
    if expected != snapshot.get("config"):
        raise ValueError("nelm_config_other_changes")
    updated = deepcopy(state)
    updated["confirmed_config"] = deepcopy(snapshot["config"])
    updated["confirmed_config_version"] = expected_new
    updated.setdefault("config_migrations", []).append({"type": "user_approved_dft_nelm",
        "from": expected_old, "to": expected_new, "nelm": value})
    return updated


def migrate(control, expected_old, expected_new, value):
    import json
    from pathlib import Path
    from filelock import FileLock
    from phase_agent.persistence.result_recovery_commit import commit_result_recovery
    from phase_agent.tools.remote.transfer_bundle import checksum
    from phase_agent.runtime.project_context import current_config_session

    path = Path(control.state_path).resolve()
    with control.chat_handler.lock, FileLock(str(path.parent / "langgraph_lifecycle.sqlite") + ".lock", timeout=30):
        before = checksum(path)
        state = json.loads(path.read_text(encoding="utf8"))
        session = current_config_session(control.chat_handler)
        if session.get("status") != "confirmed":
            raise ValueError("nelm_config_not_confirmed")
        updated = migrate_state(state, session.get("confirmed_snapshot") or {}, expected_old, expected_new, value)
        commit_result_recovery(path, updated, {}, expected_hashes={str(path): before},
            evidence={"operation": "user_approved_dft_nelm", "from": expected_old, "to": expected_new, "nelm": value},
            config_transition={"from": expected_old, "to": expected_new})
        control._codex_context_cache = None
        return {"status": "migrated", "from": expected_old, "to": expected_new, "nelm": value, "submitted": False}
