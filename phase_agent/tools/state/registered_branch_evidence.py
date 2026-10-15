"""Verify branch-generation effects against a bound inventory and the actual ledger."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path


def inspect_registered_branches(state, action, state_path):
    try:
        return _inspect(state, action, state_path)
    except (ValueError, TypeError, KeyError, OSError) as error:
        return {"status": "invalid", "errors": [type(error).__name__ + ": " + str(error)]}


def _inspect(state, action, state_path):
    if (action.get("tool") or action.get("action_type")) != "generate_branches":
        return None
    from phase_agent.tools.workflows.generation_artifacts import artifact_path
    from phase_agent.tools.workflows.branch_review_view import raw_record

    root = Path(state_path).resolve().parent.parent
    path = artifact_path(state, action.get("task_key"), action.get("parameters") or {})
    from phase_agent.tools.policy.region_sampling import is_region

    if is_region(action):
        from phase_agent.tools.workflows.region_review import path_for

        path = path_for(state, action)
    if path is None or not path.is_file():
        return None
    output = {"status": "invalid", "artifact_path": str(path), "errors": []}
    problems = output["errors"]
    if not path.resolve().is_relative_to(root / "workflow_state/branch_plans"):
        problems.append("inventory_outside_project")
        return output
    raw = raw_record(json.loads(path.read_text(encoding="utf-8")))
    for key, expected in (
        ("task_key", action.get("task_key")),
        ("model_version", state.get("active_model_version")),
        ("config_version", state.get("confirmed_config_version")),
    ):
        if raw.get(key) != expected:
            problems.append("inventory_identity_mismatch:" + key)
    if raw.get("status") != "completed":
        problems.append("inventory_not_completed")
    ledger_path = Path(state_path).resolve().parent / "ledgers/phase_data.json"
    if not ledger_path.is_file():
        problems.append("missing_structure_ledger")
        return output
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    actual = raw.get("actual") or {}
    rows = actual.get("branches") or []
    if not rows:
        problems.append("empty_registered_inventory")
    branch_ids, structure_ids, new_ids, files = set(), set(), set(), []
    for row in rows:
        bid = row.get("branch_id")
        if bid in branch_ids:
            problems.append("duplicate_branch_id")
        branch_ids.add(bid)
        if row.get("is_new"):
            new_ids.add(bid)
        branch = (ledger.get("branches") or {}).get(bid)
        if not branch:
            problems.append("branch_not_registered:" + str(bid))
            continue
        if (
            row.get("phase") != branch.get("P")
            or row.get("H") != branch.get("H")
            or Fraction(str(row.get("Na_fraction"))) != Fraction(str(branch.get("x")))
        ):
            problems.append("branch_inventory_mismatch:" + str(bid))
        structures = row.get("structures") or []
        if row.get("structure_count") != len(structures):
            problems.append("branch_structure_count_mismatch")
        for item in structures:
            sid = item.get("structure_id")
            if sid in structure_ids:
                problems.append("duplicate_structure_id")
            structure_ids.add(sid)
            registered = (ledger.get("structures") or {}).get(sid)
            if (
                not registered
                or registered.get("branch_id") != bid
                or sid not in branch.get("structure_ids", [])
            ):
                problems.append("structure_not_registered_to_branch:" + str(sid))
                continue
            source = Path(registered.get("source_path") or "")
            if (
                not source.is_absolute()
                or not source.resolve().is_relative_to(root)
                or not source.is_file()
            ):
                problems.append("missing_or_unowned_source:" + str(sid))
                continue
            if item.get("path") != str(source):
                problems.append("structure_source_mismatch:" + str(sid))
                continue
            files.append(
                {
                    "structure_id": sid,
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
    for name, count in (
        ("new_branch_count", len(new_ids)),
        ("registered_branch_count", len(branch_ids)),
        ("registered_structure_count", len(structure_ids)),
    ):
        if type(actual.get(name)) is not int or actual[name] != count:
            problems.append("registered_count_mismatch:" + name)
    output.update(
        status="verified" if not problems else "invalid",
        task_key=action.get("task_key"),
        new_branch_ids=sorted(new_ids),
        branch_ids=sorted(branch_ids),
        structure_ids=sorted(structure_ids),
        registered_branch_count=len(branch_ids),
        registered_structure_count=len(structure_ids),
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        source_files=files,
        scope="registered_branch_outputs_only; downstream input preparation not assumed completed",
    )
    return output
