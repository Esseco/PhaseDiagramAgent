"""Inspect failed regional sampling staging without accepting or deleting it."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def inspect_abandoned_region_generation(state_path, state, identity):
    from phase_agent.policies.file_approval import proposal_hash
    from phase_agent.policies.region_sampling import is_region, ledger_fingerprint

    invocation = identity.get("invocation_id")
    rows = [r for r in state.get("action_records") or [] if r.get("record_id") == invocation]
    if len(rows) != 1:
        raise ValueError("region_staging_action_missing")
    record = rows[0]
    action = record.get("final_action") or {}
    params = action.get("parameters") or {}
    if not (identity.get("tool") == "generate_branches" and is_region(action)
            and identity.get("action_hash") == proposal_hash(action)
            and record.get("status") == "failed"
            and (record.get("human_feedback") or {}).get("decision") == "approve"
            and (record.get("execution_result") or {}).get("error") == "MemoryError: "
            and params.get("initial_states_per_branch") == 1):
        raise ValueError("region_staging_action_unverified")
    key = action.get("task_key")
    if not isinstance(key, str) or not key:
        raise ValueError("region_staging_task_key_invalid")
    root = Path(state_path).resolve().parent.parent
    ledger_path = root / "workflow_state" / "ledgers" / "phase_data.json"
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes.decode("utf-8-sig"))
    frozen = (params.get("region_preview") or {}).get("ledger_fingerprint")
    if not frozen or frozen != ledger_fingerprint(SimpleNamespace(data=ledger)):
        raise ValueError("region_staging_ledger_changed")
    if any(r.get("task_key") == key for r in state.get("generation_history") or []):
        raise ValueError("region_staging_already_registered")
    config = state.get("confirmed_config") or {}
    storage = (config.get("config") or config).get("storage") or {}
    base = Path((storage.get("paths") or {}).get("structures", "structures/candidate_structures"))
    base = (base if base.is_absolute() else root / base).resolve()
    if not base.is_relative_to(root):
        raise ValueError("region_staging_outside_project")
    raw = base / ("RegionSampling-" + hashlib.sha256(key.encode()).hexdigest()[:16])
    if not raw.exists():
        return None
    if raw.is_symlink() or not raw.is_dir() or not raw.resolve().is_relative_to(base):
        raise ValueError("region_staging_directory_invalid")
    files, branches, structures = [], set(), set()
    for folder in sorted(raw.iterdir()):
        if folder.is_symlink() or not folder.is_dir() or not folder.name.startswith("B-"):
            raise ValueError("region_staging_unexpected_path")
        entries = list(folder.iterdir())
        if any(f.is_symlink() or not f.is_file() for f in entries):
            raise ValueError("region_staging_unexpected_path")
        full = [f for f in entries if f.name == "full_na_structure.vasp"]
        selected = [f for f in entries if f.name.startswith("S-") and f.suffix == ".vasp"]
        if len(entries) != 2 or len(full) != 1 or len(selected) != 1:
            raise ValueError("region_staging_inventory_invalid")
        sid = selected[0].stem
        if folder.name in (ledger.get("branches") or {}) or sid in (ledger.get("structures") or {}):
            raise ValueError("region_staging_has_registered_structure")
        branches.add(folder.name); structures.add(sid)
        for f in sorted(entries):
            data = f.read_bytes()
            if not data:
                raise ValueError("region_staging_empty_file")
            files.append({"path": f.relative_to(raw).as_posix(), "size": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})
    if not branches or len(branches) != len(structures):
        raise ValueError("region_staging_inventory_invalid")
    if any((t.get("branch_id") in branches or t.get("structure_id") in structures)
           for t in state.get("tasks") or []):
        raise ValueError("region_staging_has_registered_tasks")
    if any(r.get("structure_id") in structures for r in state.get("phase_records") or []):
        raise ValueError("region_staging_has_phase_records")
    return {"kind": "abandoned_region_generation_staging", "directory": str(raw),
            "task_key": key, "branch_count": len(branches), "structure_count": len(structures),
            "file_count": len(files), "files": files,
            "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            "ledger_fingerprint": frozen, "submitted": False}
