"""Preview MC on only the original branches using the formally imported hull."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import zipfile
from phase_agent.persistence.ledger.phase_data_manager import PhaseDataManager
from phase_agent.tools.relax_mc_batch.analysis import next_mc


def preview(state, ledger_data, bundle_id):
    record = (state.get("relax_mc_batches") or {}).get(bundle_id) or {}
    transition = record.get("threshold_transition") or {}
    if (
        transition.get("status") != "relax_phase_imported"
        or record.get("status") != "partial_relax_imported"
    ):
        raise ValueError("transition_relax_import_required")
    diagram = (state.get("phase_diagrams") or {}).get("mlip") or {}
    if diagram.get("status") != "completed" or diagram.get("model_version") != record.get(
        "model_version"
    ):
        raise ValueError("transition_current_hull_required")
    if diagram.get("version") != transition.get("phase_diagram_version"):
        raise ValueError("transition_hull_changed_requires_review")
    branches = record.get("branch_ids") or []
    if (
        not branches
        or len(set(branches)) != len(branches)
        or not set(branches) <= set(ledger_data["branches"])
    ):
        raise ValueError("transition_branch_scope_invalid")
    with zipfile.ZipFile(record["upload_archive"]) as z:
        manifest = json.loads(z.read("batch.json"))
    if any(
        manifest.get(k) != record.get(k)
        for k in ("bundle_id", "plan_checksum", "model_version", "config_version")
    ):
        raise ValueError("transition_frozen_batch_changed")
    manager = PhaseDataManager(ledger_data["boundary"], ledger_data.get("system_config"))
    manager.data = deepcopy(ledger_data)
    manager.data["branches"] = {key: manager.data["branches"][key] for key in branches}
    with tempfile.TemporaryDirectory(prefix="phase-mc-preview-") as temp:
        plan = next_mc(state, manager, manifest, Path(temp), 0)
    actions = plan.get("actions") or []
    return {
        "status": plan["status"],
        "reason": plan.get("reason"),
        "bundle_id": bundle_id,
        "hull_version": diagram["version"],
        "review_digest": transition["review_digest"],
        "mc_action_count": len(actions),
        "actions": actions,
        "missing_branch_ids": plan.get("missing_branch_ids", []),
        "excluded_candidates": plan.get("excluded_candidates", []),
        "model_version": record["model_version"],
        "config_version": record["config_version"],
        "submitted": False,
        "project_tasks_created": False,
    }
