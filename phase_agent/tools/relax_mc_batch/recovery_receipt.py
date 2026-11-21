"""Explicit, integrity-bound selection of a collect-only return for an existing batch."""

import json
from pathlib import Path
import zipfile

from phase_agent.tools.remote.transfer_bundle import checksum, validate_bundle
from phase_agent.tools.relax_mc_batch.files import read, write


def selected_recovery(record):
    directory = record.get("directory")
    if not directory:
        return None
    pointer = Path(directory) / "recovery_return.json"
    if not pointer.is_file():
        return None
    receipt = read(pointer)
    root = Path(directory).with_name(Path(directory).name + "_recovery").resolve()
    archive = _resolve_recovery_archive(receipt, root)
    if checksum(archive) != receipt.get("archive_checksum"):
        raise ValueError("recovery_receipt_checksum_mismatch")
    return archive


def validate_recovery_scope(archive, record, state, *, receipt=None):
    """Configuration drift is allowed only for tasks already bound to the original batch."""
    validate_bundle(archive, extra_result_suffixes=(".csv", ".bin"))
    with zipfile.ZipFile(archive) as zipped:
        summary = json.loads(zipped.read("result_summary.json"))
        returned = json.loads(zipped.read("workflow_state.json"))
    recovery = summary.get("recovery") or {}
    if recovery.get("kind") != "collect_only" or recovery.get("submitted") is not False:
        raise ValueError("historical_import_requires_collect_only_recovery")
    for key in ("bundle_id", "plan_checksum", "model_version", "config_version"):
        if summary.get(key) != record.get(key):
            raise ValueError("recovery_original_batch_binding_mismatch:" + key)
    original = (
        Path(record["directory"]).with_name(Path(record["directory"]).name + "_output")
        / "results.zip"
    )
    pointer = (
        receipt if receipt is not None else read(Path(record["directory"]) / "recovery_return.json")
    )
    if checksum(original) != pointer.get("original_archive_checksum"):
        raise ValueError("recovery_original_archive_changed")
    validate_bundle(original, extra_result_suffixes=(".csv", ".bin"))
    with zipfile.ZipFile(original) as zipped:
        frozen = json.loads(zipped.read("workflow_state.json"))["tasks"]
        original_summary = json.loads(zipped.read("result_summary.json"))
    for key in ("bundle_id", "plan_checksum", "model_version", "config_version"):
        if original_summary.get(key) != record.get(key):
            raise ValueError("recovery_original_archive_binding_mismatch:" + key)
    known = {t["task_id"]: t for t in state.get("tasks") or []}
    expected = {t["task_id"]: t for t in frozen}
    tasks = returned.get("tasks") or []
    if len(tasks) != len(expected) or {t.get("task_id") for t in tasks} != expected.keys():
        raise ValueError("recovery_original_task_scope_mismatch")
    keys = (
        "task_id",
        "task_key",
        "structure_id",
        "branch_id",
        "stage",
        "batch_id",
        "task_checksum",
        "model_version",
        "config_version",
    )
    for task in tasks:
        tid = task["task_id"]
        if tid not in known or known[tid].get("recovery_cancelled") is True:
            raise ValueError("recovery_requires_registered_active_task:" + tid)
        for key in keys:
            if task.get(key) != expected[tid].get(key) or task.get(key) != known[tid].get(key):
                raise ValueError("recovery_registered_task_binding_mismatch:" + key)
    return summary


def register_recovery(record, state, archive):
    """Publish only a pointer; preserve the original return and all scientific state."""
    if record.get("status") in {"cancelled", "superseded"}:
        raise ValueError("cancelled_relax_mc_bundle_requires_manual_result_review")
    if state.get("active_model_version") != record.get("model_version"):
        raise ValueError("recovery_active_model_mismatch")
    root = (
        Path(record["directory"]).with_name(Path(record["directory"]).name + "_recovery").resolve()
    )
    archive = _resolve_recovery_archive({"archive": str(archive)}, root)
    original = (
        Path(record["directory"]).with_name(Path(record["directory"]).name + "_output")
        / "results.zip"
    )
    pointer = Path(record["directory"]) / "recovery_return.json"
    receipt = {
        "archive": str(archive),
        "archive_checksum": checksum(archive),
        "original_archive_checksum": checksum(original),
        "submitted": False,
    }
    validate_recovery_scope(archive, record, state, receipt=receipt)
    write(pointer, receipt)
    return receipt


def _resolve_recovery_archive(receipt, root):
    """Resolve a receipt path after a project delivery-directory migration.

    The archive must remain inside the recovery root and match the receipt
    checksum. A legacy absolute path is accepted only when exactly one file
    under the current root has that checksum.
    """
    archive = Path(receipt.get("archive", "")).resolve()
    if archive.is_relative_to(root) and archive.is_file():
        return archive
    expected = receipt.get("archive_checksum")
    if not expected:
        raise ValueError("recovery_receipt_archive_outside_batch")
    roots = [root]
    # Projects migrated their delivery tree under submissions/<epoch>/delivery.
    # Keep the legacy root for integrity checks, then search only the matching
    # batch name in the sibling delivery tree.
    legacy_parent = root.parent
    if legacy_parent.name == "relax_mc":
        delivery_parent = legacy_parent.parent / "delivery" / "relax_mc"
        migrated = delivery_parent / root.name
        if migrated.is_dir():
            roots.append(migrated)
    matches = [
        path
        for candidate_root in roots
        for path in candidate_root.rglob("*.zip")
        if path.is_file() and checksum(path) == expected
    ]
    if not matches:
        raise ValueError("recovery_receipt_archive_outside_batch")
    # Multiple identical copies are safe; choose a deterministic current copy.
    return sorted(matches, key=lambda path: str(path).lower())[0].resolve()
