"""Safe, opt-in project storage compaction reusing archive_history and settled-batch pruning."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any


def _collect_referenced_paths(
    state: dict[str, Any], ledger: dict[str, Any] | None = None
) -> set[Path]:
    """Collect all filesystem paths referenced by active state or structure ledgers."""
    refs: set[Path] = set()

    def _scan(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                if (
                    isinstance(val, str)
                    and val
                    and (key.endswith("_path") or key in {"path", "source_path"})
                ):
                    refs.add(Path(val).resolve())
                elif isinstance(val, (dict, list)):
                    _scan(val)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _scan(item)

    for key in (
        "tasks",
        "phase_records",
        "dft_dataset_records",
        "dft_mlip_comparisons",
        "phase_diagrams",
    ):
        _scan(state.get(key))
    if ledger:
        _scan(ledger.get("structures"))
    return refs


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def compact_project_storage(
    state_path: str | Path,
    *,
    ledger_path: str | Path | None = None,
    dry_run: bool = True,
    archive_state: bool = True,
    keep_recent: int = 50,
) -> dict[str, Any]:
    """Compact settled state history and prune duplicate extracted batch folders.

    Must be run offline (while project service is stopped). Never deletes active batches,
    results archives, or any path referenced by tasks/phase_records/structures.
    """
    state_file = Path(state_path).resolve()
    if not state_file.is_file():
        raise FileNotFoundError(f"state_not_found: {state_file}")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    ledger = (
        json.loads(Path(ledger_path).read_text(encoding="utf-8"))
        if ledger_path and Path(ledger_path).is_file()
        else None
    )

    archive_report = None
    if archive_state and not dry_run:
        from phase_agent.persistence.state.archive_history import archive_state_file

        archive_report = archive_state_file(
            state_file,
            keep_recent=keep_recent,
            write_archive=True,
        )
        state = json.loads(state_file.read_text(encoding="utf-8"))

    protected = _collect_referenced_paths(state, ledger)
    candidates: list[dict[str, Any]] = []
    freed_bytes = 0

    for bundle_id, batch in sorted((state.get("relax_mc_batches") or {}).items()):
        if not isinstance(batch, dict):
            continue
        if batch.get("status") != "results_received" or not batch.get("import_checksum"):
            continue
        for key in ("upload_folder", "extracted_results_dir"):
            raw = batch.get(key)
            if not raw:
                continue
            target = Path(raw).resolve()
            if not target.is_dir():
                continue
            # Never remove if any referenced file lives inside this folder.
            if any(ref == target or target in ref.parents for ref in protected):
                continue
            size = _dir_size(target)
            candidates.append(
                {"bundle_id": bundle_id, "kind": key, "path": str(target), "bytes": size}
            )
            if not dry_run:
                shutil.rmtree(target)
            freed_bytes += size

    return {
        "status": "dry_run" if dry_run else "compacted",
        "dry_run": dry_run,
        "archive_report": archive_report,
        "pruned_folders": candidates,
        "reclaimable_bytes": freed_bytes,
    }
