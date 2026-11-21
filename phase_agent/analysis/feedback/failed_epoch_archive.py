"""Immutable, checksum-verified failed-epoch copy without breaking old paths."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def _digest(path):
    result = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _inventory(root):
    import os
    from phase_agent.analysis.feedback.physical_layout import managed_alias

    root = Path(root).absolute()
    analysis = next((p for p in (root, *root.parents) if p.name == "analysis_outputs"), root)
    rows = []
    if (root.is_symlink() or root.is_junction()) and not managed_alias(analysis, root):
        raise ValueError("failed_epoch_archive_link_not_allowed")
    for directory, directories, files in os.walk(root, followlinks=True):
        for name in directories:
            path = Path(directory) / name
            if (path.is_symlink() or path.is_junction()) and not managed_alias(analysis, path):
                raise ValueError("failed_epoch_archive_link_not_allowed")
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or path.is_junction():
                raise ValueError("failed_epoch_archive_link_not_allowed")
            if not path.resolve().is_relative_to(analysis.resolve()):
                raise ValueError("failed_epoch_archive_path_outside_root")
            rows.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": _digest(path),
                }
            )
    return sorted(rows, key=lambda row: row["path"])


def archive_failed_epoch(state, state_path):
    """Publish one frozen local copy; source paths and scientific state remain intact."""
    from phase_agent.analysis.feedback.versioned_products import validate_bundle
    from phase_agent.evidence.model_epoch import model_epoch
    from phase_agent.tools.workflows.direct_extrapolation_sampling import valid_direct_receipt

    root = Path(state["analysis_output_directory"]).resolve()
    if root.name != "analysis_outputs" or root.parent != Path(state_path).resolve().parent.parent:
        raise ValueError("failed_epoch_archive_root_invalid")
    cycle = state.get("extrapolation_recovery_cycle") or {}
    scope = cycle.get("scope_id")
    receipt = (state.get("direct_sampling_results") or {}).get(scope) or {}
    if (
        cycle.get("status")
        not in {"awaiting_reviewed_dft_labels", "awaiting_new_model_refresh", "completed"}
        or receipt.get("source_digest") != cycle.get("direct_source_digest")
        or not valid_direct_receipt(receipt)
    ):
        raise ValueError("failed_epoch_archive_recovery_unverified")
    epoch = model_epoch(state, cycle.get("invalid_model_version"))
    if not epoch:
        raise ValueError("failed_epoch_archive_epoch_unknown")
    source = root / epoch
    target = root / (epoch + "-extrapolation_failed")
    from phase_agent.analysis.feedback.physical_layout import managed_alias

    if (
        not source.is_dir()
        or source.is_symlink()
        or source.is_junction()
        or ((target.is_symlink() or target.is_junction()) and not managed_alias(root, target))
    ):
        raise ValueError("failed_epoch_archive_source_invalid")
    latest = validate_bundle(root / "latest" / "phase_diagram")
    if (latest.get("identity") or {}).get("hull_version") == cycle.get("invalid_hull_version"):
        raise ValueError("failed_epoch_archive_invalid_hull_is_latest")
    if cycle.get("invalid_hull_version"):
        invalid = validate_bundle(source / "phase_diagram" / "latest")
        if (invalid.get("identity") or {}).get("hull_version") != cycle.get("invalid_hull_version"):
            raise ValueError("failed_epoch_archive_hull_mismatch")
    identity = {
        "epoch": epoch,
        "invalid_model_version": cycle.get("invalid_model_version"),
        "invalid_hull_version": cycle.get("invalid_hull_version"),
        "direct_source_digest": cycle.get("direct_source_digest"),
    }
    if target.exists():
        manifest = json.loads((target / "ARCHIVE_MANIFEST.json").read_text(encoding="utf8"))
        actual = [
            row
            for row in _inventory(target)
            if row["path"] not in {"ARCHIVE_MANIFEST.json", "EXTRAPOLATION_ARCHIVE.md"}
        ]
        if manifest.get("identity") != identity or manifest.get("files") != actual:
            raise ValueError("failed_epoch_archive_existing_conflict")
        return {
            "status": "already_archived",
            "path": str(target),
            "source_path": str(source),
            "manifest_sha256": _digest(target / "ARCHIVE_MANIFEST.json"),
            "file_count": len(actual),
            "size_bytes": sum(row["size"] for row in actual),
        }
    files = _inventory(source)
    if not files or {"ARCHIVE_MANIFEST.json", "EXTRAPOLATION_ARCHIVE.md"}.intersection(
        row["path"] for row in files
    ):
        raise ValueError("failed_epoch_archive_source_conflict")
    with TemporaryDirectory(prefix=".failed-archive-", dir=root) as temporary:
        stage = Path(temporary) / "archive"
        stage.mkdir()
        for row in files:
            origin, destination = source / row["path"], stage / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
            if _digest(destination) != row["sha256"] or _digest(origin) != row["sha256"]:
                raise ValueError("failed_epoch_archive_source_changed")
        if _inventory(source) != files or _inventory(stage) != files:
            raise ValueError("failed_epoch_archive_source_changed")
        (stage / "EXTRAPOLATION_ARCHIVE.md").write_text(
            f"# {epoch}: extrapolation failed\n\nThis is a frozen, checksum-verified copy of `{epoch}/`. "
            "The original directory remains available because existing scientific receipts use its paths. "
            "This epoch is invalid as the current phase diagram; see `ARCHIVE_MANIFEST.json` for every file.\n",
            encoding="utf8",
        )
        (stage / "ARCHIVE_MANIFEST.json").write_text(
            json.dumps(
                {"identity": identity, "source_path": str(source), "files": files},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf8",
        )
        stage.rename(target)
    return {
        "status": "archived",
        "path": str(target),
        "source_path": str(source),
        "manifest_sha256": _digest(target / "ARCHIVE_MANIFEST.json"),
        "file_count": len(files),
        "size_bytes": sum(row["size"] for row in files),
    }
