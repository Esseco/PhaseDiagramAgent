"""Physically group analysis folders while keeping immutable old paths readable."""

import ctypes
import json
import os
from pathlib import Path
import re

from phase_agent.analysis.feedback.output_paths import analysis_metadata_directory


def destination(root, source):
    """Fixed directory rules only; never move an unknown scientific product."""
    root, source = Path(root).absolute(), Path(source).absolute()
    metadata = analysis_metadata_directory(root).absolute()
    if source.parent == metadata and source.name in {
        "Historical_Reports",
        "Previous_Latest",
        "Unassigned",
    }:
        return root / "history" / "project" / source.name
    parts = source.relative_to(root).parts
    if len(parts) == 1 and re.fullmatch(r"epoch\d+-extrapolation_failed", parts[0]):
        return root / "history" / "failed_epochs" / parts[0]
    if len(parts) == 2 and parts[0] == "out_data" and parts[1] in {"versions", "latest"}:
        return root / "history" / "out_data" / parts[1]
    if len(parts) == 2 and parts[0] == "latest" and parts[1] in {"out_data", "Historical_Reports", "training_parity"}:
        return root / "history" / "compatibility" / "latest" / parts[1]
    if not parts or not re.fullmatch(r"epoch\d+", parts[0]):
        return None
    if len(parts) == 2:
        if parts[1] == "dft_error":
            return root / parts[0] / "dft" / "summary"
        if parts[1].startswith("Search-group-"):
            return root / parts[0] / "dft" / parts[1]
        if parts[1] in {"direct_sampling", "extrapolation_review", "recovery"}:
            return root / parts[0] / "diagnostics" / parts[1]
        if parts[1] in {"phase_diagrams", "DFT_Selection", "Historical_Reports", "training"}:
            return root / "history" / parts[0] / parts[1]
    if (
        len(parts) == 3
        and parts[1] in {"phase_diagram", "dft_error"}
        and parts[2] in {"versions", "_internal"}
    ):
        return root / "history" / Path(*parts)
    if len(parts) == 4 and parts[1:3] == ("dft", "summary") and parts[3] in {
        "versions", "_internal"
    }:
        return root / "history" / parts[0] / "dft_error" / parts[3]
    return None


def managed_alias(root, source):
    source = Path(source)
    if not (source.is_junction() or source.is_symlink()):
        return False
    expected = destination(root, source)
    if expected is None or not expected.is_dir():
        return False
    if expected.is_junction() or expected.is_symlink():
        return False
    return source.resolve() == expected.absolute() and expected.resolve().is_relative_to(
        Path(root).resolve()
    )


def _inventory(root, *, dft_layout_root=None):
    from phase_agent.tools.local.organize_epoch_folders import file_hash

    result = {}
    for directory, folders, files in os.walk(root):
        for name in sorted(folders + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or path.is_junction():
                if (
                    dft_layout_root is None
                    or relative not in {"versions", "_internal"}
                    or not managed_alias(dft_layout_root, path)
                ):
                    raise ValueError("nested_analysis_link_not_supported:" + str(path))
                # Historical versions stay in place. Verify the same alias after rename.
                result[relative] = ("alias", str(path.resolve()))
                if name in folders:
                    folders.remove(name)
            elif path.is_file():
                result[relative] = file_hash(path)
    return result


def _alias(target, source):
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(source))
        kernel = ctypes.windll.kernel32
        attributes = kernel.GetFileAttributesW(str(source))
        if attributes == 0xFFFFFFFF or not kernel.SetFileAttributesW(str(source), attributes | 2):
            raise ctypes.WinError()
    else:
        source.symlink_to(target, target_is_directory=True)


def organize_physical_layout(root, *, dft_only=False):
    """Called under the existing Agent lock; no state/checkpoint rewrites or copies."""
    from phase_agent.analysis.feedback.export_dft_products import _publish_bytes

    root = Path(root).absolute()
    if root.is_symlink() or root.is_junction():
        raise ValueError("analysis_root_must_not_be_link")
    candidates = []
    for child in sorted(root.iterdir()):
        if re.fullmatch(r"epoch\d+", child.name) and child.is_dir():
            if child.is_symlink() or child.is_junction():
                raise ValueError("epoch_root_must_not_be_link")
            if dft_only:
                candidates.append(child / "dft_error")
            else:
                candidates.extend(child.iterdir())
            parents = [child / "dft" / "summary"]
            if not dft_only:
                parents.append(child / "phase_diagram")
            for parent in parents:
                if parent.is_dir():
                    if parent.is_symlink() or parent.is_junction():
                        raise ValueError("analysis_product_root_must_not_be_link")
                    candidates.extend(parent.iterdir())
        elif dft_only:
            continue
        elif child.name in {"latest", "out_data"} and child.is_dir():
            if child.is_symlink() or child.is_junction():
                raise ValueError("analysis_current_root_must_not_be_link")
            candidates.extend(child.iterdir())
        elif re.fullmatch(r"epoch\d+-extrapolation_failed", child.name):
            candidates.append(child)
    metadata = analysis_metadata_directory(root)
    candidates.extend(
        metadata / name
        for name in ("Historical_Reports", "Previous_Latest", "Unassigned")
        if not dft_only and (metadata / name).exists()
    )
    plans, aliases = [], []
    for source in candidates:
        target = destination(root, source)
        if target is None or (dft_only and not (source.exists() or source.is_symlink() or source.is_junction())):
            continue
        if source.is_symlink() or source.is_junction():
            if not managed_alias(root, source):
                raise ValueError("unrecognized_analysis_alias:" + str(source))
            aliases.append({"source": str(source), "target": str(target)})
            continue
        if not source.is_dir():
            raise ValueError("analysis_directory_expected:" + str(source))
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("analysis_target_outside_root")
        if any(
            p.is_symlink() or p.is_junction() for p in [target, *target.parents] if p != root.parent
        ):
            raise ValueError("analysis_target_contains_link")
        if target.exists():
            raise FileExistsError("analysis_destination_conflict:" + str(target))
        layout = root if source.name == "dft_error" and source.parent.parent == root else None
        plans.append((source, target, _inventory(source, dft_layout_root=layout), layout))
    # Complete all preflight checks before the first rename.
    completed = []
    try:
        for source, target, inventory, layout in plans:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            completed.append((source, target))
            if _inventory(target, dft_layout_root=layout) != inventory:
                raise ValueError("analysis_migration_bytes_changed")
            _alias(target, source)
            if not managed_alias(root, source):
                raise ValueError("analysis_alias_verification_failed")
        rows = aliases + [{"source": str(a), "target": str(b)} for a, b, _, _ in plans]
        result = {
            "status": "completed",
            "moved_directories": len(plans),
            "verified_files": sum(sum(isinstance(value, str) for value in files.values()) for _, _, files, _ in plans),
            "compatibility_aliases": sorted(rows, key=lambda row: row["source"]),
        }
        report_name = "dft_layout.json" if dft_only else "physical_layout.json"
        report = analysis_metadata_directory(root) / report_name
        _publish_bytes(
            report, (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf8")
        )
        return {**result, "report": str(report)}
    except BaseException:
        for source, target in reversed(completed):
            if source.is_junction():
                source.rmdir()
            elif source.is_symlink():
                source.unlink()
            elif source.is_dir() and not any(source.iterdir()):
                # Windows can leave the empty junction placeholder on failure.
                source.rmdir()
            target.rename(source)
        raise
