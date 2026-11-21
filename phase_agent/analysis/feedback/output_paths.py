"""Stable analysis paths independent of the current working directory."""

from pathlib import Path

from phase_agent.evidence.model_epoch import model_epoch


def analysis_metadata_directory(root):
    """Keep project-wide indexes and locks outside the MLIP-only latest view."""
    root = Path(root)
    name = "analysis" if root.name == "analysis_outputs" else "analysis_" + root.name
    return root.parent / "workflow_state" / name


def analysis_epoch_directory(root, state, model_version):
    """Use a registered epoch or retain unassigned evidence in internal metadata."""
    root = Path(root)
    epoch = model_epoch(state, model_version)
    return root / epoch if epoch else analysis_metadata_directory(root) / "Unassigned"


def dft_round_directory(version_root, group_name, round_name):
    """Group new DFT products; keep an existing round at its historical path."""
    root = Path(version_root)
    legacy = root / group_name / round_name
    current = root / "dft" / group_name / round_name
    if legacy.exists() and current.exists() and legacy.resolve() != current.resolve():
        raise ValueError("duplicate DFT round destinations require explicit review")
    return legacy if legacy.parent.is_dir() else current


def epoch_product_directory(root, epoch, kind):
    """Keep publishing to an unmigrated epoch until the organizer moves it."""
    legacy = Path(root) / epoch / kind
    if kind != "dft_error":
        return legacy
    current = legacy.parent / "dft" / "summary"
    if legacy.exists():
        if current.exists():
            if legacy.resolve() != current.resolve():
                raise ValueError("conflicting DFT summary directories")
            return current
        return legacy
    return current
