"""Load the existing manager ledger without changing it."""

from pathlib import Path

from data_layer.ledger.phase_data_manager import PhaseDataManager


def load_branch_history(source):
    if isinstance(source, PhaseDataManager):
        return source
    if isinstance(source, (str, Path)):
        return PhaseDataManager.load(source)
    if hasattr(source, "data"):
        return source
    raise TypeError("source 必须是台账路径或 PhaseDataManager")
