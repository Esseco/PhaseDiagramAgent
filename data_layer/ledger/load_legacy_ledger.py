"""Load an existing schema-v2 ledger without rewriting or copying user data."""

from data_layer.ledger.phase_data_manager import PhaseDataManager


def load_legacy_ledger(path):
    return PhaseDataManager.load(path)
