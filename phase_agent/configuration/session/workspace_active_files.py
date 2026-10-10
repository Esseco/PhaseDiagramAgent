"""Locate live metadata for new and legacy local workspace maintenance tools."""

from pathlib import Path


def workspace_active_files(workspace):
    root = Path(workspace)
    if (root / "workflow_state/state.json").is_file():
        return {
            "state": root / "workflow_state/state.json",
            "ledger": root / "workflow_state/ledgers/phase_data.json",
            "branch_pool": root / "workflow_state/ledgers/branch_energy_pools.json",
            "phase_cache": root / "workflow_state/cache/phase_identification_cache.json",
        }
    if (root / "runtime/state.json").is_file():
        return {
            "state": root / "runtime/state.json",
            "ledger": root / "runtime/ledgers/phase_data.json",
            "branch_pool": root / "runtime/ledgers/branch_energy_pools.json",
            "phase_cache": root / "runtime/cache/phase_identification_cache.json",
        }
    return {
        "state": root / "current/state.json",
        "ledger": root / "current/phase_data.json",
        "branch_pool": root / "current/branch_energy_pools.json",
        "phase_cache": root / "current/phase_identification_cache.json",
    }
