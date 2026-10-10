"""Public directory names; transport paths are distinct from scientific IDs."""

DIRECTORY_RENAMES = {
    "config": "parameters",
    "runtime": "workflow_state",
    "memory": "agent_memory",
    "inputs": "structures",
    "InitFile": "structures/reference_structures",
    "upload_batches": "submissions",
    "outputs": "analysis_outputs",
    "backups": "history_backups",
    "docs": "documentation",
}
