"""Default, versioned supercomputer orchestration settings."""


def default_supercomputer_config() -> dict:
    """Return site-neutral settings; scheduler commands remain unconfigured."""
    return {
        "path_mappings": [],
        "batch_sizes": {
            "simple_check": 100,
            "relax_and_feature": 100,
            "deep_search": 10,
            "dft_single_point": 1,
            "dft_relax": 1,
        },
        "paths": {
            "state": "outputs/search_state.json",
            "summary": "outputs/exchange/status_summary.json",
            "plans": "outputs/exchange/plans",
            "batches": "outputs/jobs",
        },
        "scheduler": {
            "submit_command": None,
            "query_command": None,
            "cancel_command": None,
        },
        "worker": {
            "command": [
                "python3", "-m", "execution_layer.slurm.run_slurm_array_task",
                "--executor", "scientific_layer.mlip.slurm_executor:execute_mlip_task",
            ],
            "action_executor": None,
        },
        "node_roles": {
            "recover": "compute",
            "advise": "login",
            "confirm": "login",
            "prepare": "compute",
            "submit": "login",
            "status": "login_or_compute",
        },
    }
