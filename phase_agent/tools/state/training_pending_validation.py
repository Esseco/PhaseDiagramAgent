"""A returned candidate must be reviewed before training another candidate."""


def training_pending_validation(state):
    return any(
        job.get("status") == "results_received"
        and not job.get("activated")
        and job.get("original_model_version") == state.get("active_model_version")
        for job in (state.get("remote_finetune_jobs") or {}).values()
    )
