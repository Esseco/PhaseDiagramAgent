"""Materialize only already-approved DFT tasks for manual upload."""

from copy import deepcopy
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.science.dft.prepare_pycode_relax import prepare_pycode_relax


def prepare_dft_upload_batches(*, action, context):
    state = deepcopy(context.get("event_state") or {})
    config = context["effective_config"]
    root = config.get("upload_batches_directory")
    if not root:
        return {"status": "not_configured", "reason": "upload_directory_missing", "state": state}
    requested = set(action.get("target_ids") or [])
    tasks = [
        row
        for row in state.get("tasks") or []
        if row.get("stage") in {"dft_relax", "dft_single_point"}
        and row.get("status") == "pending"
        and not row.get("slurm_batch_id")
        and (not requested or row.get("task_id") in requested)
    ]

    from phase_agent.persistence.shared_dft import reusable_results
    from phase_agent.tools.state.reconcile_task_results import reconcile_task_results
    from phase_agent.tools.workflows.apply_scientific_feedback import apply_scientific_feedback

    reused = reusable_results(
        state, context["manager"], config, task_ids={t["task_id"] for t in tasks}
    )
    if reused:
        state = reconcile_task_results(state, reused)["state"]
        state = apply_scientific_feedback(
            state,
            reused,
            manager=context["manager"],
            ledger_path=config.get("ledger_path"),
            phase_diagram_directory=config.get("phase_diagram_directory"),
            phase_references=context.get("phase_references"),
        )["state"]
        reused_ids = {t["task_id"] for t in reused}
        tasks = [t for t in tasks if t["task_id"] not in reused_ids]

    def dispatch(task):
        from pathlib import Path
        import re
        from phase_agent.tools.workflows.comparison_model_registry import original_round_model

        model, digest, _ = original_round_model(state, config, task.get("model_version"))
        from phase_agent.configuration.schema.python_environments import remote_comparison_model

        model = remote_comparison_model(model, config)
        if digest:
            model["model_sha256"] = digest
        result = prepare_pycode_relax(task, manager=context["manager"], comparison_model=model)
        if task.get("reviewed_submit_script"):
            script = re.sub(
                r"(?m)^#SBATCH --job-name=.*$",
                "#SBATCH --job-name=DFT-" + task["task_id"],
                task["reviewed_submit_script"],
            )
            Path(task["work_directory"]).joinpath("submit_gpu.sh").write_text(
                script, encoding="utf-8", newline="\n"
            )
        return result

    runner = ManualUploadBatchRunner(root, worker_command=[], dispatcher=dispatch)
    # Restrict the existing runner to this approved preparation request.
    select = runner._select
    ids = {row["task_id"] for row in tasks}
    runner._select = lambda current, rows: select(
        current, [row for row in rows if row.get("task_id") in ids]
    )
    batches = []
    while True:
        prepared = runner.prepare(state)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batches.append(prepared["batch"])
    from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs

    delivery = finalize_prepared_handoffs(state)
    if delivery.get("status") == "not_configured":
        return delivery
    return {
        "status": "prepared" if batches else "already_prepared",
        **delivery,
        "state": state,
        "batches": batches,
        "batch_count": len(batches),
        "reused_dft_count": len(reused),
        "submitted": False,
    }
