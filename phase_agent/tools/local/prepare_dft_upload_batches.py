"""Materialize only already-approved DFT tasks for manual upload."""

from copy import deepcopy
from time import perf_counter
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.science.dft.prepare_pycode_relax import prepare_pycode_relax


def prepare_dft_upload_batches(*, action, context, _owned=False):
    state = context.get("event_state") if _owned else deepcopy(context.get("event_state") or {})
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

    generation_samples = []

    def dispatch(task):
        from pathlib import Path
        import re
        from phase_agent.tools.workflows.comparison_model_registry import original_round_model

        model, digest, _ = original_round_model(state, config, task.get("model_version"))
        from phase_agent.configuration.schema.python_environments import remote_comparison_model

        model = remote_comparison_model(model, config)
        if digest:
            model["model_sha256"] = digest
        started, succeeded = perf_counter(), False
        try:
            result = prepare_pycode_relax(task, manager=context["manager"], comparison_model=model)
            succeeded = True
        finally:
            generation_samples.append((perf_counter() - started, succeeded))
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
    from phase_agent.runtime.turn_process import timed_call, record_timing, process_event

    process_event("复用既定DFT输入计划", {"task_count": len(tasks), "scope": "input_files_only"})
    try:
        state, batches = timed_call(
            "生成DFT输入并封装任务", _prepare_batches, runner, state, category="dft_inputs"
        )
    finally:
        total = sum(seconds for seconds, _ in generation_samples)
        failed = any(not ok for _, ok in generation_samples)
        record_timing(
            "DFT工作流生成（不含封装打包）",
            total,
            category="dft_inputs",
            status="failed" if failed else "completed",
        )
        process_event(
            "DFT输入生成统计",
            {
                "attempted": len(generation_samples),
                "completed": sum(ok for _, ok in generation_samples),
                "generation_seconds": round(total, 3),
                "slowest_point_seconds": round(
                    max((seconds for seconds, _ in generation_samples), default=0), 3
                ),
            },
        )
    from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs

    if batches:
        from phase_agent.tools.remote.submission_layout import write_submission_index

        timed_call("更新DFT输入索引", write_submission_index, root, state, category="dft_inputs")
    delivery = timed_call(
        "生成整批DFT上传包并校验", finalize_prepared_handoffs, state, category="dft_inputs"
    )
    if delivery.get("status") == "not_configured":
        return delivery
    prepared_ids = {batch["batch_id"] for batch in batches}
    handoffs = {
        batch_id: item["upload_archive"]
        for item in delivery.get("handoffs") or []
        for batch_id in item.get("batch_ids") or []
        if batch_id in prepared_ids
    }
    for batch in [*batches, *(state.get("slurm_batches") or [])]:
        if batch.get("batch_id") in handoffs:
            batch["upload_archive"] = handoffs[batch["batch_id"]]
    return {
        "status": "prepared" if batches else "already_prepared",
        **delivery,
        "state": state,
        "batches": batches,
        "batch_count": len(batches),
        "reused_dft_count": len(reused),
        "submitted": False,
    }


def _prepare_batches(runner, state):
    from phase_agent.tools.remote.prepare_options import supported_prepare_options

    prepare = runner.prepare
    options = supported_prepare_options(
        prepare, _owned=True, _defer_index=True, _defer_archive=True
    )
    if len(options) != 3:
        from phase_agent.runtime.turn_process import process_event

        process_event(
            "DFT封装接口兼容模式",
            {"supported_optimizations": sorted(options), "retry_on_error": False},
        )
    batches = []
    while True:
        prepared = prepare(state, **options)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            return state, batches
        batches.append(prepared["batch"])
