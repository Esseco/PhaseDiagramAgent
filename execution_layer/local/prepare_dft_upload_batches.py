"""Materialize only already-approved DFT tasks for manual upload."""
from copy import deepcopy
from execution_layer.remote.manual_upload_runner import ManualUploadBatchRunner
from scientific_layer.dft.prepare_pycode_relax import prepare_pycode_relax


def prepare_dft_upload_batches(*, action, context):
    state = deepcopy(context.get("event_state") or {})
    config = context["effective_config"]
    root = config.get("upload_batches_directory")
    if not root:
        return {"status": "not_configured", "reason": "upload_directory_missing", "state": state}
    requested = set(action.get("target_ids") or [])
    tasks = [row for row in state.get("tasks") or []
             if row.get("stage") in {"dft_relax", "dft_single_point"}
             and row.get("status") == "pending" and not row.get("slurm_batch_id")
             and (not requested or row.get("task_id") in requested)]
    if any(row["stage"] != "dft_relax" for row in tasks):
        raise ValueError("当前 Py-Code 接口只到结构优化；DFT 单点方案需要重新确认")
    runner = ManualUploadBatchRunner(root, worker_command=[],
        dispatcher=lambda task: prepare_pycode_relax(task, manager=context["manager"]))
    # Restrict the existing runner to this approved preparation request.
    select = runner._select
    ids = {row["task_id"] for row in tasks}
    runner._select = lambda current, rows: select(current, [row for row in rows if row.get("task_id") in ids])
    batches = []
    while True:
        prepared = runner.prepare(state)
        state = prepared["state"]
        if prepared["status"] == "no_tasks":
            break
        batches.append(prepared["batch"])
    return {"status": "prepared" if batches else "already_prepared", "state": state,
            "batches": batches, "batch_count": len(batches), "submitted": False}
