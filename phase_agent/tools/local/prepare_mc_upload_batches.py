"""Export approved, budget-reserved MC tasks as portable manual-upload jobs."""

from copy import deepcopy
from pathlib import Path

from phase_agent.configuration.defaults.default_slurm_cluster_config import (
    default_slurm_cluster_config,
)
from phase_agent.tools.remote.manual_upload_runner import ManualUploadBatchRunner
from phase_agent.science.mlip.slurm_executor import create_mlip_task_preparer


def prepare_mc_upload_batches(*, action, context):
    config = context["effective_config"]
    state = deepcopy(context.get("event_state") or {})
    if (state.get("dedup_gate") or {}).get("status") != "ready":
        return {"status": "not_configured", "reason": "structure_dedup_not_ready", "state": state}
    root = config.get("upload_batches_directory")
    command = ((config.get("supercomputer") or {}).get("worker") or {}).get("command") or []
    if not root or not command or any("YOUR_" in str(value) for value in command):
        return {
            "status": "not_configured",
            "reason": "upload_directory_or_worker_command_missing",
            "state": state,
        }
    pending = [
        row
        for row in state.get("tasks") or []
        if row.get("stage") == "deep_search"
        and row.get("status") == "pending"
        and not row.get("slurm_batch_id")
    ]
    if not pending:
        from phase_agent.tools.remote.submission_handoff import finalize_prepared_handoffs

        delivery = finalize_prepared_handoffs(state)
        return {
            "status": "already_prepared",
            "task_count": 0,
            "batch_count": 0,
            "state": state,
            **delivery,
        }
    missing = [
        row.get("task_id")
        for row in pending
        if not row.get("structure_path") or not Path(row["structure_path"]).is_file()
    ]
    if missing:
        return {
            "status": "not_configured",
            "reason": "mc_input_structure_file_missing",
            "task_ids": missing,
            "state": state,
        }
    # A later MC segment references its downloaded final structure directly;
    # its ledger ID is still needed to resolve the branch and full-Na template.
    from phase_agent.science.mlip.slurm_executor import _resolve_structure_id

    for row in pending:
        structure_id = _resolve_structure_id(context["manager"], row)
        record = context["manager"].data["structures"][structure_id]
        if row.get("branch_id") and record.get("branch_id") != row["branch_id"]:
            return {
                "status": "not_configured",
                "reason": "mc_input_structure_branch_mismatch",
                "task_ids": [row["task_id"]],
                "state": context.get("event_state") or {},
            }
        row["structure_id"] = structure_id
    runner = ManualUploadBatchRunner(
        root,
        worker_command=command,
        stage_batch_sizes=((config.get("supercomputer") or {}).get("batch_sizes") or {}),
        stage_profiles=default_slurm_cluster_config(),
        task_preparer=create_mlip_task_preparer(
            context["manager"], context.get("phase_references") or {}, config
        ),
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
        "task_count": sum(len(row["task_ids"]) for row in batches),
        "batch_count": len(batches),
        "batches": [
            {
                "batch_id": row["batch_id"],
                "directory": row["upload_directory"],
                "results_directory": row["results_directory"],
                "task_ids": row["task_ids"],
            }
            for row in batches
        ],
        "submitted": False,
    }
