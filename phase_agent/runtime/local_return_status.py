"""Presence-only facts for registered pending DFT deliveries; never import results."""

from pathlib import Path

from phase_agent.tools.state.task_waiting import awaiting_task_result


def pending_dft_return_facts(state):
    pending = {
        row.get("task_id")
        for row in state.get("tasks") or []
        if row.get("stage") in {"dft_single_point", "dft_relax"} and awaiting_task_result(row)
    }
    if not pending:
        return []
    from phase_agent.tools.remote.legacy_recovery import recovery_handoffs

    deliveries = [(h, h["task_ids"]) for h in recovery_handoffs(state)]
    deliveries.extend(
        (row.get("handoff") or {}, row.get("task_ids") or [])
        for row in state.get("slurm_batches") or []
    )
    facts, covered = [], set()
    for handoff, task_ids in deliveries:
        members = pending.intersection(task_ids) - covered
        directory = handoff.get("return_directory")
        if not members or not directory:
            continue
        root = Path(directory)
        # Match the collector's supported nonrecursive return locations. The
        # presence of a ZIP is not proof of completion, validity or convergence.
        archives = sorted(
            {
                path
                for folder in (root, root / "transfer", root / "return-results")
                for path in folder.glob("*.results.zip")
                if path.is_file() and path.stat().st_size > 0
            }
        )
        if not archives:
            continue
        covered.update(members)
        facts.append(
            {
                "task_ids": sorted(members),
                "directory": str(root),
                "archives": [str(path) for path in archives],
                "status": "awaiting_local_validation",
            }
        )
    return facts
