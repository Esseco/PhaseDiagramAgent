"""Read-only facts for a verified batch with a small terminal failure.

The recovery ledger keeps failed attempts immutable.  This module only projects
the two safe choices that can follow a verified partial return: continue with
the successful subset, or prepare a retry for the failed task IDs only.
"""

from copy import deepcopy

from phase_agent.analysis.state.dft_round_status import dft_recovery_rounds
from phase_agent.analysis.state.task_failure_facts import task_failure_facts
from phase_agent.tools.state.task_waiting import awaiting_task_result


def _retryable_ids(state):
    return {
        row.get("task_id")
        for row in (task_failure_facts(state).get("actionable_failures") or [])
        if row.get("task_id")
    }


def _fact(*, kind, identity, stage, expected, successful, failed_ids, waiting_ids, scope=None):
    failed_ids = sorted({str(value) for value in failed_ids if value})
    waiting_ids = sorted({str(value) for value in waiting_ids if value})
    expected = int(expected or 0)
    successful = int(successful or 0)
    return {
        "kind": kind,
        "identity": identity,
        "stage": stage,
        "scope": deepcopy(scope) if isinstance(scope, dict) else None,
        "successful_count": successful,
        "expected_count": expected,
        "failed_count": len(failed_ids),
        "failed_task_ids": failed_ids,
        "waiting_task_ids": waiting_ids,
        "coverage": f"{successful}/{expected}",
        "partial": bool(failed_ids and successful and expected and successful < expected),
        "retryable_task_ids": [],
        "choice_required": True,
        "continue_action": "continue_partial",
        "fill_action": "fill_failed_only",
        "instruction": "可继续使用已回收成功结果，或只补全失败任务；不会重跑整批。",
    }


def partial_recovery_facts(state):
    """Return verified current-model partial batches that need a human choice."""

    model = state.get("active_model_version")
    retryable = _retryable_ids(state)
    facts = []

    dft_groups = {}
    for row in dft_recovery_rounds(state):
        scope = row.get("scope") or {}
        if model and scope.get("model_version") != model:
            continue
        # A DFT submission can be split into search groups.  Present one
        # coverage fraction for the upload operation so 12/13 is not shown as
        # separate 10/11 and 2/2 fragments.
        key = (
            scope.get("model_version"),
            scope.get("upload_operation_id"),
            scope.get("parent_relax_round"),
        )
        group = dft_groups.setdefault(
            key,
            {
                "expected": 0,
                "successful": 0,
                "failed_ids": [],
                "waiting_ids": [],
                "search_groups": [],
                "scope": {
                    "model_version": scope.get("model_version"),
                    "upload_operation_id": scope.get("upload_operation_id"),
                    "parent_relax_round": scope.get("parent_relax_round"),
                },
            },
        )
        group["expected"] += int(row.get("expected_tasks") or 0)
        group["successful"] += int(row.get("successful_tasks") or 0)
        group["failed_ids"].extend(row.get("failed_task_ids") or [])
        group["waiting_ids"].extend(row.get("pending_task_ids") or [])
        if scope.get("search_group_index") is not None:
            group["search_groups"].append(scope["search_group_index"])

    for key, group in dft_groups.items():
        if group["waiting_ids"] or not group["failed_ids"]:
            continue
        scope = {
            **group["scope"],
            "search_group_indices": sorted(set(group["search_groups"])),
        }
        identity = "dft:" + ":".join(str(value or "") for value in key)
        fact = _fact(
            kind="dft",
            identity=identity,
            stage="dft",
            expected=group["expected"],
            successful=group["successful"],
            failed_ids=group["failed_ids"],
            waiting_ids=group["waiting_ids"],
            scope=scope,
        )
        if fact["partial"]:
            fact["retryable_task_ids"] = sorted(
                set(fact["failed_task_ids"]) & retryable
            )
            facts.append(fact)

    for bundle_id, batch in (state.get("relax_mc_batches") or {}).items():
        if model and batch.get("model_version") != model:
            continue
        if batch.get("status") in {"cancelled", "superseded"} or not batch.get(
            "import_checksum"
        ):
            continue
        members = [
            row
            for row in state.get("tasks") or []
            if row.get("relax_mc_bundle_id") == bundle_id
        ]
        summary = batch.get("result_summary") or {}
        failed_ids = list(summary.get("failed_task_ids") or [])
        failed_ids.extend(
            row.get("task_id")
            for row in members
            if row.get("status") in {"failed", "timeout"}
        )
        failed_ids = sorted({value for value in failed_ids if value})
        waiting_ids = [
            row.get("task_id") for row in members if awaiting_task_result(row)
        ]
        successful_ids = [
            row.get("task_id") for row in members if row.get("status") == "completed"
        ]
        expected = summary.get("task_count") or len(members)
        resolved_ids = set(successful_ids) | set(failed_ids)
        if (
            waiting_ids
            or not failed_ids
            or not successful_ids
            or len(resolved_ids) < int(expected or 0)
        ):
            continue
        fact = _fact(
            kind="mc",
            identity=bundle_id,
            stage="deep_search",
            expected=expected,
            successful=len(successful_ids),
            failed_ids=failed_ids,
            waiting_ids=waiting_ids,
        )
        if fact["partial"]:
            fact["retryable_task_ids"] = sorted(
                set(fact["failed_task_ids"]) & retryable
            )
            facts.append(fact)
    return facts
