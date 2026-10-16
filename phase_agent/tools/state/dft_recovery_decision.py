"""Human-only decision to wait for or stop waiting for a precise partial DFT round."""

from copy import deepcopy
import hashlib
import json

from phase_agent.analysis.state.dft_round_status import dft_recovery_rounds, dft_round_scope
from phase_agent.tools.state.task_waiting import active_pending_tasks


def cancel_dft_recovery_waits(state, task_ids, *, user_message):
    """Cancel only explicitly selected DFT tracking; never submit remote commands."""
    ids = set(task_ids)
    if not ids or not str(user_message or "").strip():
        raise ValueError("explicit_dft_cancellation_required")
    rows = {t.get("task_id"): t for t in state.get("tasks") or []}
    for task_id in ids:
        task = rows.get(task_id) or {}
        already = task.get("status") == "cancelled" and task.get("recovery_cancelled") is True
        if task.get("stage") not in {"dft_single_point", "dft_relax"} or (
            not already
            and task.get("status")
            not in {"pending", "running", "submitted", "unknown", "submission_unknown"}
        ):
            raise ValueError("dft_cancellation_scope_invalid:" + str(task_id))
    current = deepcopy(state)
    for task in current.get("tasks") or []:
        if task.get("task_id") not in ids or task.get("recovery_cancelled") is True:
            continue
        current.setdefault("dft_recovery_cancellations", []).append(
            {
                "task_id": task["task_id"],
                "previous_status": task["status"],
                "scope": dft_round_scope(task),
                "user_message": str(user_message),
                "remote_cancel_sent": False,
            }
        )
        task.update(
            status="cancelled", recovery_cancelled=True, cancellation_reason=str(user_message)
        )
    if "pending_tasks" in current:
        current["pending_tasks"] = active_pending_tasks(current.get("tasks") or [])
    return current


def classify_dft_recovery_reply(message, state):
    """Only exact replies to a saved question; never infer a choice from 'continue'."""
    question = state.get("pending_dft_recovery_question")
    if not question:
        return None
    text = str(message).strip().lower().rstrip("。.!！").replace(" ", "")
    close = {
        "否",
        "不",
        "不用了",
        "不等了",
        "不回收",
        "不回收了",
        "不再回收",
        "不再等待",
        "取消，不再回收",
        "取消剩余dft",
        "剩下的不回收了",
        "不再回收剩余dft",
        "就用已有结果继续",
        "不回收了，继续",
    }
    wait = {"是", "继续回收", "继续等待", "等剩余结果", "还要回收", "回收剩余dft"}
    # Plain approval words cannot consume a scientific action's approval.
    if not state.get("pending_execution_policies"):
        close |= {"拒绝", "reject", "no"}
        wait |= {"同意", "approve", "yes"}
    choice = "close" if text in close else "wait" if text in wait else None
    return (
        {"decision": choice, "question_id": question["question_id"], "user_message": str(message)}
        if choice
        else None
    )


def update_dft_recovery_question(state, decision=None):
    """Record an exact human choice; do not cancel jobs or settle unknown costs."""
    current = deepcopy(state)
    question = current.get("pending_dft_recovery_question")
    if decision is not None:
        if (
            not question
            or decision.get("question_id") != question.get("question_id")
            or decision.get("decision") not in {"wait", "close"}
        ):
            raise ValueError("DFT recovery question changed; refresh before deciding")
        authorized_ids = set(question["pending_task_ids"])
        changed = []
        if decision["decision"] == "close":
            for task in current.get("tasks") or []:
                if task.get("task_id") in authorized_ids and task.get("status") in {
                    "pending",
                    "running",
                    "submitted",
                    "unknown",
                }:
                    if dft_round_scope(task) != question["scope"]:
                        raise ValueError("DFT task round changed; refresh before deciding")
                    task["recovery_wait_waived"] = True
                    task["recovery_decision_id"] = question["question_id"]
                    changed.append(task["task_id"])
            if changed:
                current = cancel_dft_recovery_waits(
                    current,
                    changed,
                    user_message=decision.get("user_message") or "用户明确不再回收",
                )
        current.setdefault("dft_recovery_decisions", []).append(
            {
                "question_id": question["question_id"],
                "scope": question["scope"],
                "decision": decision["decision"],
                "user_message": decision.get("user_message"),
                "asked_task_ids": question["pending_task_ids"],
                "waived_task_ids": sorted(changed),
                "cancelled_task_ids": sorted(changed),
                "recovered_tasks_at_question": question["recovered_tasks"],
            }
        )
        current.setdefault("dft_recovery_preferences", {})[question["scope_key"]] = {
            "decision": decision["decision"],
            "question_id": question["question_id"],
        }
        current.pop("pending_dft_recovery_question", None)
    current["pending_tasks"] = active_pending_tasks(current.get("tasks") or [])
    rounds = dft_recovery_rounds(current)
    # Keep the question's round stable, but refresh its facts after new returns.
    if question and decision is None:
        rounds.sort(key=lambda row: row["scope_key"] != question["scope_key"])
    for row in rounds:
        if not row["recovered_tasks"] or not row["pending_task_ids"]:
            continue
        evidence = {
            "scope": row["scope"],
            "pending": row["pending_task_ids"],
            "recovered": row["recovered_task_ids"],
        }
        question_id = (
            "dft-recovery-"
            + hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()[:16]
        )
        previous = (current.get("dft_recovery_preferences") or {}).get(row["scope_key"]) or {}
        if previous.get("decision") == "wait" and previous.get("question_id") == question_id:
            continue
        current["pending_dft_recovery_question"] = {**row, "question_id": question_id}
        return current
    current.pop("pending_dft_recovery_question", None)
    return current
