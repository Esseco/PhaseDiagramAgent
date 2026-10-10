"""Human-only decision to wait for or stop waiting for a precise partial DFT round."""

from copy import deepcopy
import hashlib
import json

from phase_agent.analysis.state.dft_round_status import dft_recovery_rounds, dft_round_scope
from phase_agent.tools.state.task_waiting import active_pending_tasks


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
        current.setdefault("dft_recovery_decisions", []).append(
            {
                "question_id": question["question_id"],
                "scope": question["scope"],
                "decision": decision["decision"],
                "user_message": decision.get("user_message"),
                "asked_task_ids": question["pending_task_ids"],
                "waived_task_ids": sorted(changed),
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
