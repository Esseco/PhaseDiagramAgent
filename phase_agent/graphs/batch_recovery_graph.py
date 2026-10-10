"""Checkpointed stage recovery summaries; no submission or inferred retry authority."""

from copy import deepcopy
from pathlib import Path
from typing import TypedDict
import hashlib
import json

STAGES = ("relax_and_feature", "deep_search", "dft_single_point", "dft_relax")


class BatchState(TypedDict, total=False):
    tasks: list[dict]
    reports: dict
    waiting: bool


def build_batch_recovery_graph(checkpointer=None):
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt
    from phase_agent.tools.state.task_waiting import awaiting_task_result

    def collect(state):
        return {"reports": {}, "waiting": False}

    def stage_node(stage):
        def node(state):
            rows = [row for row in state["tasks"] if row.get("stage") == stage]
            return {
                stage: {
                    "completed_ids": [r["task_id"] for r in rows if r.get("status") == "completed"],
                    "failed_ids": [
                        r["task_id"] for r in rows if r.get("status") in {"failed", "timeout"}
                    ],
                    "waiting_ids": [r["task_id"] for r in rows if awaiting_task_result(r)],
                    "waived_ids": [
                        r["task_id"]
                        for r in rows
                        if r.get("status") in {"pending", "running", "submitted", "unknown"}
                        and not awaiting_task_result(r)
                    ],
                }
            }

        return node

    # Distinct state channels make parallel stage aggregation deterministic.
    class ParallelState(BatchState, total=False):
        relax_and_feature: dict
        deep_search: dict
        dft_single_point: dict
        dft_relax: dict

    graph = StateGraph(ParallelState)
    graph.add_node("reconcile_latest_task_status", collect)
    for stage in STAGES:
        graph.add_node(stage, stage_node(stage))
        graph.add_edge("reconcile_latest_task_status", stage)

    def join(state):
        reports = {stage: state[stage] for stage in STAGES}
        for report in reports.values():
            report["status"] = (
                "partial"
                if report["completed_ids"] and report["waiting_ids"]
                else (
                    "awaiting_results"
                    if report["waiting_ids"]
                    else "failures_require_review"
                    if report["failed_ids"]
                    else "settled"
                )
            )
        return {"reports": reports, "waiting": any(r["waiting_ids"] for r in reports.values())}

    graph.add_node("join_stage_recovery", join)
    graph.add_edge(list(STAGES), "join_stage_recovery")

    def wait(state):
        answer = interrupt({"kind": "batch_results", "reports": state["reports"]})
        if not isinstance(answer, dict) or not isinstance(answer.get("tasks"), list):
            raise ValueError("invalid_batch_resume")
        return {"tasks": answer["tasks"]}

    graph.add_node("wait_for_partial_or_remaining_results", wait)
    graph.add_edge(START, "reconcile_latest_task_status")
    graph.add_conditional_edges(
        "join_stage_recovery",
        lambda s: "wait" if s["waiting"] else "done",
        {"wait": "wait_for_partial_or_remaining_results", "done": END},
    )
    graph.add_edge("wait_for_partial_or_remaining_results", "reconcile_latest_task_status")
    return graph.compile(checkpointer=checkpointer, name="batch_recovery")


def recover_batch_stages(state, state_path, *, graph=None):
    from copy import copy
    from filelock import FileLock
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    tasks = [
        {
            key: deepcopy(row.get(key))
            for key in (
                "task_id",
                "stage",
                "status",
                "model_version",
                "recovery_wait_waived",
                "model_refresh_id",
                "refresh_wait_waived",
            )
        }
        for row in state.get("tasks", [])
        if row.get("stage") in STAGES and row.get("task_id")
    ]
    if not state_path or not tasks:
        return {}
    path = Path(state_path).resolve().parent / "langgraph_batches.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "configurable": {
            "thread_id": hashlib.sha256(str(Path(state_path).resolve()).encode()).hexdigest()
        },
        "recursion_limit": 16,
    }
    with FileLock(str(path) + ".lock", timeout=10):
        with SqliteSaver.from_conn_string(str(path)) as saver:
            executable = copy(graph) if graph is not None else build_batch_recovery_graph(saver)
            executable.checkpointer = saver
            snapshot = executable.get_state(config)
            if snapshot.next and any(t.interrupts for t in snapshot.tasks):
                executable.invoke(Command(resume={"tasks": tasks}), config)
            elif snapshot.next:
                executable.update_state(
                    config, {"tasks": tasks}, as_node="wait_for_partial_or_remaining_results"
                )
                executable.invoke(None, config)
            else:
                executable.invoke({"tasks": tasks}, config)
            return executable.get_state(config).values["reports"]
