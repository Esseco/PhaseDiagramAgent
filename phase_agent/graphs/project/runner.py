"""Run the one project graph with durable invocation identity."""

from phase_agent.graphs.runtime_context import WorkflowRuntime


def _stale_lifecycle_recovery(graph, config, thread, invocation_id, snapshot):
    """Inspect an isolated recovery thread without replaying the old graph."""
    import hashlib

    recovery_thread = (
        f"{thread}:recovery:{hashlib.sha256(str(invocation_id).encode()).hexdigest()[:16]}"
    )
    recovery_config = {
        **config,
        "configurable": {**config.get("configurable", {}), "thread_id": recovery_thread},
    }
    recovered = graph.get_state(recovery_config, subgraphs=True)
    completed = (recovered.values or {}).get("completed_response")
    if completed is not None:
        return completed
    old_request = (snapshot.values or {}).get("request_id")
    return {
        "status": "execution_reconciliation_required",
        "submitted": False,
        "steps_executed": 0,
        "recovery_report": {
            "unsettled": [
                {
                    "identity": {
                        "invocation_id": old_request or invocation_id,
                        "tool": "scientific_lifecycle",
                    },
                    "reason": "unsafe_inflight_checkpoint",
                    "next_action": (
                        "核对旧生命周期检查点、动作审批及执行回执后再提出新方案；"
                        "本轮未重放原动作。"
                    ),
                }
            ],
            "checks": [],
        },
    }


def _read_business_state(state_path):
    if not state_path:
        return None
    try:
        from phase_agent.tools.step_runner.file_protocol import read_json

        state = read_json(state_path, {}) or {}
    except (OSError, ValueError, TypeError):
        return None
    return state if isinstance(state, dict) else None


def _business_state_is_clean_for_stale_checkpoint(
    state_path, *, pending_plan_id=None, state=None
):
    """Allow a fresh request only when current business facts are settled.

    A stale graph checkpoint may contain an old action/approval frame.  We do
    not replay that frame.  If the authoritative business state has a clear
    receipt report, no pending approval/recovery, no active reservation and no
    running task, the old frame can be isolated and a new request can start
    from current facts.  Missing or incomplete evidence keeps the conservative
    reconciliation block.
    """
    if not state_path:
        return False
    if state is None:
        state = _read_business_state(state_path)
    if state is None:
        return False
    recovery = state.get("execution_recovery_report") or {}
    if recovery.get("status") != "clear" or recovery.get("unsettled"):
        return False
    pending = state.get("pending_execution_policies") or {}
    if pending:
        # A reviewed plan is the one safe exception to the normal pending
        # approval fence.  The approval path is about to consume exactly this
        # plan from a fresh lifecycle; unrelated or multiple plans remain a
        # hard stop.
        if not pending_plan_id or set(pending) != {pending_plan_id}:
            return False
    if state.get("pending_execution_recoveries"):
        return False
    active_reservations = {"reserved", "submitted", "running", "unknown"}
    current_config = state.get("confirmed_config_version")
    for reservation in (state.get("budget_reservations") or {}).values():
        if not isinstance(reservation, dict):
            return False
        if reservation.get("reconciliation_required") or reservation.get("status") in active_reservations:
            # A reservation from an older confirmed configuration is an
            # accounting record for a superseded action.  It must not fence a
            # fresh request after the authoritative recovery report is clear;
            # current-config reservations remain a hard safety boundary.
            if (
                reservation.get("status") in active_reservations
                and current_config
                and reservation.get("config_version")
                and reservation.get("config_version") != current_config
                and not reservation.get("reconciliation_required")
            ):
                continue
            return False
    active_tasks = {"pending", "queued", "submitted", "running", "unknown"}
    for task in state.get("tasks") or []:
        if isinstance(task, dict) and task.get("status") in active_tasks:
            return False
    return True


def _start_fresh_lifecycle_request(
    graph, config, thread, invocation_id, stages, *, initial_state=None
):
    """Run the current request on an isolated thread after stale-frame fencing."""
    import hashlib

    fresh_thread = (
        f"{thread}:fresh:{hashlib.sha256(str(invocation_id).encode()).hexdigest()[:16]}"
    )
    fresh_config = {
        **config,
        "configurable": {**config.get("configurable", {}), "thread_id": fresh_thread},
    }
    from phase_agent.graphs.runtime_context import WorkflowRuntime

    context = WorkflowRuntime(stages={**stages, "durable": True}, expose_response=False)
    graph.invoke(
        initial_state
        or {
            "request_id": invocation_id,
            "response": None,
            "completed_response": None,
            "business_frame": {},
        },
        fresh_config,
        context=context,
    )
    if context.response is None:
        raise RuntimeError("Fresh lifecycle request did not return a response")
    from phase_agent.runtime.turn_process import process_event

    process_event(
        "隔离旧生命周期检查点并启动当前请求",
        {
            "old_thread": thread,
            "fresh_thread": fresh_thread,
            "old_checkpoint_replayed": False,
            "business_state_recovery": "clear",
        },
    )
    return context.response
def _rebind_pending_subgraphs(graph, snapshot, request_id):
    """Bind a safe retry through every pending child before it runs.

    The parent checkpoint can carry a pending observation subgraph whose own
    request_id is still the previous message. Updating the child at its last
    completed node keeps its pending queue and business frame intact.
    """
    for task in snapshot.tasks:
        child = getattr(task, "state", None)
        if child is None or not getattr(child, "next", None):
            continue
        _rebind_pending_subgraphs(graph, child, request_id)
        values = getattr(child, "values", {}) or {}
        node = values.get("current_node")
        if node:
            graph.update_state(child.config, {"request_id": request_id}, as_node=node)


def _pending_approval_for_request(state_path, invocation_id, human_feedback, *, state=None):
    """Return the exact pending plan authorized by this approval turn.

    The review surface validates the proposal before entering the lifecycle.
    Re-checking the durable plan here lets an old/incomplete graph checkpoint
    be isolated safely without allowing arbitrary feedback into a running
    request.
    """
    if not isinstance(human_feedback, dict) or human_feedback.get("decision") not in {
        "approve",
        "reject",
        "comment",
    }:
        return None
    if not state_path:
        return None
    if state is None:
        state = _read_business_state(state_path)
    if state is None:
        return None
    pending = state.get("pending_execution_policies") or {}
    if not isinstance(pending, dict) or invocation_id not in pending:
        return None
    return invocation_id


def run_search_workflow_graph(
    *,
    workflow_graph=None,
    checkpoint_path=None,
    invocation_id=None,
    project_thread=None,
    business_state_path=None,
    human_feedback=None,
    **stages,
):
    if workflow_graph is None:
        from phase_agent.graphs.scientific_graph import current_scientific_graph

        workflow_graph = current_scientific_graph()
    if checkpoint_path and invocation_id:
        from pathlib import Path
        from copy import copy
        from filelock import FileLock
        from phase_agent.persistence.compact_checkpoints import CompactSqliteSaver as SqliteSaver

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from phase_agent.tools.state.workflow_control import current_revision

        revision = current_revision(business_state_path)
        thread = project_thread or invocation_id
        if revision:
            # Keep old checkpoints intact; an audited control revision starts a fresh lifecycle.
            thread += ":control:" + revision
        config = {
            "configurable": {"thread_id": thread},
            "recursion_limit": 32,
            "max_concurrency": 1,
        }
        with FileLock(str(path) + ".lock", timeout=10):
            # The project lifecycle only needs the latest checkpoint and its
            # immediate parent to resume safely.  A generic eight-entry window
            # multiplied large business frames across every chat thread.
            with SqliteSaver.from_conn_string(
                str(path), history_keep=3, namespace_keep=8
            ) as saver:
                graph = copy(workflow_graph)
                graph.checkpointer = saver
                snapshot = graph.get_state(config, subgraphs=True)
                if (
                    not snapshot.next
                    and snapshot.values.get("completed_response") is not None
                    and (not project_thread or snapshot.values.get("request_id") == invocation_id)
                ):
                    return snapshot.values["completed_response"]
                if (
                    snapshot.next
                    and any(task.interrupts for task in snapshot.tasks)
                    and snapshot.values.get("request_id") == invocation_id
                ):
                    return snapshot.values["completed_response"]
                context = WorkflowRuntime(stages={**stages, "durable": True}, expose_response=False)
                # A native project wait is a completed request boundary. The
                # resume command below enters initialize with fresh facts; it
                # must not restore the old request's configuration/frame.
                new_request_at_wait = (
                    bool(project_thread)
                    and tuple(snapshot.next) == ("wait_for_project_input",)
                    and any(task.interrupts for task in snapshot.tasks)
                    and (
                        snapshot.values.get("request_id") != invocation_id
                        or human_feedback is not None
                    )
                )
                approval_state = (
                    _read_business_state(business_state_path) if human_feedback is not None else None
                )
                pending_approval = _pending_approval_for_request(
                    business_state_path,
                    invocation_id,
                    human_feedback,
                    state=approval_state,
                )
                different_inflight_request = (
                    bool(snapshot.next)
                    and not new_request_at_wait
                    and snapshot.values.get("request_id") != invocation_id
                )
                if snapshot.next and not new_request_at_wait:
                    if human_feedback is not None:
                        if pending_approval and _business_state_is_clean_for_stale_checkpoint(
                            business_state_path,
                            pending_plan_id=pending_approval,
                            state=approval_state,
                        ):
                            return _start_fresh_lifecycle_request(
                                graph,
                                config,
                                thread,
                                invocation_id,
                                stages,
                                initial_state={
                                    "request_id": invocation_id,
                                    "response": None,
                                    "completed_response": None,
                                    "business_frame": {},
                                },
                            )
                        raise ValueError(
                            "Cannot apply new human feedback to an in-flight lifecycle invocation"
                        )
                    if different_inflight_request:
                        from phase_agent.runtime.studio_recovery import safe_read_checkpoint

                        if not safe_read_checkpoint(snapshot):
                            if _business_state_is_clean_for_stale_checkpoint(business_state_path):
                                return _start_fresh_lifecycle_request(
                                    graph,
                                    config,
                                    thread,
                                    invocation_id,
                                    stages,
                                )
                            return _stale_lifecycle_recovery(
                                graph,
                                config,
                                thread,
                                invocation_id,
                                snapshot,
                            )
                        # The root Command below updates the parent. Bind the
                        # pending child checkpoints first so their completion
                        # cannot restore the old request ID.
                        _rebind_pending_subgraphs(graph, snapshot, invocation_id)
                    prepared = stages["initialize"]()
                    if not prepared.get("_workflow_prepared"):
                        raise ValueError("Cannot restore lifecycle runtime dependencies")
                    if prepared.get("snapshot", {}).get("config_version") != (
                        snapshot.values.get("business_frame") or {}
                    ).get("snapshot", {}).get("config_version"):
                        raise ValueError(
                            "Lifecycle configuration changed; cannot replay old invocation"
                        )
                    context.frame = prepared
                    context.stages["restore_business_frame"] = True
                from langgraph.types import Command

                if snapshot.next and any(task.interrupts for task in snapshot.tasks):
                    value = Command(resume={"signal": "reconcile", "request_id": invocation_id})
                elif snapshot.next:
                    value = (
                        Command(update={"request_id": invocation_id})
                        if different_inflight_request
                        else None
                    )
                else:
                    value = {
                        "request_id": invocation_id,
                        "response": None,
                        "completed_response": None,
                        "business_frame": {},
                    }
                try:
                    graph.invoke(value, config, context=context)
                except Exception:
                    # A safe read-only node may fail again. Keep the same new
                    # request retryable without changing any scientific facts.
                    if different_inflight_request:
                        _rebind_pending_subgraphs(
                            graph, graph.get_state(config, subgraphs=True), invocation_id
                        )
                    raise
                if context.response is None:
                    raise RuntimeError("Durable lifecycle did not return a response")
                return context.response
    context = WorkflowRuntime(stages=stages, expose_response=False)
    workflow_graph.invoke({}, {"recursion_limit": 32}, context=context)
    if context.response is None:
        raise RuntimeError("Scientific workflow did not return a business response")
    return context.response
