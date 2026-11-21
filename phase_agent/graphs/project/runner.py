"""Run the one project graph with durable invocation identity."""

from phase_agent.graphs.runtime_context import WorkflowRuntime


def _stale_lifecycle_recovery(graph, config, thread, invocation_id, snapshot):
    """Report the current block; legacy reply caches cannot prove completion."""
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
                        "核对旧生命周期检查点、动作审批及执行回执后再提出新方案；本轮未重放原动作。"
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


def _settle_completed_input_delivery(state_path):
    """Canonicalize proven local delivery before examining old lifecycle frames."""
    if state_path:
        from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state
        from phase_agent.runtime.turn_process import has_reusable_initial_state
        from phase_agent.tools.workflows.training_input_settlement import training_input_eligible

        if not has_reusable_initial_state(state_path):
            indexed = open_indexed_state(state_path)
            if indexed is not None:
                try:
                    needs_settlement = any(
                        training_input_eligible(
                            indexed,
                            record.get("final_action") or {},
                            record.get("execution_result") or {},
                        )
                        for record in indexed.get("action_records") or []
                    )
                    indexed.verify(strong=True)
                    if not needs_settlement:
                        return None
                except IndexUnavailable:
                    pass  # Fall back to the authoritative full-state reader.
    from phase_agent.persistence.state.read_versions import read_identity

    expected_identity = read_identity(state_path, strong=True) if state_path else None
    state = _read_business_state(state_path)
    if state is None:
        return None
    from phase_agent.tools.workflows.training_input_settlement import settle_saved_training_inputs
    from phase_agent.tools.step_runner.file_protocol import write_json_if_version

    settled = settle_saved_training_inputs(state)
    if settled is not state:
        write_json_if_version(state_path, settled, expected_identity)
    return settled


def _finished_generation_hold(state_path, state, key, reservation):
    """A saved, completed local generation is not an in-flight task.

    Keep its reservation and all accounting intact.  A return alone is not
    enough: both the claimed action and the saved business record must match.
    """
    if (
        reservation.get("stage") != "generate_branches"
        or reservation.get("status") != "reserved"
        or reservation.get("reconciliation_required")
        or reservation.get("reserved_cost", reservation.get("relative_cost")) != 0
    ):
        return False
    from phase_agent.tools.state.execution_receipts import execution_events
    from phase_agent.policies.file_approval import proposal_hash

    matches = []
    for invocation, record in (state.get("invocations") or {}).items():
        if not isinstance(record, dict):
            continue
        action = record.get("final_action") or record.get("action") or {}
        execution = record.get("execution") or record.get("execution_result") or {}
        if (
            action.get("tool") == "generate_branches"
            and action.get("task_key") == key
            and execution.get("status") == "completed"
        ):
            matches.append((invocation, action))
    if len(matches) != 1:
        return False
    invocation, action = matches[0]
    events = execution_events(state_path, invocation)
    by_name = {row["event"]: row["payload"] for row in events}
    claimed = by_name.get("claimed") or {}
    returned = by_name.get("tool_returned") or {}
    saved = by_name.get("business_saved") or {}
    identity = claimed.get("identity") or {}
    return bool(
        identity.get("invocation_id") == invocation
        and identity.get("tool") == "generate_branches"
        and identity.get("config_version") == reservation.get("config_version")
        and identity.get("action_hash") == proposal_hash(action)
        and identity.get("action_hash") == proposal_hash(claimed.get("action") or {})
        and (claimed.get("action") or {}).get("task_key") == key
        and returned.get("identity") == identity
        and returned.get("outcome") == "completed"
        and saved.get("identity") == identity
        and saved.get("state_file_sha256")
    )


def _business_state_is_clean_for_stale_checkpoint(state_path, *, pending_plan_id=None, state=None):
    """Allow a fresh request only when current business facts are settled.

    A stale graph checkpoint may contain an old action/approval frame.  We do
    not replay that frame.  If the authoritative business state has a clear
    receipt report, no pending approval/recovery, no active execution reservation
    and no running task, the old frame can be isolated and a new request can start
    from current facts.  Missing or incomplete evidence keeps the conservative
    reconciliation block.
    """
    if not state_path:
        return False
    if state is None:
        state = _read_business_state(state_path)
    if state is None:
        return False
    # The stored report is a snapshot and can remain stale after an audited
    # reconciliation. Do not infer clearance from a missing or unexplained report.
    recovery = state.get("execution_recovery_report") or {}
    if recovery.get("status") != "clear" or recovery.get("unsettled"):
        old_issues = recovery.get("unsettled") or []
        audited = state.get("execution_reconciliations") or {}
        def audited_issue(item):
            invocation = (item.get("identity") or {}).get("invocation_id")
            record = audited.get(invocation) or {}
            return bool(invocation and record.get("resolution") in {
                "verified_no_effect", "verified_registered_effects", "verified_abandoned_staging"
            } and (record.get("identity") or {}).get("invocation_id") == invocation
                and record.get("evidence") and record.get("reviewed_at"))

        if not old_issues or not all(audited_issue(item) for item in old_issues):
            return False
    from phase_agent.tools.state.execution_receipts import recovery_report

    import sqlite3

    try:
        current_receipts = recovery_report(state_path, state)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return False
    unsettled = current_receipts.get("unsettled") or []
    if current_receipts.get("status") != "clear" or unsettled:
        # A newly approved plan claims its invocation before the durable
        # runner can isolate an old unsafe checkpoint.  Permit that exact
        # handoff only when evidence proves the claim has produced no effect;
        # any file, task, phase record, or registered output keeps the fence.
        if not pending_plan_id or len(unsettled) != 1:
            return False
        identity = unsettled[0].get("identity") or {}
        if identity.get("invocation_id") != pending_plan_id:
            return False
        from phase_agent.tools.state.execution_recovery_evidence import inspect_execution

        try:
            checks = inspect_execution(state, unsettled[0], state_path=state_path)
        except (OSError, ValueError, TypeError, sqlite3.Error):
            return False
        registered_effects = [
            row
            for row in checks.get("registered_branch_outputs") or []
            if row.get("status") != "invalid"
            or row.get("registered_branch_count", 0)
            or row.get("registered_structure_count", 0)
            or row.get("new_branch_ids")
            or row.get("structure_ids")
        ]
        if (
            checks.get("task_count")
            or checks.get("phase_record_ids")
            or registered_effects
            or any(row.get("status") == "file" for row in checks.get("artifact_checks") or [])
            or (checks.get("preparation_progress") or {}).get("boundary")
            in {"effects_possible", "complete"}
        ):
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
    from phase_agent.tools.workflows.model_refresh_scope import is_refresh_planning_hold
    from phase_agent.tools.state.scheduler_terminal_evidence import terminal_reservation_is_proven

    for key, reservation in (state.get("budget_reservations") or {}).items():
        if not isinstance(reservation, dict):
            return False
        if is_refresh_planning_hold(state, key, reservation):
            continue  # Keep the allowance; fresh analysis decides its next use.
        if terminal_reservation_is_proven(state, key, reservation):
            continue  # Terminal execution is separate from unresolved cost accounting.
        from phase_agent.tools.state.scheduler_terminal_evidence import user_waived_dft_reservation
        if user_waived_dft_reservation(state, key, reservation):
            continue  # User stopped waiting; keep the existing unsettled cost reservation.
        try:
            if _finished_generation_hold(state_path, state, key, reservation):
                continue
        except (OSError, ValueError, TypeError, sqlite3.Error):
            return False
        if (
            reservation.get("reconciliation_required")
            or reservation.get("status") in active_reservations
        ):
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
        if node == "initialize_confirmed_run" and tuple(child.next) == ("collect_and_reconcile",):
            # The observation subgraph inherits the parent's last node before
            # its first node has run.  That label is not a child graph node.
            node = "__start__"
        if node:
            graph.update_state(child.config, {"request_id": request_id}, as_node=node)


def _validate_verified_resume(snapshot, invocation_id, evidence, row):
    import hashlib

    if (
        not isinstance(evidence, dict)
        or snapshot.config["configurable"].get("checkpoint_id") != evidence.get("checkpoint_id")
        or (snapshot.values or {}).get("request_id") != invocation_id
        or evidence.get("request_id") != invocation_id
        or tuple(snapshot.next) != (evidence.get("resume_next", "react"),)
        or evidence.get("resume_next", "react") not in {"react", "analyze"}
        or not row
        or hashlib.sha256(row[0]).hexdigest() != evidence.get("checkpoint_sha256")
    ):
        raise ValueError("verified_resume_checkpoint_changed")


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
        from phase_agent.persistence.state.field_index import IndexUnavailable, open_indexed_state

        indexed = open_indexed_state(state_path)
        if indexed is not None:
            try:
                pending = indexed.get("pending_execution_policies") or {}
                indexed.verify(remember=True)
                return invocation_id if isinstance(pending, dict) and invocation_id in pending else None
            except IndexUnavailable:
                pass  # The authoritative reader also supports unindexed and older state.
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
    resume_verified_checkpoint=None,
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
        from phase_agent.persistence.lifecycle_routes import (
            lifecycle_base_thread,
            current_lifecycle_thread,
            fresh_lifecycle_thread,
            activate_lifecycle_thread,
        )

        base_thread = lifecycle_base_thread(project_thread or invocation_id, business_state_path)
        thread = base_thread
        config = {
            "configurable": {"thread_id": thread},
            "recursion_limit": 32,
            "max_concurrency": 1,
        }
        with FileLock(str(path) + ".lock", timeout=10):
            business_state = _settle_completed_input_delivery(business_state_path)
            # The project lifecycle only needs the latest checkpoint and its
            # immediate parent to resume safely.  A generic eight-entry window
            # multiplied large business frames across every chat thread.
            with SqliteSaver.from_conn_string(str(path), history_keep=3, namespace_keep=8) as saver:
                graph = copy(workflow_graph)
                graph.checkpointer = saver
                thread = current_lifecycle_thread(saver.conn, base_thread)
                config["configurable"]["thread_id"] = thread
                # This pass only selects the request, route and resume boundary.
                # Actual execution loads and verifies the persisted business frame.
                with saver.serde.routing_only():
                    snapshot = graph.get_state(config, subgraphs=True)
                if resume_verified_checkpoint is not None:
                    row = saver.conn.execute(
                        "SELECT checkpoint FROM checkpoints WHERE thread_id=? AND checkpoint_ns='' "
                        "ORDER BY checkpoint_id DESC LIMIT 1",
                        (thread,),
                    ).fetchone()
                    _validate_verified_resume(
                        snapshot, invocation_id, resume_verified_checkpoint, row
                    )
                if thread != base_thread:
                    if not snapshot.values:
                        raise ValueError(
                            "当前生命周期路由缺少检查点，请核对持久化数据库；未重放动作。"
                        )
                    activate_lifecycle_thread(saver.conn, base_thread, thread)
                approval_state = business_state if human_feedback is not None else None
                pending_approval = _pending_approval_for_request(
                    business_state_path, invocation_id, human_feedback, state=approval_state
                )
                if (
                    not snapshot.next
                    and not pending_approval
                    and snapshot.values.get("completed_response") is not None
                    and (not project_thread or snapshot.values.get("request_id") == invocation_id)
                ):
                    return graph.get_state(config).values["completed_response"]
                if (
                    snapshot.next
                    and not pending_approval
                    and any(task.interrupts for task in snapshot.tasks)
                    and snapshot.values.get("request_id") == invocation_id
                ):
                    return graph.get_state(config).values["completed_response"]
                context = WorkflowRuntime(
                    stages={**stages, "durable": True, "seal_checkpoint_lists": True},
                    expose_response=False,
                )

                # A native wait completed its prior request. All subsequent
                # input resumes this same selected lifecycle, even with active tasks.
                def at_project_wait():
                    return (
                        bool(project_thread)
                        and tuple(snapshot.next) == ("wait_for_project_input",)
                        and any(task.interrupts for task in snapshot.tasks)
                        and (
                            snapshot.values.get("request_id") != invocation_id
                            or human_feedback is not None
                        )
                    )

                new_request_at_wait = at_project_wait()
                different_inflight_request = (
                    bool(snapshot.next)
                    and not new_request_at_wait
                    and snapshot.values.get("request_id") != invocation_id
                )
                from phase_agent.runtime.studio_recovery import (
                    safe_read_checkpoint,
                    initial_checkpoint,
                )

                needs_isolation = (
                    bool(snapshot.next)
                    and not new_request_at_wait
                    and (
                        (
                            human_feedback is not None
                            and not (
                                snapshot.values.get("request_id") == invocation_id
                                and safe_read_checkpoint(snapshot)
                            )
                        )
                        or (different_inflight_request and not safe_read_checkpoint(snapshot))
                    )
                )
                if needs_isolation:
                    clean = (human_feedback is None or pending_approval) and (
                        _business_state_is_clean_for_stale_checkpoint(
                            business_state_path,
                            pending_plan_id=pending_approval,
                            state=business_state,
                        )
                    )
                    if not clean:
                        if human_feedback is not None:
                            raise ValueError(
                                "Cannot apply new human feedback to an in-flight lifecycle invocation"
                            )
                        return _stale_lifecycle_recovery(
                            graph, config, thread, invocation_id, snapshot
                        )
                    previous_thread = thread
                    thread = fresh_lifecycle_thread(base_thread, invocation_id)
                    config["configurable"]["thread_id"] = thread
                    snapshot = graph.get_state(config, subgraphs=True)
                    if snapshot.values:
                        # A prior interrupted retry may have persisted a
                        # partial fresh checkpoint.  The business-state
                        # evidence above has already proved that it has no
                        # registered effect, so allocate one more deterministic
                        # retry thread instead of replaying that checkpoint.
                        from time import time_ns

                        thread = fresh_lifecycle_thread(
                            base_thread, f"{invocation_id}:retry:{time_ns()}"
                        )
                        config["configurable"]["thread_id"] = thread
                        snapshot = graph.get_state(config, subgraphs=True)
                        if snapshot.values:
                            raise ValueError("隔离请求已有检查点，不能重新初始化未完成动作。")
                    # Persist the ordinary graph start before committing its route.
                    # A hard crash between the two is adopted from the newest
                    # root checkpoint; no business callback has run at this point.
                    graph.update_state(
                        config,
                        {
                            "request_id": invocation_id,
                            "response": None,
                            "completed_response": None,
                            "business_frame": {},
                        },
                        as_node="__start__",
                    )
                    snapshot = graph.get_state(config, subgraphs=True)
                    activate_lifecycle_thread(saver.conn, base_thread, thread)
                    from phase_agent.runtime.turn_process import process_event

                    process_event(
                        "隔离旧生命周期检查点并接续当前请求",
                        {
                            "old_thread": previous_thread,
                            "fresh_thread": thread,
                            "old_checkpoint_replayed": False,
                            "business_state_recovery": "clear",
                        },
                    )
                    new_request_at_wait = False
                    different_inflight_request = False
                if snapshot.next and not new_request_at_wait and not initial_checkpoint(snapshot):
                    if different_inflight_request:
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
                # graph.invoke reads the checkpoint again. Retaining the prior
                # materialized snapshot and business-state copy here can double
                # peak memory for large approval continuations.
                # The wait node clears the previous turn's business frame and reply.
                # Share verified, read-only history only while loading that exact checkpoint.
                wait_checkpoint_id = (
                    snapshot.config["configurable"].get("checkpoint_id")
                    if new_request_at_wait
                    else None
                )
                business_state = approval_state = snapshot = None
                try:
                    with saver.serde.share_wait_history(wait_checkpoint_id):
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
