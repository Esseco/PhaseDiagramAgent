"""Lifecycle node adapters; scientific work remains in injected business callbacks."""

from langgraph.runtime import Runtime
from phase_agent.graphs.runtime_context import WorkflowRuntime, require_runtime


def create_lifecycle_nodes(
    *,
    initialize=None,
    collect=None,
    analyze=None,
    wait=None,
    assess=None,
    act=None,
    finalize=None,
    action_graph=None,
):
    def accept(result, runtime, phase):
        if result.get("_workflow_prepared"):
            require_runtime(runtime).frame = result
            return {"phase": phase}
        return {"phase": phase, "response": require_runtime(runtime).finish(result)}

    def initial(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return accept(context.callback("initialize", initialize)(), runtime, "initialized")

    def stage(name, callback, phase):
        def node(state, runtime: Runtime[WorkflowRuntime]):
            context = require_runtime(runtime)
            context.frame = context.callback(name, callback)(context.frame)
            return {"phase": phase}

        return node

    from phase_agent.graphs.training_handoff_graph import build_training_handoff_graph

    training_graph = build_training_handoff_graph()

    def training(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        if "pre_reconciled" in context.frame and "effective_config" in context.frame:
            from phase_agent.tools.workflows.lifecycle_recovery import _advance_training_workflow

            context.frame = _advance_training_workflow(context.frame, graph=training_graph)
        return {"phase": "training_checked"}

    from phase_agent.graphs.batch_recovery_graph import build_batch_recovery_graph

    batch_graph = build_batch_recovery_graph()

    def batches(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        frame = context.frame
        if "pre_reconciled" in frame:
            from phase_agent.graphs.batch_recovery_graph import recover_batch_stages

            business = frame["pre_reconciled"]["state"]
            reports = recover_batch_stages(
                business,
                frame.get("state_path") or frame.get("effective_config", {}).get("state_path"),
                graph=batch_graph,
            )
            business["batch_recovery_reports"] = reports
        return {"phase": "batches_checked"}

    def direction(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        if "pre_reconciled" in context.frame:
            from phase_agent.tools.workflows.lifecycle_recovery import _review_training_direction

            context.frame = _review_training_direction(context.frame)
        return {"phase": "direction_reviewed"}

    def waiting(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return accept(context.callback("wait", wait)(context.frame), runtime, "wait_checked")

    def actions(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        # Pass the exact approved-action graph directly to the single-action runner.
        # LangGraph can discover it and stream its nested node executions.
        callback = context.callback("act", act)
        context.frame = (
            callback(context.frame, tool_graph=action_graph)
            if action_graph is not None
            else callback(context.frame)
        )
        return {"phase": "acted"}

    def final(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return {
            "phase": "finalized",
            "response": context.finish(context.callback("finalize", finalize)(context.frame)),
        }

    nodes = {
        "initialize_confirmed_run": initial,
        "collect_and_reconcile": stage("collect", collect, "collected"),
        "batch_recovery": batches,
        "training_lifecycle": training,
        "edge_direction_review": direction,
        "scientific_feedback": stage("analyze", analyze, "analyzed"),
        "results_wait_gate": waiting,
        "assess_and_export_round": stage("assess", assess, "assessed"),
        "bounded_action_graph": actions,
        "prepare_inputs_and_finalize": final,
    }

    from time import perf_counter

    def traced(name, callback, child=None):
        def call(state, runtime: Runtime[WorkflowRuntime]):
            if child is not None:
                graph_name = child.name  # Retain compiled child for Studio discovery.
            from phase_agent.graphs.progress_events import emit_progress

            emit_progress(name, "running")
            start = perf_counter()
            error = None
            try:
                context = require_runtime(runtime)
                if context.stages.get("durable") and name != "initialize_confirmed_run":
                    from phase_agent.graphs.business_checkpoint import restore_frame

                    context.frame = restore_frame(context.frame, state.get("business_frame") or {})
                from phase_agent.graphs.cancellation import check_cancelled

                check_cancelled()
                update = callback(state, runtime)
                if context.stages.get("durable"):
                    from phase_agent.graphs.business_checkpoint import checkpoint_frame

                    update["business_frame"] = checkpoint_frame(context.frame)
                    if context.response is not None:
                        import json

                        update["completed_response"] = json.loads(json.dumps(context.response))
                update["current_node"] = name
                business_status = (context.response or {}).get("status", "")
                from phase_agent.graphs.wait_contract import wait_boundary

                waiting = wait_boundary(business_status) is not None
                update["node_status"] = "waiting" if waiting else "completed"
                emit_progress(name, update["node_status"])
                return update
            except Exception as exc:
                error = exc
                emit_progress(name, "failed")
                raise
            finally:
                from phase_agent.graphs.node_trace import append_node_trace

                append_node_trace(
                    require_runtime(runtime).frame, name, perf_counter() - start, error=error
                )

        # Keep the child closure discoverable to Studio.
        return call

    children = {
        "batch_recovery": batch_graph,
        "training_lifecycle": training_graph,
        "bounded_action_graph": action_graph,
    }
    return {name: traced(name, callback, children.get(name)) for name, callback in nodes.items()}
