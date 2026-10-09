"""Lifecycle node adapters; scientific work remains in injected business callbacks."""
from langgraph.runtime import Runtime
from orchestration.runtime_context import WorkflowRuntime, require_runtime


def create_lifecycle_nodes(*, initialize=None, collect=None, analyze=None, wait=None,
                           assess=None, act=None, finalize=None, action_graph=None):
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

    def waiting(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return accept(context.callback("wait", wait)(context.frame), runtime, "wait_checked")

    def actions(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        # This closure contains the same compiled graph passed to the real loop.
        # LangGraph can discover it and stream its nested node executions.
        callback = context.callback("act", act)
        context.frame = (callback(context.frame, event_graph=action_graph)
                         if action_graph is not None else callback(context.frame))
        return {"phase": "acted"}

    def final(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return {"phase": "finalized", "response": context.finish(
            context.callback("finalize", finalize)(context.frame))}

    return {
        "initialize_confirmed_run": initial,
        "collect_and_reconcile": stage("collect", collect, "collected"),
        "scientific_feedback": stage("analyze", analyze, "analyzed"),
        "results_wait_gate": waiting,
        "assess_and_export_round": stage("assess", assess, "assessed"),
        "bounded_action_graph": actions,
        "prepare_inputs_and_finalize": final,
    }
