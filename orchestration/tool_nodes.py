"""Tool node adapters; scientific work remains in injected business callbacks."""
from langgraph.runtime import Runtime
from orchestration.runtime_context import WorkflowRuntime, require_runtime


def create_tool_nodes(*, initialize=None, refresh=None, prepare=None, approve=None,
                      validate=None, dispatch=None, finish=None):
    def accept(result, runtime, phase):
        if result.get("_graph_prepared"):
            require_runtime(runtime).frame = result
            return {"phase": phase}
        return {"phase": phase, "response": require_runtime(runtime).finish(result)}

    def initial(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return accept(context.callback("initialize", initialize)(), runtime, "initialized")

    def gated(name, callback, phase):
        def node(state, runtime: Runtime[WorkflowRuntime]):
            context = require_runtime(runtime)
            return accept(context.callback(name, callback)(context.frame), runtime, phase)
        return node

    def validation_node(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        context.frame = context.callback("validate", validate)(context.frame)
        selected = context.frame["action"]["tool"] if context.frame["dispatch_ready"] else ""
        return {"phase": "validated", "selected_tool": selected}

    def dispatch_node(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        context.frame = context.callback("dispatch", dispatch)(context.frame)
        return {"phase": "executed"}

    def finish_node(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return {"phase": "audited", "response": context.finish(
            context.callback("finish", finish)(context.frame))}

    return {
        "initialize_action": initial,
        "model_refresh_barrier": gated("refresh", refresh, "refresh_checked"),
        "prepare_proposal": gated("prepare", prepare, "proposed"),
        "approval_gate": gated("approve", approve, "approval_checked"),
        "validate_action": validation_node,
        "audit_outcome": finish_node,
        "dispatch": dispatch_node,
    }
