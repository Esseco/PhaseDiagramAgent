"""Tool node adapters; scientific work remains in injected business callbacks."""

from langgraph.runtime import Runtime
from phase_agent.graphs.runtime_context import WorkflowRuntime, require_runtime


def create_tool_nodes(
    *,
    initialize=None,
    refresh=None,
    prepare=None,
    approve=None,
    validate=None,
    dispatch=None,
    finish=None,
):
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
            result = context.callback(name, callback)(context.frame)
            if name == "prepare":
                from phase_agent.runtime.decision_visibility import decision_summary
                from phase_agent.runtime.turn_process import process_event

                frame = result if result.get("_graph_prepared") else context.frame
                proposal = frame.get("proposal") or result.get("agent_proposal") or {}
                summary = decision_summary(proposal.get("raw_action") or {})
                if summary:
                    process_event("本轮科学判断（建议，非执行）", summary)
            return accept(result, runtime, phase)

        return node

    def validation_node(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        context.frame = context.callback("validate", validate)(context.frame)
        selected = context.frame["action"]["tool"] if context.frame["dispatch_ready"] else ""
        from phase_agent.runtime.turn_process import process_event

        process_event(
            "实际动作路由",
            {
                "selected_tool": selected or None,
                "dispatch_ready": bool(context.frame["dispatch_ready"]),
                "含义": "通过校验后的路线选择，不代表工具执行成功",
            },
        )
        return {"phase": "validated", "selected_tool": selected}

    def dispatch_node(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        context.frame = context.callback("dispatch", dispatch)(context.frame)
        return {"phase": "executed"}

    def finish_node(state, runtime: Runtime[WorkflowRuntime]):
        context = require_runtime(runtime)
        return {
            "phase": "audited",
            "response": context.finish(context.callback("finish", finish)(context.frame)),
        }

    return {
        "initialize_action": initial,
        "model_refresh_barrier": gated("refresh", refresh, "refresh_checked"),
        "prepare_proposal": gated("prepare", prepare, "proposed"),
        "approval_gate": gated("approve", approve, "approval_checked"),
        "validate_action": validation_node,
        "audit_outcome": finish_node,
        "dispatch": dispatch_node,
    }
