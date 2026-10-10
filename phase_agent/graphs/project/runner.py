"""Run the one project graph with durable invocation identity."""

from phase_agent.graphs.runtime_context import WorkflowRuntime


def run_search_workflow_graph(
    *, workflow_graph=None, checkpoint_path=None, invocation_id=None, project_thread=None, **stages
):
    if workflow_graph is None:
        from phase_agent.graphs.scientific_graph import current_scientific_graph

        workflow_graph = current_scientific_graph()
    if checkpoint_path and invocation_id:
        from pathlib import Path
        from copy import copy
        from filelock import FileLock
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "configurable": {"thread_id": project_thread or invocation_id},
            "recursion_limit": 32,
        }
        with FileLock(str(path) + ".lock", timeout=10):
            with SqliteSaver.from_conn_string(str(path)) as saver:
                graph = copy(workflow_graph)
                graph.checkpointer = saver
                snapshot = graph.get_state(config)
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
                if snapshot.next:
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
                from langgraph.types import Command

                if snapshot.next and any(task.interrupts for task in snapshot.tasks):
                    value = Command(resume={"signal": "reconcile", "request_id": invocation_id})
                elif snapshot.next:
                    value = None
                else:
                    value = {
                        "request_id": invocation_id,
                        "response": None,
                        "completed_response": None,
                        "business_frame": {},
                    }
                graph.invoke(value, config, context=context)
                if context.response is None:
                    raise RuntimeError("Durable lifecycle did not return a response")
                return context.response
    context = WorkflowRuntime(stages=stages, expose_response=False)
    workflow_graph.invoke({}, {"recursion_limit": 32}, context=context)
    if context.response is None:
        raise RuntimeError("Scientific workflow did not return a business response")
    return context.response
