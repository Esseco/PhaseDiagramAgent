"""Bounded action iteration; business callbacks own explicit persistence."""
from orchestration.state import EventGraphState
from orchestration.runtime_context import EventLoopRuntime
from langgraph.runtime import Runtime


def build_event_loop_graph(*, execute=None, persist=None, route=None, tool_graph=None):
    from langgraph.graph import StateGraph, START, END

    if any(callback is not None for callback in (execute, persist, route)):
        if not all(callable(callback) for callback in (execute, persist, route)):
            raise ValueError("All bound event loop callbacks are required")
        execute_node, persist_node, route_node = execute, persist, route
    else:
        def context_of(runtime):
            if not isinstance(runtime.context, EventLoopRuntime):
                raise ValueError("Pass context=EventLoopRuntime() to the action loop")
            return runtime.context

        def execute_node(state, runtime: Runtime[EventLoopRuntime]):
            context = context_of(runtime)
            context.outcome = (context.execute(state, tool_graph=tool_graph)
                               if tool_graph is not None else context.execute(state))
            return {"step_id": context.outcome["step_id"],
                    "response": {"status": context.outcome["response"]["status"]}}

        def persist_node(state, runtime: Runtime[EventLoopRuntime]):
            context = context_of(runtime)
            return context.persist({**state, **context.outcome})

        def route_node(state, runtime: Runtime[EventLoopRuntime]):
            return context_of(runtime).route(state)

    graph = StateGraph(EventGraphState, context_schema=EventLoopRuntime)
    graph.add_node("execute_validated_action", execute_node)
    graph.add_node("persist_action_outcome", persist_node)
    graph.add_edge(START, "execute_validated_action")
    graph.add_edge("execute_validated_action", "persist_action_outcome")
    graph.add_conditional_edges("persist_action_outcome", route_node,
                                [END, "execute_validated_action"])
    return graph.compile(name="bounded_action_iteration")


def run_event_loop_graph(*, max_steps, execute, persist, route, event_graph=None):
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps 必须是正整数")
    if event_graph is None:
        return build_event_loop_graph(execute=execute, persist=persist, route=route).invoke(
            {"offset": 0}, {"recursion_limit": max_steps * 2 + 5})
    return event_graph.invoke({"offset": 0}, {"recursion_limit": max_steps * 2 + 5},
        context=EventLoopRuntime(execute=execute, persist=persist, route=route))
