"""Invocation-local dependencies, deliberately excluded from graph state."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowRuntime:
    """One instance per invoke/stream; never share across concurrent runs.

    Stage adapters carry managers, clients and their invocation-local frame.
    Durable facts are projected into the serializable business_frame channel;
    live dependencies are rebuilt and must not be persisted in graph state.
    """

    frame: dict = field(default_factory=dict)
    stages: dict = field(default_factory=dict)
    response: dict | None = None
    expose_response: bool = True

    def callback(self, name, bound=None):
        callback = bound if bound is not None else self.stages.get(name)
        if not callable(callback):
            raise ValueError(f"Missing workflow runtime stage: {name}")
        return callback

    def finish(self, result):
        self.response = result
        return result if self.expose_response else {"status": result.get("status", "returned")}


def require_runtime(runtime):
    if not isinstance(runtime.context, WorkflowRuntime):
        raise ValueError("Pass context=WorkflowRuntime() for each graph invocation")
    return runtime.context


@dataclass(frozen=True)
class ProposalRuntime:
    """Decision dependencies only; never embedded in proposal graph state."""

    agent_client: Any
    payload: dict


@dataclass
class EventLoopRuntime:
    """Callbacks and full tool outcomes stay out of Studio checkpoint channels."""

    execute: Any
    persist: Any
    route: Any
    outcome: dict = field(default_factory=dict)
