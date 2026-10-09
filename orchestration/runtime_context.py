"""Invocation-local dependencies, deliberately excluded from graph state."""
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkflowRuntime:
    """One instance per invoke/stream; never share across concurrent runs.

    The legacy callbacks carry managers, clients and business data together.
    Keeping that frame here prevents treating it as checkpointable state.
    Durable resume still requires explicit serializable business state design.
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
