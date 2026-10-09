"""Graph-visible routing state; invocation dependencies belong in runtime_context."""
from typing import TypedDict, Literal


class SearchWorkflowState(TypedDict, total=False):
    phase: Literal["initialized", "collected", "training_checked", "batches_checked", "analyzed", "wait_checked", "assessed", "acted", "finalized"]
    response: dict


class ToolActionState(TypedDict, total=False):
    phase: Literal["initialized", "refresh_checked", "proposed", "approval_checked", "validated", "executed", "audited"]
    selected_tool: str
    response: dict


class EventGraphState(TypedDict, total=False):
    offset: int
    step_id: str
    response: dict


class ProposalState(TypedDict):
    proposal: dict
