"""One typed contract for non-executing conversation outcomes."""

from copy import deepcopy
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError, Field
from phase_agent.runtime.response_preferences import ACTIONABLE_REPLY_RULE


DIALOGUE_STATUSES = frozenset(
    {"answered", "configuration_view_requested", "configuration_edit_requested"}
)


class DialogueOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["answer", "inspect_configuration", "configure", "advance"]
    answer: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]

    evidence_refs: list[
        Annotated[
            str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=240)
        ]
    ] = Field(default_factory=list, max_length=32)


class ProposalReviewOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["review"]
    plan_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    decision: Literal["approve", "reject"]


def dialogue_output_contract(*, allow_controls=False):
    schema = DialogueOutcome.model_json_schema()
    if not allow_controls:
        schema["properties"]["kind"]["enum"].remove("advance")
    return {
        "schema": schema,
        "reply_guidance": ACTIONABLE_REPLY_RULE,
        "meaning": {
            "answer": "解释、讨论、查询事实；不推进科学执行",
            "inspect_configuration": "只读查看或返回设置；不创建修订草稿、不改变待审批方案",
            "configure": "用户要求修改持久设置；交给配置草稿入口校验",
            **(
                {
                    "advance": "用户告知回传/回收完成或要求推进：进入原流程核对、分析并准备下一步审批，不执行新动作",
                    "review": "用户明确批准或拒绝刚展示的具体方案：引用真实plan_id，复用该方案，不重新规划",
                }
                if allow_controls
                else {}
            ),
        },
        **({"review_schema": ProposalReviewOutcome.model_json_schema()} if allow_controls else {}),
        "controls_available": allow_controls,
        "annotation_policy": "Optional read-only annotations do not route or execute work. Put citations in evidence_refs. Tool, parameters, patch, approval and execution fields cannot accompany dialogue. Only the dedicated review schema may carry plan_id and decision, for the user's explicit decision on a displayed frozen proposal; it cannot authorize a new action. Malformed required fields require correction; the program never infers permission from model prose.",
        "exclusive": "科学动作采用工具契约；不能同时返回对话和动作。对话记忆、继续或配置查看不是执行授权。",
    }


def is_dialogue(action):
    return (
        isinstance(action, dict)
        and isinstance(action.get("kind"), str)
        and action["kind"] in {"answer", "inspect_configuration", "configure", "advance", "review"}
    )


# These fields carry side-effect intent or authority, never optional annotations.
EXECUTION_FIELDS = frozenset(
    {
        "tool",
        "action_type",
        "action",
        "parameters",
        "target_ids",
        "budget",
        "task_key",
        "round_budget_review",
        "post_dft_review",
        "patch",
        "execute",
        "submitted",
        "approval",
        "approved",
        "decision",
        "human_feedback",
        "final_action",
        "state",
        "status",
        "steps_executed",
        "write_requested",
        "config_patch",
        "commands",
        "tool_calls",
    }
)


def normalize_dialogue(action):
    """Quarantine optional annotations only after an explicit read-only kind.

    No inference from user words or model prose; executable/configuration fields
    remain visible to strict validation. Unknown values never enter the state.
    """
    if not is_dialogue(action) or action["kind"] not in {"answer", "inspect_configuration"}:
        return action
    if EXECUTION_FIELDS.intersection(action):
        return action
    known = set(DialogueOutcome.model_fields) | {"_llm_usage"}
    extras = sorted(set(action) - known)
    if not extras:
        return action
    from phase_agent.runtime.turn_process import process_event

    process_event("Dialogue annotations quarantined", {"field_count": len(extras)})
    return {key: deepcopy(value) for key, value in action.items() if key in known}


def dialogue_errors(action, enabled):
    if not enabled:
        return ["dialogue outcome is unavailable for this invocation"]
    action = normalize_dialogue(action)
    try:
        schema = ProposalReviewOutcome if action.get("kind") == "review" else DialogueOutcome
        schema.model_validate({k: v for k, v in action.items() if k != "_llm_usage"})
    except ValidationError as error:
        return [
            "dialogue." + ".".join(map(str, row["loc"])) + ": " + row["msg"]
            for row in error.errors(include_input=False, include_url=False)
        ]
    return []


def dialogue_result(action, state):
    action = normalize_dialogue(action)
    statuses = {
        "answer": "answered",
        "inspect_configuration": "configuration_view_requested",
        "configure": "configuration_edit_requested",
    }
    return {
        "status": statuses[action["kind"]],
        "answer": action["answer"].strip(),
        "evidence_refs": deepcopy(action.get("evidence_refs") or []),
        "state": deepcopy(state),
        "submitted": False,
        "steps_executed": 0,
    }
