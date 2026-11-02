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
            str,
            StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=240),
        ]
    ] = Field(default_factory=list, max_length=32)


class ProposalReviewOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["review"]
    plan_id: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
    decision: Literal["approve", "reject"]


class CancelWaitsOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["cancel_waits"]
    scope_id: Annotated[str, StringConstraints(strict=True, pattern=r"^mc-wait-[0-9a-f]{20}$")]


class PhaseDiagramOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["phase_diagram"]
    operation: Literal["plot", "locate", "export_csv"]
    method: Literal["mlip", "dft", "combined"] = "mlip"
    formats: list[Literal["png", "pdf", "svg"]] = Field(
        default_factory=lambda: ["png"], min_length=1, max_length=3
    )


class PhaseQualityReviewOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["phase_quality_review"]
    decision: Literal["remove", "retain", "restore"]
    review_ids: list[Annotated[str, StringConstraints(pattern=r"^pq-[0-9a-f]{24}$")]] = Field(
        min_length=1, max_length=100
    )


class PhaseQualityPolicyOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["phase_quality_policy"]
    cutoffs: dict[
        Literal["h_cutoff", "d_cutoff"],
        Annotated[float, Field(gt=0, allow_inf_nan=False)],
    ] = Field(default_factory=dict, max_length=2)
    source: Literal["human", "agent"]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
    release_manual: list[Literal["h_cutoff", "d_cutoff"]] = Field(
        default_factory=list, max_length=2
    )


class AdvanceOutcome(DialogueOutcome):
    """Advance delegates to observation; an unused explanation is optional."""

    kind: Literal["advance"]
    answer: Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)] = (
        "读取最新事实并准备下一步方案。"
    )


class EvidenceRequestOutcome(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    kind: Literal["inspect_evidence"]
    topics: list[
        Literal[
            "mc",
            "dft",
            "tasks",
            "convergence",
            "budget",
            "history",
            "model",
            "configuration",
            "plans",
        ]
    ] = Field(min_length=1, max_length=9)


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
                    "phase_diagram": "请求绘图（含口语和近义表达）用operation=plot；明确只导出CSV/数据表用operation=export_csv；要求绘制当前相图用operation=plot，直接本地绘图并返回实际文件；询问相图文件或文件夹在哪用operation=locate，只读查询。默认method=mlip，明确指定DFT或综合才改为dft/combined。默认300 dpi且仅PNG，formats省略或为PNG列表；只有用户明确索要PDF、SVG时才在formats加入相应格式，不能自行附送。只用已有已验证数据，不计算或推进搜索；数据不足报告原因，不用answer假称已绘图。",
                    "phase_quality_review": "用户明确对当前相图审查编号选择删除用remove；暂不处理/不管并继续选DFT用retain。编号以phase_diagram_quality.review_groups.number为准，选择该项完整review_ids；指定具体结构才引用review_targets的精确review_id，不能混用内部结构序号。问题或泛指继续不能授权删除，目标不清先问。恢复用restore和restoration_targets原review_ids，保留审计并校验备份和边界。retain不宣称正常；新计算仍走原审批。",
                    "phase_quality_policy": "只调两个参数：h_cutoff和d_cutoff，单位meV/O2；H>100直接移除固定。用户明确给值用source=human；授权agent判断时可用source=agent并说明依据，不能覆盖人工指定值或伪称human。明确交回agent调节可用release_manual。询问现值用answer。程序自动处理版本和审计，本轮只保存参数，不删除或提交计算。",
                    "cancel_waits": "用户明确取消一组当前MC的本地等待并继续：只能引用task_wait_cancellation_scopes中的精确scope_id；不取消远端作业、不结算未知费用、不标成功。问题、泛指取消或仅说继续不能使用；目标不清先询问。取消后进入原流程准备下一步方案，仍须审批。",
                }
                if allow_controls
                else {}
            ),
        },
        **(
            {
                "review_schema": ProposalReviewOutcome.model_json_schema(),
                "advance_schema": AdvanceOutcome.model_json_schema(),
                "evidence_schema": EvidenceRequestOutcome.model_json_schema(),
                "cancel_waits_schema": CancelWaitsOutcome.model_json_schema(),
                "phase_diagram_schema": PhaseDiagramOutcome.model_json_schema(),
                "phase_quality_review_schema": PhaseQualityReviewOutcome.model_json_schema(),
                "phase_quality_policy_schema": PhaseQualityPolicyOutcome.model_json_schema(),
            }
            if allow_controls
            else {}
        ),
        "controls_available": allow_controls,
        "annotation_policy": "Optional read-only annotations do not route or execute work. Put citations in evidence_refs. Tool, parameters, patch, approval and execution fields cannot accompany dialogue. Only the dedicated review schema may carry plan_id and decision, for the user's explicit decision on a displayed frozen proposal; it cannot authorize a new action. Malformed required fields require correction; the program never infers permission from model prose.",
        "exclusive": "科学动作采用工具契约；不能同时返回对话和动作。对话记忆、继续或配置查看不是执行授权。",
    }


def is_dialogue(action):
    return (
        isinstance(action, dict)
        and isinstance(action.get("kind"), str)
        and action["kind"]
        in {
            "answer",
            "inspect_configuration",
            "configure",
            "advance",
            "review",
            "inspect_evidence",
            "cancel_waits",
            "phase_diagram",
            "phase_quality_review",
            "phase_quality_policy",
        }
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
    if not is_dialogue(action):
        return action
    # Transport tracing never carries dialogue authority, including advance/review.
    action = {
        key: value
        for key, value in action.items()
        if key not in {"_proposal_harness", "_analysis_route"}
    }
    type_tag = action.get("type")
    if isinstance(type_tag, str) and type_tag in {"dialogue", action["kind"]}:
        action = {key: value for key, value in action.items() if key != "type"}
    if action["kind"] not in {
        "answer",
        "inspect_configuration",
        "inspect_evidence",
    }:
        return action
    if EXECUTION_FIELDS.intersection(action):
        return action
    known = set(
        EvidenceRequestOutcome.model_fields
        if action["kind"] == "inspect_evidence"
        else DialogueOutcome.model_fields
    ) | {"_llm_usage"}
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
        schema = (
            PhaseQualityPolicyOutcome
            if action.get("kind") == "phase_quality_policy"
            else PhaseQualityReviewOutcome
            if action.get("kind") == "phase_quality_review"
            else PhaseDiagramOutcome
            if action.get("kind") == "phase_diagram"
            else CancelWaitsOutcome
            if action.get("kind") == "cancel_waits"
            else EvidenceRequestOutcome
            if action.get("kind") == "inspect_evidence"
            else ProposalReviewOutcome
            if action.get("kind") == "review"
            else AdvanceOutcome
            if action.get("kind") == "advance"
            else DialogueOutcome
        )
        schema.model_validate({k: v for k, v in action.items() if k != "_llm_usage"})
    except ValidationError as error:
        return [
            "dialogue." + ".".join(map(str, row["loc"])) + ": " + row["msg"]
            for row in error.errors(include_input=False, include_url=False)
        ]
    return []


def repair_read_only_dialogue(client, payload, action):
    """One format correction on failure; never widen a read-only reply's authority."""
    errors = dialogue_errors(action, payload.get("unified_dialogue") is True)
    if (
        payload.get("unified_dialogue") is not True
        or not errors
        or action.get("kind")
        not in {
            "answer",
            "inspect_configuration",
            "inspect_evidence",
        }
    ):
        return action
    corrected = client(
        {
            **payload,
            "mode": "dialogue_contract_repair",
            "allowed_tools": [],
            "invalid_dialogue": deepcopy(action),
            "validation_errors": errors,
            "instruction": "Correct the JSON format once. The reply must remain a dialogue of the same kind. No scientific tools, controls, configuration changes or execution authority may be added. Return the complete corrected read-only response.",
        }
    )
    corrected = normalize_dialogue(corrected)
    if not isinstance(corrected, dict) or corrected.get("kind") != action["kind"]:
        raise ValueError("correction must remain a dialogue of the same kind")
    return corrected


def dialogue_result(action, state):
    if action.get("kind") == "inspect_evidence":
        return {
            "status": "evidence_requested",
            "topics": list(dict.fromkeys(action["topics"])),
            "state": state,
            "submitted": False,
            "steps_executed": 0,
        }
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
