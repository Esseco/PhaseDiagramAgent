"""Strict MCP contracts for native Codex consultation and existing memory review."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from phase_agent.agents.contracts import ExpertAdvice, Role
from phase_agent.agents.simulation.contracts import SimulationPlan


class Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Start(Input):
    question: str = Field(min_length=1, max_length=1000)
    consult: list[Role] = Field(
        default_factory=lambda: ["exploration", "reliability"], max_length=2
    )

    @field_validator("consult")
    @classmethod
    def unique(cls, roles):
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate expert role")
        return roles


class Read(Input):
    team_id: str | None = None


class Revision(Input):
    team_id: str
    revision: int = Field(ge=1)


class Reconcile(Revision):
    checkpoint_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_note: str = Field(min_length=1, max_length=4000)


class Advice(Revision):
    advice: ExpertAdvice


class Draft(Revision):
    action: dict
    rationale: str = Field(min_length=1, max_length=4000)
    memory_ids: list[str] = Field(default_factory=list, max_length=10)
    simulation_plan: SimulationPlan | None = None


class Simulation(Revision):
    plan: SimulationPlan


class Feedback(Revision):
    decision: Literal["accept", "revise", "pause"]
    feedback: str = Field(min_length=1, max_length=4000)


class MemoryProposal(Input):
    record: dict


class MemoryReview(Input):
    proposal_id: str = Field(min_length=1)
    approved: bool


_CONTRACTS = {
    "phase_team_start": (
        Start,
        "协调者开始项目内专家咨询，返回原生子Agent任务；不调用项目模型或执行科学动作。",
    ),
    "phase_team_read": (Read, "读取当前或历史咨询、人工意见、草案和状态；恢复原讨论，不重新提交。"),
    "phase_team_reconcile": (
        Reconcile,
        "用原终态检查点及执行收据核销已证明零执行的提案校验失败；仅允许重新修订，不重放、不批准。",
    ),
    "phase_team_advise": (
        Advice,
        "协调者登记原生子Agent返回的结构化只读意见，严格核对角色、事实版本和引用。",
    ),
    "phase_team_draft": (
        Draft,
        "协调者保存路线草案和取舍，等待用户介入；不会创建执行审批或执行任务。",
    ),
    "phase_team_feedback": (
        Feedback,
        "仅转达用户对协调决策的明确意见：修改方向、接受草案或暂缓。接受方向不等于执行批准。",
    ),
    "phase_team_simulate": (
        Simulation,
        "登记原生 Simulation Agent 的结构化执行计划；只绑定当前草案，不执行或批准计算。",
    ),
    "phase_team_propose": (
        Revision,
        "把用户接受的冻结团队草案送入原LangGraph审批流程；不复用方向接受作为执行授权。",
    ),
    "phase_memory_propose": (
        MemoryProposal,
        "把可复用经验送入现有记忆审核队列；不自动激活，不改变科学参数。",
    ),
    "phase_memory_review": (
        MemoryReview,
        "仅在用户明确批准或拒绝指定记忆提案后转达审核；不批准科学动作。",
    ),
}
TOOLS = tuple(
    {"name": name, "description": description, "inputSchema": model.model_json_schema()}
    for name, (model, description) in _CONTRACTS.items()
)
TOOL_NAMES = frozenset(_CONTRACTS)


def call_team_tool(control, name, arguments):
    value = _CONTRACTS[name][0].model_validate(arguments)
    from phase_agent.agents.strategic.codex_team import CodexTeam

    team = CodexTeam(control)
    if name == "phase_team_start":
        return team.start(**value.model_dump())
    if name == "phase_team_read":
        return team.read(value.team_id)
    if name == "phase_team_reconcile":
        return team.reconcile(**value.model_dump())
    if name == "phase_team_advise":
        return team.advise(value.team_id, value.revision, value.advice.model_dump())
    if name == "phase_team_draft":
        return team.draft(
            value.team_id,
            value.revision,
            value.action,
            rationale=value.rationale,
            memory_ids=value.memory_ids,
            simulation_plan=value.simulation_plan.model_dump() if value.simulation_plan else None,
        )
    if name == "phase_team_simulate":
        return team.simulate(value.team_id, value.revision, value.plan.model_dump())
    if name == "phase_team_feedback":
        return team.feedback(value.team_id, value.revision, value.decision, value.feedback)
    if name == "phase_team_propose":
        return team.propose(value.team_id, value.revision)
    # Reuse the same lock as native coordination; reviewed memory changes invalidate
    # the next context digest, never silently approving a stale consultation.
    from phase_agent.agents.strategic.codex_team import team_lock

    with control.chat_handler.lock, team_lock(control.state_path):
        if name == "phase_memory_propose":
            return control.propose_memory(value.record)
        return control.review_memory(value.proposal_id, approved=value.approved)


def native_team_workflow():
    return {
        "backend": "codex_native",
        "strategic": "current Codex; inherit the user's selected model",
        "roles": ["exploration", "reliability", "simulation"],
        "sequence": [
            "If registered results have returned, phase_recover completes the canonical recovery and training handoff BEFORE starting expert consultation",
            "phase_team_read to resume an existing discussion, or phase_team_start for a new decision",
            "Spawn the requested native Codex subagents with the exact returned requests; they return opinions only",
            "Register each returned JSON with phase_team_advise",
            "Compare opinions and retrieved memory; phase_team_draft records one action and rationale",
            "If requests.simulation is returned, delegate to the native Simulation Agent and register its plan with phase_team_simulate",
            "If awaiting_human_input, show the draft and wait for human direction; ready_to_propose may instead reflect an explicit confirmed policy grant",
            "phase_team_feedback conveys only that explicit reply; revise invalidates old advice",
            "After accept, phase_team_propose enters the existing execution approval",
            "Show the pending plan; phase_decide only after explicit approval of that specific action",
            "A completed policy-authorized action must not be approved or replayed again. Inspect its execution_authorization and receipts",
            "When simulation_report.diagnosis_request is present, delegate that read-only diagnosis to native Simulation; return findings to Strategic without calling the project model or retrying",
        ],
        "memory": "Reviewed applicable knowledge is in decision_context.memory_context; propose durable lessons separately, and never auto-approve them.",
        "limitations": "MCP does not spawn agents or authenticate natural-language human intent. Native Codex must perform delegation and faithfully convey user feedback; role instructions do not create a separate sandbox.",
    }
