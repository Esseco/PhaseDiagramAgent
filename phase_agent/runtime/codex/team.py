"""Strict MCP contracts for native Codex consultation and existing memory review."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from phase_agent.agents.contracts import ExpertAdvice, Role, SpecialistTask
from phase_agent.agents.simulation.contracts import SimulationPlan


class Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Start(Input):
    role_tasks: dict[Role, SpecialistTask] = Field(default_factory=dict)
    question: str = Field(min_length=1, max_length=1000)
    consult: list[Role] | None = Field(default=None, max_length=2)
    human_directed: bool = False
    compact: bool = True

    @field_validator("consult")
    @classmethod
    def unique(cls, roles):
        if roles is not None and len(roles) != len(set(roles)):
            raise ValueError("duplicate expert role")
        return roles


class Read(Input):
    team_id: str | None = None
    request_role: Literal["strategic", "exploration", "reliability", "simulation"] | None = None
    include_history: bool = False
    include_details: bool = False


class Evidence(Input):
    team_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    snapshot_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=25, ge=1, le=100)
    task_ids: list[str] | None = None


class Revision(Input):
    team_id: str
    revision: int = Field(ge=1)


class Propose(Revision):
    compact: bool = True


class Reconcile(Revision):
    checkpoint_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_note: str = Field(min_length=1, max_length=4000)


class RestoreDftAudit(Revision):
    apply: bool = False


class Advice(Revision):
    advice: ExpertAdvice
    compact: bool = True


class Draft(Revision):
    action: dict
    rationale: str = Field(min_length=1, max_length=4000)
    memory_ids: list[str] = Field(default_factory=list, max_length=10)
    simulation_plan: SimulationPlan | None = None
    replace_draft: bool = False
    compact: bool = True


class Simulation(Revision):
    plan: SimulationPlan
    compact: bool = True


class Clarify(Revision):
    role: Role
    gap_index: int = Field(ge=0, le=5)


class Feedback(Revision):
    decision: Literal["accept", "revise", "pause"]
    feedback: str = Field(min_length=1, max_length=4000)
    supersedes: list[str] = Field(default_factory=list, max_length=100)
    role_tasks: dict[Role, SpecialistTask] | None = None
    human_directed: bool | None = None
    compact: bool = True


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
    "phase_team_read": (
        Read,
        "默认只读已保存的团队状态，不重建科学证据。指定 request_role 获取角色请求、include_history 查看历史，或 include_details 获取完整视图。",
    ),
    "phase_team_evidence": (Evidence, "按咨询快照和来源字段分页读取被摘要的原始证据；只读。"),
    "phase_team_reconcile": (
        Reconcile,
        "用原终态检查点及执行收据核销已证明零执行的提案校验失败；仅允许重新修订，不重放、不批准。",
    ),
    "phase_team_restore_dft_audit": (
        RestoreDftAudit,
        "预检或恢复已核销的 DFT 校验失败前的独立抽查名单，不创建或提交计算。",
    ),
    "phase_team_advise": (
        Advice,
        "协调者登记原生子Agent返回的结构化只读意见，严格核对角色、事实版本和引用。",
    ),
    "phase_team_clarify": (
        Clarify,
        "针对已声明缺口进行同版本只读澄清；每角色每修订最多一次，不批准补算。",
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
    "phase_team_plot_dft": (
        Revision,
        "为当前DFT草案在analysis_outputs生成绑定相图的选点图；不执行科学动作。",
    ),
    "phase_team_propose": (
        Propose,
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
        if not (value.include_details or value.request_role or value.include_history):
            return team.read_summary(value.team_id)
        return team.read(**value.model_dump(exclude={"include_details"}))
    if name == "phase_team_evidence":
        return team.evidence(**value.model_dump())
    if name == "phase_team_reconcile":
        return team.reconcile(**value.model_dump())
    if name == "phase_team_restore_dft_audit":
        return team.restore_dft_audit(**value.model_dump())
    if name == "phase_team_advise":
        return team.advise(
            value.team_id, value.revision, value.advice.model_dump(), compact=value.compact
        )
    if name == "phase_team_clarify":
        return team.clarify(**value.model_dump())
    if name == "phase_team_draft":
        return team.draft(
            value.team_id,
            value.revision,
            value.action,
            rationale=value.rationale,
            memory_ids=value.memory_ids,
            replace_draft=value.replace_draft,
            simulation_plan=value.simulation_plan.model_dump() if value.simulation_plan else None,
            compact=value.compact,
        )
    if name == "phase_team_simulate":
        return team.simulate(
            value.team_id, value.revision, value.plan.model_dump(), compact=value.compact
        )
    if name == "phase_team_feedback":
        return team.feedback(
            value.team_id,
            value.revision,
            value.decision,
            value.feedback,
            supersedes=value.supersedes,
            human_directed=value.human_directed,
            compact=value.compact,
            role_tasks={role: task.model_dump() for role, task in value.role_tasks.items()}
            if value.role_tasks is not None
            else None,
        )
    if name == "phase_team_plot_dft":
        return team.plot_dft_draft(value.team_id, value.revision)
    if name == "phase_team_propose":
        return team.propose(value.team_id, value.revision, compact=value.compact)
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
            "For a failed recovery use phase_recovery_diagnostic with the returned batch_id before retrying. Expand only that batch for the log.",
            "Use phase_team_read for saved status only; freshness is unchecked. Request only the required role, or phase_team_start for a new decision",
            "Spawn the requested native Codex subagents with the exact self-contained returned requests and role instructions only; use a fresh child context when supported, rather than copying the coordinator chat. They return opinions only; report client limitations instead of claiming isolation",
            "Use distinct role_tasks with completion_criteria under one question; initial opinions remain independent",
            "For an unchanged-facts unaccepted draft edit use replace_draft=true; never reuse accepted direction or old Simulation plan",
            "Only explicit human revision may supersede active guidance IDs; raw history remains available with include_history=true",
            "For insufficient_evidence with evidence_gaps, phase_team_clarify permits one targeted same-snapshot follow-up per role/revision; new facts require revise",
            "When a user explicitly chooses the direction, use human_directed=true on start or revise and convey the exact scope. Draft against current facts first. Address only required blockers or missing expert evidence and return to the same draft; reopen a choice only when a concrete conflict needs the user.",
            "For an open scientific question choose consult explicitly from [], one role or both. Reuse sufficient evidence in the same revision.",
            "Register each returned JSON with phase_team_advise; preserve its consultation_binding team_id/revision/snapshot_id. Registration returns a compact acknowledgement by default; reuse independently complete tasks already received from start/read, or read one missing role when needed",
            "Requests contain complete role facts; the top snapshot may be metadata only. Use phase_team_read(request_role='strategic') for full coordinator facts or request_role for one pending task; include_history=true reads durable events",
            "Compare opinions and retrieved memory; phase_team_draft records one action and rationale",
            "If requests.simulation is returned, delegate to the native Simulation Agent and register its plan with phase_team_simulate",
            "If awaiting_human_input, show the draft and wait for human direction; ready_to_propose may instead reflect an explicit confirmed policy grant",
            "phase_team_feedback conveys only that explicit reply; revise invalidates old advice",
            "After accept, phase_team_propose enters the existing execution approval",
            "Show the pending plan; phase_decide only after explicit approval of that specific action",
            "A completed policy-authorized action must not be approved or replayed again. Inspect its execution_authorization and receipts",
            "For failed DFT, phase_dft_recalculation_review returns a bound, read-only Simulation diagnosis request for exact task IDs. Human decides whether to retry; if so bind the returned diagnosis and failure_evidence_digest into a dft_recalculation restart_failed_task draft with exact per-task strategies and budget. Register retries only after separate execution approval; input preparation requires a new approval.",
            "When simulation_report.diagnosis_request or dft_failure_reviews[*].diagnosis_request is present, delegate that read-only diagnosis to native Simulation; report findings and limitations without calling the project model or retrying",
        ],
        "memory": "Reviewed applicable knowledge is in decision_context.memory_context; propose durable lessons separately, and never auto-approve them.",
        "limitations": "MCP does not spawn agents or authenticate natural-language human intent. Native Codex must perform delegation and faithfully convey user feedback; role instructions do not create a separate sandbox.",
    }
