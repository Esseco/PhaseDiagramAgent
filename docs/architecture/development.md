# 源码阅读与开发约定

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [源码阅读与修改指南](#doc-source_guide)

<a id="doc-source_guide"></a>

## 源码阅读与修改指南

### 先找到实际流程

从 `langgraph.json` 的注册入口开始，依次阅读 `graphs/studio_chat_graph.py`、`graphs/dialogue/graph.py` 和 `graphs/project/graph.py`。单次科学动作沿 `graphs/actions/graph.py` 进入注册工具。graph.py 看连线，nodes.py 看业务调用，state.py 看序列化事实；模型、manager、锁和回调看 Runtime context。

### 按职责定位

配置问题读 configuration；模型语义与输出错误读 decisions/agent；审批、预算、准备与回执读 tools；证据与相图读 analysis；算法和远端计算读 science；状态和记忆读 persistence；启动及服务读 runtime。HTTP/API 仅是传输接口，不能另建决策循环。详细位置见各包 README。

### 扩展模块的最短路径

| 要改什么 | 所属位置 | 必须保持的边界 |
| --- | --- | --- |
| 新项目、文件路径、离线诊断 | `runtime/local_project_launcher.py`、`runtime/project_diagnostics.py` | 项目数据独立；诊断只读，坏文件分段报告 |
| 新参数或结构自由度 | `configuration/schema`、`configuration/defaults` | 返回字段错误；不替用户猜测科学边界 |
| 对话意图、配置查看、审批衔接 | `graphs/dialogue/graph.py`、`nodes.py`、`saved_control.py` | 路由接入已有图；查看不修订、不执行 |
| 回传→分析→下一份方案 | `graphs/project/graph.py` 与相应节点 | 先观察实际证据；每轮最多执行一个已批准动作 |
| 增加科学动作 | `graphs/actions`、`tools/dispatch` 现有注册入口 | 复用契约、预算、版本和回执；不另建执行循环 |
| 常见文件故障隔离 | `tools/remote/batch_runner.py`、`integrity.py` | 单任务失败不抛弃整批；校验读取的同一份字节 |
| 会话连续性、已批准知识 | `persistence/memory/` | 项目/线程隔离；历史对话不能授权，模型版本过滤保留 |

图中只存可序列化事实，模型/锁/回调放 Runtime context。新的故障恢复路径先确定副作用归属：文件读取可重新尝试，未知执行必须走原回执核对。`node_trace` 是可选观测记录，不能成为科学操作成功与否的裁决者。

### 旧数据兼容

已保存的旧模块描述符通过 module_references.py 转换，旧科学数据有显式读取/迁移工具。保留数据兼容不等于保留旧执行框架。Pydantic 只验证契约；可选 Deep Agents/混合客户端只产生分析提案，仍进入同一个 LangGraph 校验与审批路径。

### 写给后续维护者

函数用明确的输入与返回值，说明科学单位、未知状态和副作用。拆分长表达式和复合语句，保持模块职责；不要为了减少行数隐藏决策或审批。更改副作用之前核对执行回执和可重放边界，禁止在恢复时重新解释旧批准。

静态检查与格式：`python -m ruff check phase_agent`、`python -m ruff format --check phase_agent`。默认使用 py1；回归使用假模型/假超算。源代码改动不能证明真实 API 或超算已验证。源码部署先核对基线并外置备份，不修改科学项目的数据和检查点。


### Generation and approval invariants

`decisions/agent/generation_plan.py` projects enabled strategies from explicit lists and boundary facts. Both proposal context and tool policy use it, including older unsynchronized snapshots. A single legal phase excludes competing_phase; a single TM species excludes tm_ordering. Never redistribute an explicit allocation to bypass a rejected constraint.

`tools/dispatch/execute_tool_action.py` supplies a deep copy of the approved action to the handler. Handler annotations cannot mutate the stored proposal or its receipt identity. A returned or failed execution receipt does not authorize replay. `reserved` means an occupied task key, not proof of submission or completion. Pending-plan advice must respect that persisted blocker.

Tests: `tests/test_generation_runtime_guards.py` uses fake tools and temporary receipts. It does not establish real API/HPC behavior. The current project's existing reservations still require file/receipt reconciliation before release. Parent-dependent strategies in an empty ledger and exhaustive phase/Na/H combinations remain separate integration cases to audit; these guards do not claim to validate every scientific generation path.

### Define the scientific scope before planning

New project files leave boundary.P and boundary.TM_ratio empty and the T role unset. Template examples are not scientific choices. Both short and full editable templates behave this way; existing project files and confirmed snapshots are preserved. Import checks the problem definition before H enumeration. Startup confirmation rejects missing boundaries and invalid variable roles.

Use system.configuration_space.roles for H/P/x/T/N. A binary TM ratio does not imply mutable ordering. Fixed T uses fixed_T_source=phase_reference and preserves ordered reference occupancy while permitting atomic relaxation. If a newly configured single-species system leaves T unset, synchronization resolves it to fixed reference occupancy because no exchange degree of freedom exists. It does not remove physical TM interactions.

The shared configured_generation_strategies projection excludes tm_ordering for fixed T, competing_phase for fixed P or one phase, composition for fixed x, and periodic_extension for fixed H. Model context, transport, proposal validation, dispatch policy and generator use these facts; explicit conflicting allocations are rejected, never silently zeroed. Configuration review prints the actual boundary, TM ratio, roles, fixed values and legal strategies before confirmation. Semantics are interpreted by the LLM into this existing schema, not a keyword classifier.

Tests: tests/test_scientific_scope.py and tests/test_branch_proposals.py cover unresolved initialization, preservation of explicit boundaries, single/binary TM combinations, fixed-variable legality, transport and reference occupancy. Live LLM interpretation and real HPC execution are separate validations; these tests use synthetic data and fake dependencies.

### Read-only dialogue resilience

The dialogue kind and nonempty answer remain required. Optional annotations on explicit answer/inspect_configuration outcomes are quarantined before strict validation; only typed evidence_refs is retained. The process trace records the quarantined field count without persisting arbitrary model values. Annotations never affect routing or grant authority. Tool/action, parameters, budget, patch, approval, state, commands and execution fields are rejected even when empty. configure outcomes remain strict and enter the existing configuration validator.

Malformed dialogue receives at most one model correction in post_dft_review.request_validated_action, with original context and schema errors. A correction must remain dialogue and cannot turn into a scientific tool action; failure reports an error without execution. Both model calls contribute to usage. The program does not infer intent from user keywords or invent an answer when correction fails.

Tests: test_dialogue_resilience.py plus conversation_contract, unified_react_dialogue and dialogue_evidence cover complete read-only exchanges, side-effect isolation, bounded repair and usage. Scientific proposal regression tests ensure these changes do not soften execution contracts.

### Contextual concise interaction

fresh_turn_context supplies previous_reply from thread-scoped memory and bounded pending plan identities/statuses including task-key reservations. Historical replies are conversational evidence only, not proof of execution. decide_from_saved_facts passes an explicit concise interaction policy alongside original user input: resolve short follow-ups from context, explain failures without retrying, and never treat continuation as approval. Multiple pending plans permit answers and configuration viewing; scientific progression still requires selection of a plan. No keyword intent classifier was added.

Tests: test_interaction_context.py checks the actual production dialogue path, context delivery, no scientific call on questions and preservation of pending plans. These fake-model tests establish transport and execution boundaries, not live LLM answer quality.


### Failed tool reservation lifecycle

Generation allocation consistency is validated before reservation and dispatch. A handler exception changes the effective decision to reconciliation_required and annotates the still-reserved budget, keeping the budget reserved until artifacts are checked. Never infer no effects from an exception. A reviewed no-effect failure is settled with zero cost and marked failed/verified_no_effect; a fresh approved invocation may then reuse its task key, archiving the previous reservation. Original execution receipts and approvals remain claimed.


### Read-only execution consistency gate

The existing lifecycle recovery gate also checks execution_state_issues before proposing new science, even without a receipts database. It reports failed_action_still_reserved, handler_failed_effects_unknown and verified_failure_not_settled. Checks do not modify state or release budget. Matching returned receipts alone cannot override contradictory reservations. Active reserved actions without failure evidence remain valid; reviewed settled failures are clear. Replay prevention remains attached to the original invocation.


### Concise user-facing proposals

proposal_presentation provides task labels and brief generation summaries from the stored allocations. Brief replies distinguish candidate quota, dedup selection limit and initial states; retain phase/Na/H constraints and approval scope. Future cost scenarios and scientific rationale remain in detailed mode and saved proposal files. format_epoch_reply removes only an identical repeated stage line. Presentation never changes task parameters or grants approval.


### Agent ownership and recorded feedback

Studio uses react_proposal; other interactive initial selections use the LLM whenever present, bypassing offline next-action selection. Explicit lower-level revision/calculation APIs remain compatible. Legacy offline helpers remain available without a model. Stale approvals are rejected rather than transferred. decision_summary stores optional bounded public observation, tradeoffs and expected outcome in the existing proposal envelope; missing/malformed optional explanation does not invalidate execution parameters. decision_feedback supplies task-key-matched outcomes and measured costs in the next decision context, with model/config provenance and unknown values retained. Model statements are unverified explanations, never authority or automatic long-term knowledge. Input preparation is not proof that a calculation completed. Each production turn still has at most one approved action.


### Relax before MC allocation

For generated projects, mc_entry_conditions is shared between model context, bounded proposal repair and dispatch validation. MC allocation/mc_inputs requires a nonempty current-model Relax pool and completed versioned MLIP hull. Relax input preparation is an independent approval; upload/submission and recovery precede the MC decision. The prerequisite blocks invalid actions without code-selecting a replacement. Imported legacy state without generation lineage retains existing backend validation.


### Proactive proposal after generation

After successful generation with registered structures, lifecycle action completion prepares at most one further proposal using the existing interactive approval interface. A deterministic :next-proposal invocation carries no human feedback or replay record. It never authorizes another action. Generation results remain visible if proposal preparation fails; new pending approval is rebound for the conversation. Result recovery already precedes analysis and the next decision in the project graph; completion reports must enter that workflow rather than be treated as verified completion or answer-only guidance. This does not poll HPC or run without a new user/recovery event.


### Project scaffold and input admission ownership

`configuration/runtime/project_template.py` owns runtime defaults, bootstrap hints and the thin data-project entry. `configuration/session/project_scaffold.py` stages and reads back a new project; the launcher registers it only after publication. Do not copy application code into each data workspace.

`tools/remote/input_validation.py` checks final upload bytes and keeps bounded, content-addressed process caches. `submission_handoff` and `relax_mc_batch/handoff` enforce it inside existing approved preparation; no second scheduler or extra scientific action is introduced. `environment_preflight.py` is embedded in standalone compute entries, uses module discovery and Bash parsing without importing model frameworks, and runs before child submission. Successful state facts contain only status, validator version, fingerprint and remote-check scope; presentation must not re-read ZIP files. See [recovery guide](../guides/recovery.md) for commands, limits and recovery semantics.
