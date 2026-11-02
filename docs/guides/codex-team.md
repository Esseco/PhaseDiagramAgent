# Codex 原生专家协作、人工介入与记忆

本入口实现方案 A：当前 Codex 是 Strategic Agent，Codex 原生子 Agent 提供探索、可靠性意见及 Simulation 执行计划/诊断。MCP 负责提供任务和核对意见，不负责创建原生子 Agent，不调用项目配置的 DeepSeek。原有 LangGraph 主图继续负责科学动作的校验、审批、执行、收据和恢复。

## 连接与使用

仍使用仓库的 codex-mcp.cmd。客户端启动命令的参数为：

~~~powershell
& 'E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram\codex-mcp.cmd' --repo-root 'E:\jupyter notebook\1-AL_for_PhaseDiagram\Process_PhaseDiagram' --project-root 'E:\0-PhaseDiagram'
~~~

stdio 命令由 MCP 客户端拉起；单独运行终端进程不会自动连接对话。选择项目后向 Codex 说：

> 以四 Agent 方式接管这个项目。你担任Strategic，按需调用探索、可靠性子 Agent，并将计算草案交给 Simulation 子 Agent 规划。先读取当前讨论和项目记忆；新方案先给我看方向和取舍，允许我补充约束，接受方向后再生成独立待审批方案。现在只讨论，不执行计算。

项目级原生角色定义在 .codex/agents/phase_exploration.toml 与 phase_reliability.toml、phase_simulation.toml；均继承用户当前模型。客户端没有加载自定义角色时，Strategic仍可用原生子 Agent 加载返回的角色请求。若原生子 Agent 能力不可用，应明确报告，不能把单模型扮演多个角色冒充原生协作。

## 严格只读连接

只检查旧项目状态、记忆和待审方案时，MCP 启动参数增加 --read-only，可用 --runtime-config 绑定一个已有项目。该模式直接绑定已保存文件，不初始化或迁移科研运行时，不创建模型客户端，不依赖已启动的项目 HTTP 服务。

只暴露 phase_status、phase_memory、phase_pending 和项目列表/选择/当前身份。服务端拒绝其余调用，包括绘图、提案、审批、回收、创建项目及记忆写入。--analysis-only 是允许派生图表文件的独立模式，不能与 --read-only 同时使用。

完整四角色科研接管仍走下述原入口；只读配置不授权生成或执行新动作。Codex 用户配置还可用 enabled_tools 限定相同工具列表；修改配置后需重连 MCP 或重启客户端加载。

## 一次决策

1. 有已回传结果时，先 `phase_recover` 完成原流程回收及训练交接，再 `phase_context` 读取事实、合法动作与 codex_team_workflow；用 `phase_team_read` 恢复已有讨论。尚无讨论时用 phase_team_start(question, consult) 开始，consult 可以为两个专家、一个专家或空列表，由 Strategic按实际需要选择。已返回的训练结果缺少正式交接时，start 返回 `training_recovery_required`，不创建讨论或浪费专家调用。回收不包含模型激活、项目模型咨询或作业提交；只读和 analysis-only 连接不提供该能力。
2. 返回 requests 中每个角色的完整任务、同一 snapshot_id、记忆与 JSON 契约。Strategic把请求交给对应原生子 Agent。专家只返回意见，不执行项目操作。两位专家可并行。
3. Strategic用 phase_team_advise(team_id, revision, advice) 登记意见。角色、证据、允许的候选动作和事实身份均被检查，重复相同意见幂等；不同内容需要修订讨论。
4. Strategic比较意见并作出取舍，用 phase_team_draft 保存一个 action、rationale 和实际采用的 memory_ids。计算动作先进入 planning_simulation，返回 requests.simulation。将该请求交给原生 Simulation 子 Agent，用 phase_team_simulate(team_id, revision, plan) 登记输出，再根据确认配置进入 awaiting_human_input 或 ready_to_propose。只读检查/暂停可直接审阅。也可在 draft 时传 simulation_plan；不会创建科学待审方案。
5. awaiting_human_input 时向用户说明动作、理由、路线与风险，等待明确意见，不能自行接受自己的方向。只有已确认 workflow.auto_approved_operations 明确允许且没有重大/高成本/敏感风险的动作，程序才可标注 policy_authorized 并进入 ready_to_propose；Codex 不自行增加授权。
6. phase_team_feedback 仅转达用户明确意见：
   - revise：加入本次指导，递增 revision，清除旧意见与草案，用新快照重新咨询。附带新条件的“同意”应按 revise 处理。
   - pause：暂停讨论，不执行、不生成方案。
   - accept：仅接受当前方向，进入 ready_to_propose；不是科学执行批准。
7. phase_team_propose 提交已审阅或确认策略授权的冻结草案，进入原 LangGraph 校验和审批。人工模式返回 pending，核对并明确批准后用 phase_decide 及方案身份字段处理；确认策略允许的普通动作可执行，记录 execution_authorization.source=confirmed_policy，不产生虚假人工反馈。只凭返回状态不能重复执行，须核对精确匹配的持久动作和回执。上传、提交及模型激活仍服从现有边界。

当前 Codex 的原生调度由 Codex 客户端完成。Python 代码不另建主控制器，也不通过 SDK 再启动一套 Codex。工具契约不能鉴别一句自然语言是否确实来自人；Strategic必须忠实转达用户的明确决定。原生子 Agent 继承父会话权限，角色文件的只读指令并不形成额外沙箱。

### 减少协调等待与编码返工

- 复用客户端已经启动的同一个 stdio MCP 进程。服务端按已选项目保留 `CodexOffline` 实例；不要为 context、read、advise、draft 每一步另起 Python、重新构造运行时。没有已连接 MCP 的离线调用可在一个 `CodexOffline` 实例中顺序完成已授权的相关操作。
- 完成回收和训练交接后才固定专家事实快照。两位专家可并行；回收期间不提前发出即将失效的咨询。事实真正变化仍须 revise 和重新核对，不能为加速关闭绑定检查。
- 每次加锁团队操作只构建一份新上下文，校验与响应复用它；下一次操作重新读取校验，未增加跨调用的缓存或放宽配置、模型、记忆失效规则。
- MCP stdin/stdout 均为 UTF-8。人工离线交接 JSON 使用 `Path(...).read_text(encoding="utf-8")`，或在输出时用 `ensure_ascii=True`；避免把中文 JSON 通过采用 GBK 解码的临时管道传入。交接文件登记后删除；不要把整个项目状态复制给专家。
- 候选切换草案缺少 `user_approval_reason` 时，适配器只从当前 revision 的真实 `human_accept` 反馈绑定原因，不从“继续”、旧 revision 或程序自动策略生成。该绑定不改变科学目标，不授予执行权；原独立执行审批和 Simulation 意图校验仍保留。

Strategic 可在 action 中附 research_plan 与 human_review_request；前者只有 active_step 对应本次动作，后续步骤没有执行权。严重冲突与重大决策必须说明审核理由；权限矩阵和确认策略可从 phase_context.decision_context 读取。

## 记忆怎样参与

decision_context.memory_context 是普通项目模型与 Codex 共用的有界投影：
- 当前决策可用的已审核知识，保留适用体系、动作、模型、成熟度和 knowledge_id。
- 短期事实和最近已登记动作状态；历史经历保留版本，不能把待执行收益说成实测。
- 待审核记忆只有数量提示，不作为已生效知识。

Strategic和专家共享这些信息。草案中的 memory_ids 必须确实存在于本轮已检索知识。当前事实、确认配置与权限优先；记忆不是自动执行规则，也不改变科学参数。

人的临时修改保存在当前讨论，默认不晋升为长期规则。值得复用的经验通过 phase_memory_propose(record) 进入现有证据核验和冲突审核；用户明确同意该记忆提案后用 phase_memory_review(proposal_id, approved)。接受策略方向、批准科学动作与批准长期记忆是三件独立的事。phase_memory 查看现有记忆和队列。

执行后的任务、实际结果和费用继续由原有 state_manager / collect_memory_candidates 处理，团队不会另建科学记忆数据库或自动批准经验。团队意见、人工反馈和采用知识随 raw_action._decision_team 保留在原动作记录中。

## 恢复与旧入口

讨论只在 workflow_state/codex_team.json 和 codex_teams/<team_id>.json 保存咨询与人机交互；科学事实仍在原 state/ledger 和检查点。每个项目独立；讨论恢复无需重跑专家。事实、配置、模型或已审核记忆变化后，旧讨论会标记 stale，必须 revise 并重新咨询。

人工修改使用 revision 检查，旧子 Agent 迟到的回复不能混入新讨论。草案接受后不得换 action；活跃讨论期间直接 phase_propose 也不能绕过人工介入。前置校验拒绝可以修订；实际进入工作流前记录 submitting。明确零执行的校验拒绝也允许修订；其他响应不确定时阻止重试，先核对 phase_pending 与原有回执。仅当原动作记录与冻结草案精确匹配时，才能登记已提交并开始后续讨论；不得删除讨论或收据绕过保护。已有待审批科学方案时先处理它，不另起讨论覆盖。

未启用讨论的历史 Codex 调用保留原接口；项目模型的新四 Agent LangGraph 路径增加按风险触发的持久方向审阅及 Simulation 计划；旧冻结方案和普通 callable 保留历史契约。只有显式项目接管的科学决策使用本原生流程，源码开发任务不自动触发科研 Agent。

## 代码与测试

- agents/strategic/codex_team.py：有版本绑定的讨论、人的介入和冻结提交。
- runtime/codex/team.py：严格工具契约和原生工作流说明。
- persistence/memory/decision_context.py：共享记忆投影。
- runtime/local_agent_control.py：复用原提案/审批入口。
- tests/test_codex_native_team.py：失效、修改、角色/引用校验、记忆审核和防重放。
- tests/test_codex_offline_cold.py：旧直接调用与原生团队均走真实应用主图的合成新项目测试。

真实项目服务、模型 API 和超算作业仍由用户按原流程管理，本次开发验证不操作真实科研数据。


## 验证范围

本次变更与测试结果见 [四 Agent 验收](../architecture/four-agents-validation.md)。真实客户端原生调度、真实模型 API 与超算未由合成测试替代。服务和 MCP 由用户明确启动/重连，新代码不会自动重启现有进程。

## Codex 桌面 App 的只读连接验证（2026-11-01）

本机已在 C:/Users/Esseco/.codex/config.toml 增加 phase_diagram，默认 py1、--read-only，并绑定 O3-NiCo。该配置供桌面 App 使用；用户直接在 Codex 对话提出读取请求，不需要打开终端或使用 CLI。配置中的命令由 MCP 客户端在后台启动。当前会话不会保证热加载新增工具；重启 Codex App 后再读取，或按客户端支持的方式重连 MCP。

当前 Codex 担任 Strategic，三个原生角色定义同时放在用户 ~/.codex/agents/ 与仓库 .codex/agents/，继承当前模型。只读连接只供读取；角色指令不会授予执行权，尚未在桌面当前会话现场派生三个角色。完整科学动作须另行明确授权并切换原完整连接。

实际使用配置中的 stdio 命令完成 initialize、工具发现、phase_current_project、phase_status、phase_memory、phase_pending；只读模式只暴露六个查询/项目选择工具。NiCo 返回 epoch0、模型 mace-mh-1、待审方案 0，长期记忆记录与记忆审核队列 0；任务登记 completed=1893、pending=114。实际等待状态为 recovery_blocked（结果回收受阻），未进行回收或修复。

读取前后比较 20,277 个文件的路径/大小/修改时间及 51 个状态、配置和数据库文件的 SHA256，完全一致。该结果证明读取成功，不代表科研流程没有阻塞。未启动真实科研服务、API 或 HPC 作业。

本轮只读/完整/分析三种入口及旧接口合成回归：68 passed（64.74 秒）；ruff 检查通过。后台 CLI 仅用于校验配置是否被识别，不是用户使用方式。桌面窗口自动化工具拒绝操作 ChatGPT 窗口，所以未声称已替用户完成 App 内重连；需用户重启 App 加载。


## 提案校验失败的零执行核销
`phase_team_reconcile(team_id, revision, checkpoint_id, request_id, checkpoint_sha256, evidence_note)` 只处理原流程预算提案校验失败导致的 `submission_uncertain`。程序核对本项目最新根检查点的SHA256、终态、请求身份、精确失败原因及唯一校验失败事件，并拒绝已有科学记录或任何执行收据。成功仅将讨论标为 `validation_rejected`，保留原证据，之后必须 revise；不会重放、批准或执行。未知、非终态、实际执行失败或不匹配的检查点继续阻塞。旧调用缺少派发请求ID时仍要求最新检查点、原失败原因与缺失预算契约三者吻合。预算说明现在在所有Codex动作派发前校验，减少进入主流程后才拒绝的情况。

本次针对核销、身份/摘要/终态/执行事件/执行收据拒绝以及原生团队的合成回归35项通过；没有重启真实服务。
