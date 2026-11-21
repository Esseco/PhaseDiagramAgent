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

## 接管首轮采样

Codex 在提出科学方向前读取当前项目配置、`region_sampling_policy`、`initial_coverage`、预算和已有结果。合法相/H/Na 列表只表示可采空间，数量不是采样目标，也不要求穷举；不得自行把少数示例组分替代全范围需求，或把所有合法组合转换成待执行任务。首轮使用 coverage，在允许相和 Na 范围内按需求广泛采样；每个 region 必须给出 sample_count（本轮补充候选数），在 reason 中说明需求依据；全轮合计不超过 1000。程序按获批需求采样后统一去重，不因重复或候选不足自动补足，不将 1000 或固定默认数当作目标。生成后报告实际分布及未覆盖区域。资源阈值是上限而非目标，未采区域不表示收敛。规则、成本预检和现有冻结审批有冲突时停止并核对，不猜测或绕过。

## 一次决策

1. 有已回传结果时，先 `phase_recover` 完成原流程回收及训练交接，再 `phase_context` 读取事实、合法动作与 codex_team_workflow；用 `phase_team_read` 恢复已有讨论。尚无讨论时用 phase_team_start(question, consult) 开始，consult 可以为两个专家、一个专家或空列表，由 Strategic按实际需要选择。已返回的训练结果缺少正式交接时，start 返回 `training_recovery_required`，不创建讨论或浪费专家调用。回收不包含模型激活、项目模型咨询或作业提交；只读和 analysis-only 连接不提供该能力。
2. 返回 requests 中每个角色的完整任务、同一 snapshot_id、记忆与 JSON 契约。Strategic把请求交给对应原生子 Agent。专家只返回意见，不执行项目操作。两位专家可并行。
3. Strategic用 phase_team_advise(team_id, revision, advice) 登记意见。角色、证据、允许的候选动作和事实身份均被检查，重复相同意见幂等；不同内容需要修订讨论。
4. Strategic比较意见并作出取舍，用 phase_team_draft 保存一个 action、rationale 和实际采用的 memory_ids。缺少必要专家时，draft 返回 missing_roles、原因和对应 requests，不保存可执行草案；先登记真实意见再提交草案。新探索需要 Exploration，DFT选点/训练/可靠性需要 Reliability，收敛或结构化跨领域路线需要两者；未知动作保守要求两者。按明确已登记待办清单准备 DFT 输入可由程序完成 Simulation 预检，记录 _simulation_program，不返回模型任务；参数、版本或任务状态不满足固定协议时阻断或保守处理。其他计算动作进入 planning_simulation，返回 requests.simulation。将该请求交给原生 Simulation 子 Agent，用 phase_team_simulate(team_id, revision, plan) 登记输出，再根据确认配置进入 awaiting_human_input 或 ready_to_propose。只读检查/暂停可直接审阅。也可在 draft 时传 simulation_plan；不会创建科学待审方案。
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

未启用讨论的历史 Codex 调用保留原接口；项目模型的新四 Agent LangGraph 路径增加按风险触发的持久方向审阅及 Simulation 计划；旧冻结方案保留历史契约；普通 callable 的提案接口保留，但新的科学选择同样不得绕过必要专家，能力不足会明确阻断。只有显式项目接管的科学决策使用本原生流程，源码开发任务不自动触发科研 Agent。

### 等待同一审批请求

- 发送一次 `phase_decide`，保留 `plan_id`、`state_version`、`proposal_hash` 与客户端请求标识。默认收简明回执，需要审计细节时再查原记录。
- 请求仍在运行时等待同一调用。只有阶段变化、确认完成、明确失败或需要人工操作时才更新用户。需要刷新队列时，把上次的 `queue_revision` 作为 `phase_pending.if_queue_revision` 传回，收到 `unchanged` 后继续等待。
- 超时或响应丢失时，先查原请求、同一方案的持久审批和执行回执。队列中找不到方案只说明当前没有这条待审项。完成须有相匹配的回执，无法核实时保留未知状态，避免重复提案、审批或执行。

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

## 采样规模偏好

空池首轮和较广区域补充应优先评估数百候选，按相、Na区间、扩胞多样性和初态成本说明依据。不能把每区域十几个或全轮几十个作为默认；选择较小规模时应说明有限空间、明确局部需求、已有覆盖或资源约束。后续补充同样按需求广度判断，不机械沿用小批数量。数百不是固定配额或最低要求，1000仍是保护上限。

## Codex 接管的项目模型边界

Codex 接管的离线和 MCP 运行时禁止调用项目模型；已有方案的审批和执行失败诊断也不会回退到 DeepSeek。需要修改待审方案时交回 Codex 重新形成并展示，不由项目模型修订。普通 Agent 对话仍使用其原有模型配置。已运行的 MCP 或服务须由用户重连或重启后才加载新代码。


## 上下文投影与响应读取

普通专家和原生专家复用 evidence.snapshot.role_context。原始 state/ledger 仍是唯一事实权威；snapshot_id 在角色投影前计算，角色视图不创建独立记忆或事实版本。共同身份、事实/模型/配置版本、用户限制、预算、审批、审核记忆和跨阶段风险保留；完整候选指标不按行数或字符数截断。context_projection 记录角色关注字段、严格相等的重复字段去向、略去的其他角色协议和操作契约。不同值或不同 JSON 类型不合并，未知已传入字段保留。

原生咨询从当前 state 复用 build_decision_context 补齐摘要中没有的证据；读取前后校验强 state/control 身份，变化则拒绝本次读取。控制层从当前台账得到的 branch_pool_facts 继续优先。该读取不运行项目模型，但比原摘要读取增加本地解析、派生事实及校验开销；没有新增跨调用事实缓存。

有待办请求时，默认顶层 snapshot 只含身份、snapshot_id 和引用键；每个 requests.<role> 自带完整角色任务、事实、限制与契约，可直接独立交给子 Agent。原生任务包复用现有无损表格编码，所有行、列、公共值、字典和分组索引都在该请求内，evidence_encoding 说明解读方法；不依赖子 Agent 无法读取的外部路径。程序需要原始行时使用 models.transport_tables.expand_evidence_tables。

phase_team_read 新增可选参数：request_role="strategic" 返回完整当前战略事实且不重复返回专家任务；指定 exploration/reliability/simulation 只返回该角色当前仍待办的任务。没有待办任务时，默认响应也包含完整当前战略事实。请求角色已完成时不会制造新任务。include_history=true 返回全部持久事件；默认仅显示当前 revision 的事件元数据，略去与 advice/草案重复的意见和计划正文。event_count 是持久事件总数。磁盘中的历史、旧草案和审批记录不因响应裁剪而删除。

Simulation 的 facts 采用同一自包含契约：身份/版本、当前任务与输入校验事实、预算/预留、未结清审批、审核记忆、人类指导及相关诊断事实；操作目录只含与冻结动作 tool/operation/parameters/stage 相符的契约。新计划必须原样返回 request.context_binding；普通路径绑定 snapshot_id，原生路径另绑定 team_id 和 revision。缺失、错轮或旧事实计划拒绝登记，即使 action_hash 未变。新草案直接提供 simulation_plan 时也需绑定当前讨论快照。已冻结历史计划的恢复仍按原契约校验，不追溯要求新增字段；方向接受和执行批准依旧分离。

升级后的团队事实绑定多了强来源身份，升级前讨论可能显示 stale；使用既有 revise/重新咨询流程，不能删除讨论或复用旧意见绕过。已批准的科学方案仍走原恢复入口。详细核实、验证和字节测量见 [上下文审查记录](../architecture/context-projection-review.md)。


## 原生团队性能与确认响应

phase_team_advise 现在默认 compact=true：只返回绑定本轮身份的登记确认、registered_role、remaining_roles，execution_approved 仍为 false。协调者复用 start/read 已返回的独立任务包及子 Agent 原意见；不要为了确认登记成功重新索取整份事实。缺少某个待办包时用 phase_team_read(request_role="reliability" 等)；需要完整战略事实时 request_role="strategic"；需要完整登记响应时显式 compact=false。Python CodexTeam.advise 默认完整响应保持兼容，持久意见和历史不删。

每次只读咨询优先复用经完整验证的字段索引，在同一请求内复用字段、检查状态和快照；旧格式/索引失效/不兼容访问及未转移初始状态仍完整回退。前后强 state/control 校验保留，没有跨请求完整状态缓存。角色包中的所有科学候选、预算、用户约束、审批边界及未知风险继续保留。Simulation 新计划绑定和旧冻结方案恢复不变。

按实际问题显式选择 consult=[]、一个或两个角色；默认双角色保持兼容。客户端支持时给原生子 Agent 新上下文，只传该独立请求及角色指令，不复制协调者长聊天；能力不可保证时如实报告。不能因此跳过必要专家或事实核验。

详细测量方法、进程内存、耗时、响应字节、来源读取次数与限制见 [原生团队性能审查](../architecture/native-team-performance-review.md)。这些是源码及隔离合成验证，不代表运行中 MCP 已加载新版本，也不意味着真实大项目或模型/客户端延迟已经解决。不得以超时或取消为由自动重放有副作用动作。


## 未接受草案修改、指导替代和缺口澄清

- start 的 question 是共同目标；role_tasks 可按角色指定 question 和 completion_criteria。原生任务与普通团队共用此契约，初次分析保持独立。
- 事实、目标、指导和子问题不变，仅修改尚未接受的草案：phase_team_draft(replace_draft=true)。保留原始专业意见，动作和 Simulation 计划重新绑定；已接受或提交中的草案必须按原流程处理。
- 用户明确替代旧要求时，用 phase_team_feedback(decision="revise", supersedes=[当前 guidance_id])。默认 guidance/effective_guidance 为当前要求；include_history=true 查看全部原文。未明确替代的要求仍有效，不自动解释文本、不升级长期规则。revise 总是清空当前意见及计划，旧子问题归档；可同时传新 role_tasks。
- 专家用 insufficient_evidence 和 evidence_gaps 声明缺口；phase_team_clarify(role, gap_index) 仅允许同快照下每角色每修订一次定向澄清。新结果必须 revise；未知不能靠反复讨论变成事实，也不授权补算。
- operation_readiness 的已注册和已知门槛通过不代表完整输入就绪或获准执行。Simulation 的 checks 是计划文字；程序前检、结构校验、实际执行检查及科学完成各自独立。
- 旧讨论保留原快照编码，显式修订后才启用新指导契约；旧冻结方案不追溯改变规则。细节与假模型流程指标见 docs/architecture/decision-flow-review.md。

## 轻量状态与确认回执

MCP phase_team_read 不带选项时只读取已保存的团队会话，返回当前状态、修订号、剩余角色、草案目标数和已有选点图。stale=null 表示本次没有重新核验科学事实。需要某个角色的任务包时传 request_role。include_details=true 或 include_history=true 走完整视图。Python CodexTeam.read 保留完整行为。

MCP 登记 Simulation 后以及接受方向、暂缓后默认只返回回执。需要完整视图时传 compact=false。修订返回新版本的专家任务包，按新角色任务生成证据绑定。接受方向仍核验当前事实并保持原执行审批门槛。
