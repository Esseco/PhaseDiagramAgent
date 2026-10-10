# Graph 架构与审批恢复

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [Graph 架构与维护入口](#doc-graph_architecture)
- [原生审批恢复](#doc-native_approval_recovery)
- [Batch recovery and lifecycle tracing](#doc-batch_recovery_and_trace)
- [Execution Policy Layer](#doc-execution_policy)

<a id="doc-graph_architecture"></a>

## Graph 架构与维护入口

### 先从哪里读

1. `phase_agent/runtime/local_project_launcher.py`：实际启动入口；`langgraph.json`：Studio 注册入口。
2. `phase_agent/graphs/studio_chat_graph.py`：唯一 Studio 顶层，receive_message → agent → respond；消息防重在 agent 调用边界检查，失败恢复不能重复派发。
3. `phase_agent/graphs/dialogue/graph.py`：展开 agent 后查看读取项目 → 上下文准备 → Agent 决策 → 配置、科研流程或直接答复。审批反馈进入 review，科学动作进入独立 scientific_workflow 节点。
4. `phase_agent/graphs/project/graph.py`：initialize → observe → wait → analyze → react → finalize。
5. `phase_agent/graphs/actions/graph.py`：准备提案 → 审批 → 校验 → 唯一工具 → 审计。

### 顶层阅读与运行边界

Studio 顶层只保留接收消息、Agent 对话流程、输出回复。展开 agent 后直接看到 configure / inspect_configuration / review / scientific_workflow / present 等实际分支；prepare_context 收拢历史恢复和既有控制请求适配，不新增关键词决策器。普通语义仍由 decide 中的 LLM 理解。

scientific_workflow 是真实执行节点，而非展示占位。decide 仅准备本轮科研调用；模型建议复用回调存放在 TurnRuntime，节点执行时临时绑定并在 finally 恢复模型客户端，不写入图 State。科学检查点、审批身份、防重收据及每轮动作限制保持现有规则。

Chat 属于交互界面；回复后结束本轮，新消息才触发下一轮，不建立自动对话回跳。子图登记的是同一运行对象，不维护第二套调度。配置尚未确认时，现有配置会话可直接返回，不保证每次请求都会进入科研子图。

此次更换 Studio 网关节点拓扑，升级后请新建 Studio thread；旧 Studio thread 的未完成网关检查点不能按新拓扑继续。科研项目的配置、任务、审批及科研生命周期检查点不迁移、不删除，既有待审批方案在新会话中仍需先核对绑定。

### 科研调度的唯一推进者

生产科研流程由 project/graph.py 推进；react 直接连接 approved_scientific_action，取消生产路径中的 bounded_action_iteration 包装和循环边。单轮运行入口先验证 max_steps 必须为整数 1，配置、任务或账本写入之前拒绝多步请求。

run_action_turn 复用现有 prepare_event_loop 的方案身份、工具执行和结果入账回调，只执行一次，保存后判断等待状态，再返回项目主图收尾。审批后仍使用原方案，不授予后续动作权限。结果回收与等待由项目主图负责，动作子图只负责本次提案、审批、校验、执行与审计。

低层 run_event_loop / actions/iteration.py 仍有独立测试和调用用途，保留其多步接口；生产 scientific_graph 和 runtime/main 不导入或调用循环构图。它不是聊天或自动科学提交的平行入口。

现有持久项目图的 initialize / observe / wait / analyze / react / finalize 节点与线程身份不变，避免改写历史检查点；节点追踪键 bounded_action_graph 为兼容保留，当前含义是单次动作调用。只改变无检查点动作子图的连接，不删除审批数据库或任务结果。

### 目录职责

| 目录 | 职责 |
| --- | --- |
| phase_agent/graphs/dialogue | 对话图、原始请求传递、会话记忆、审批展示绑定 |
| phase_agent/graphs/project | 科学项目主图及业务节点适配 |
| phase_agent/graphs/actions | 单动作图、审批动作、工具分组及节点（多步循环仅供低层接口） |
| phase_agent/graphs/*recovery*、*handoff* | 结果回收、训练交接等子图 |
| phase_agent/decisions/agent | 模型输入、输出契约及模型返回校验 |
| phase_agent/tools/policy | 权限、预算与审批约束 |
| phase_agent/tools/workflows | 已有科学业务编排适配，供图节点调用 |
| phase_agent/science、phase_agent/analysis、phase_agent/persistence | 科学工具、分析和持久化事实 |
| phase_agent/runtime | 启动、HTTP/Studio、配置交互等传输适配 |

### 一次对话如何运行

普通对话直接带最新事实、确认配置、待审批方案和有限会话记忆进入决策模型。模型可以回答、请求配置修订或提出一个科学动作。配置修订传递原始用户信息；科学动作由程序检查工具范围、参数和预算，再展示并等待批准。批准使用存储方案及其版本/哈希，模型没有授权执行的能力。

已存在方案时，答疑保留方案；修改方案产生新修订，需要重新批准。状态和方案查看、批准和拒绝是显式控制接口。继续不代表批准。

`dialogue/memory.py` 保存按会话隔离且有长度限制的近期记录；记忆只能提供语义上下文，不能授予审批权限。权威状态仍是项目 state、配置快照、预算账本和执行回执。

### 持久化与调试

主图使用新的 `project-scientific-lifecycle-v2` 线程，避免新节点拓扑恢复旧线程。已有配置和科学账本继续沿用。observe 子图继承检查点，以便失败重启不重复完成的回收步骤。动作业务状态及原生审批独立持久化。

`subgraph_visibility.py` 显式登记实际调用的无检查点动作子图，以适配 LangGraph 对 checkpointer=False 子图的自动隐藏。它只登记可见信息，不增加调用或检查点。

`python -m phase_agent.graphs.graph_overview` 根据实际编译图生成流程总览。`phase_agent/runtime/turn_process.py` 保存本轮过程及错误；Studio 看节点跳转，过程页看模型请求、校验失败和执行结果。所有验证使用假模型/假超算，不能据此声称真实 API 或超算已验证。

### 唯一流程入口与已清理代码

聊天 transport 不再继承旧 handler，所有会话进入 dialogue/graph.py。该包拆分为 graph.py（连线）、state.py（状态与依赖）、nodes.py（节点）、history.py（恢复）和 maintenance/（导出、回收决定、文件重建等具名节点）。配置、审批和科学工具继续调用各自业务模块。

project/graph.py 是唯一生命周期构图，project/runner.py 负责检查点运行。已删除旧聊天类、第二套意图路由/问答/配置分类器、重复构图，以及已无生产调用方的 run_pipeline/search_iteration/active_learning_cycle 旧执行链。旧链中的自动阶段选择不再是潜在入口；训练标签测试已迁移到生产使用的 record_dft_products。

旧的架构、对话路由及延迟历史说明已合并到本文件。保留科学工具和独立命令行 step_runner；它们有实际用途，不是第二套聊天决策流程。源码备份仅用于恢复，不参与导入或启动。

### 审批与安全恢复

新会话先展示既有方案，再接受批准。原生回收子图中只读分析失败允许按同一消息恢复；执行节点失败不能借消息重放再次运行。会话网关读取 v2 检查点及嵌套恢复位置，保持执行回执防重。

### 维护与删除规则

应用源码集中在 phase_agent。旧规则评分和阶段选择模块已移除；当前在线动作由 LLM 决策，程序校验权限、参数、版本与预算。保留在用的评分证据和科学算法，不用评分器代替 Agent 选择动作。

决策后端直接调用 graphs/proposal_graph.py；Studio 负责服务生命周期。消息防重使用当前会话网关与执行回执，已删除无调用的旧进程内 guard 和第二套 UI 服务运行器。

先检查静态导入、包导出、动态注册、模块描述符和文档入口，再删除源码。部署前核对哈希并备份到源码目录外；科学数据、配置、任务、结果、模型和检查点不在清理范围。阅读索引见 [文档目录](../README.md)。

<a id="doc-native_approval_recovery"></a>

## 原生审批恢复

### 范围

有state_path的交互工作流在同级目录自动创建langgraph_approvals.sqlite，使用官方SqliteSaver持久化审批子图。
审批通过interrupt暂停，通过Command(resume=...)恢复。状态只含动作ID、配置版本、方案hash、修订号和决定；不保存模型客户端、结构数组或工具执行上下文。
无state_path的调用不创建数据库，仍可使用原无持久化接口。

### 授权与恢复

原执行策略先判断当前人工回复是否合法，然后才调用原生子图。检查点中的批准不会自行成为本轮授权。
旧pending记录仍由业务层定位和校验；首次遇到时建立原生等待检查点，只有本轮明确审批才恢复。审批页仍校验state版本、proposal hash和敏感操作确认。
不同方案hash、配置版本、revision分别创建审批身份；不同工作区有独立数据库。

审批决定交付工具流程前，先持久标记delivered。此后若业务invocations已有完成记录，原幂等逻辑直接返回；若缺少完成记录，再次交付会返回approval_reconciliation_required，要求检查既有文件、任务和台账，不自动再执行。
此策略宁可把“批准后、执行前崩溃”也视为需人工对账，不推测工具是否已执行。它不是分布式exactly-once，也不是任意节点恢复。
文件锁串行保护同一审批数据库；工具执行仍受原运行服务和state/ledger协议约束，不支持多个服务同时写同一业务工作区。

### 文件管理

数据库属于流程状态，不是相图分析输出。等待审批时不要删除、移动或复制它到另一个运行中。
数据库故障会显式报错，不静默退回自动批准；不要通过删数据库绕过delivered保护。
审批子图没有工具节点，不启用工具自动重试。完整科学工作流仍从业务state/ledger回收对账后推进。

实现：phase_agent/graphs/approval_graph.py；桥接：phase_agent/tools/policy/native_approval.py。

### 状态契约与执行收据

生命周期等待门在回收和分析之后、派发新动作之前，读取execution_receipts并与业务invocations对账。收据返回状态必须与已保存业务执行状态一致才视为已记录；其他记录返回恢复清单并阻断新动作。只读检查不会创建数据库，也不自动删除收据或推断任务完成。

phase_agent/graphs/contracts.py严格校验审批身份（动作ID、配置版本、方案hash、修订号）与执行身份（动作ID、配置版本、动作hash、工具）。拒绝空身份、多余字段及类型错误；审批修订号不接受字符串或布尔值。校验失败不会创建检查点。阶段标签有明确类型范围，但尚不等于完整业务状态已可持久化。

公共工具适配器execute_tool_action在有state_path时创建同级execution_receipts.sqlite。只读check_convergence与pause_search除外。该边界也覆盖模型刷新，不仅覆盖人工审批动作。

- started：调用工具前原子领取执行权；同一state路径及动作ID只能领取一次。
- returned：工具返回了成功或失败状态，不证明文件、任务与业务台账已完成入账。
- 相同动作ID改变参数或配置版本仍被拦截，不通过换hash绕过保护。

已有业务完成记录时走原幂等返回；没有完成记录却存在收据时，返回execution_reconciliation_required并阻断本轮动作。工具抛错、进程退出或返回后未入账均不自动重试；需核对已有文件、任务和state/ledger后再决定处理方式。SQLite串行领取不代表支持多个服务同时写业务状态。

两个数据库都属于流程状态：langgraph_approvals.sqlite保存审批交付，execution_receipts.sqlite保存执行边界。不要删除数据库来重试，也不要将收据视为科学结果。双写不是跨文件事务，也不承诺exactly-once。无state_path的低层调用没有持久恢复保护；完整科学图仍不支持任意节点自动恢复。

验证：py1隔离回归841项测试及5项子测试通过；包含并发领取、身份错误、异常/退出后的防重放以及返回后失账保护。未运行实际超算任务。

经核对无产出的中断动作可在 execution_reconciliations 登记 verified_no_effect、完整原始 identity、核对证据及 reviewed_at。recovery_report 仅在身份精确匹配且证据存在时排除该未结项；原收据保留，旧调用仍禁止重放。不得仅因任务为空自动解除阻塞。NaCrO2 已在工作流锁保护下核对两条旧生成动作，台账、任务、生成历史和相关输出为空，备份后登记；未启动计算。18 项收据与显示测试通过。

### 实现入口

[phase_agent/graphs/approval_graph.py](../../phase_agent/graphs/approval_graph.py)。

<a id="doc-batch_recovery_and_trace"></a>

## Batch recovery and lifecycle tracing

The production scientific lifecycle includes discoverable batch_recovery before training_lifecycle and scientific feedback. Existing collectors read returned files and reconcile task results first. The recovery subgraph then independently aggregates Relax, MC, DFT single-point and DFT relaxation statuses in parallel and joins them. It persists task IDs and completed/failed/waiting/waived classifications, not raw structures or live clients. Pending, running, submitted and unknown statuses are waiting unless an existing explicit wait waiver applies.

Native interrupt waits for remaining or partial returns; workflow_state/langgraph_batches.sqlite stores the checkpoint. Resume supplies the newest canonical reconciled statuses. Completed records are retained, new returns update reports, waived DFT waits exit, failed/time-out IDs remain visible for the existing Agent retry proposal and approval. The graph never resubmits or retries a job. Actual execution and file collection remain existing adapters, not new scientific backends. Partial data analysis continues through existing scientific feedback; downstream submission still respects existing manual wait, DFT recovery decision, budget and approval gates. This does not automatically pipeline unfinished batches or waive their dependencies.

workflow_state/node_trace.jsonl is a local append-only lifecycle log with timestamp, node, elapsed time, error type, available wait counts and reported usage (null if absent). Raw prompts/configs/structures are excluded. Proposal-level usage and hybrid analysis timings continue in existing response/action records; the lifecycle trace does not claim an aggregate price or infer missing usage. Keep logs for audit; rotation/retention is not implemented here.

Main scientific graph nodes are timed. The stage summaries themselves have checkpoint history and Studio traces, not separate JSONL timing for every task. Other execution adapters retain their existing idempotency and approval mechanisms. File locks isolate each project's database/log. No HPC computation or automatic submission was performed.

Tests cover real SQLite restart, partial results, submitted/unknown waiting, failed records, explicit waiver, Studio child discovery, existing training checkpoints and scientific lifecycle routing. Actual multi-job HPC end-to-end recovery remains to be validated against returned files.

[phase_agent/graphs/batch_recovery_graph.py](../../phase_agent/graphs/batch_recovery_graph.py)。

<a id="doc-execution_policy"></a>

## Execution Policy Layer

每版 proposal 包含下一轮计算量、分阶段校准成本、总成本和预计剩余预算。DFT 没有
固定配额：Agent 可以建议 0 个或少量代表性任务，其职责是校准/纠正 MLIP，而不是
覆盖大部分候选。

Execution Policy 位于 Agent proposal 与现有 action validation 之间，只控制 action 是否继续，不进行科学计算、合法性校验或预算扣账。

### Interactive

第一次调用只生成并保存 proposal：

```python
proposed = run_tool_step(
    state,
    session,
    registry=registry,
    agent_client=agent_client,
    execution_mode="interactive",
    invocation_id="round-12-action-1",
)
```

返回 `status="awaiting_approval"`，其中 `agent_proposal` 包含状态分析、action、参数、原因、预计成本和预期目的。

使用相同 `invocation_id` 恢复并处理反馈。只有 comment 最后一条非空内容精确为 `approve` 或“同意”才会执行：

```python
approved = run_tool_step(
    proposed["state"],
    session,
    registry=registry,
    execution_mode="interactive",
    human_feedback={"decision": "approve", "comment": "同意"},
    invocation_id="round-12-action-1",
)
```

文字意见不会执行 action，而是交给 Agent 分析并生成下一版 proposal：

```python
human_feedback={
    "decision": "comment",
    "comment": "减少本轮预算，并说明为什么选择这个区域",
}
```

修订可以重复多轮。每一版都保持 `awaiting_approval`，直到明确批准。也可以人工修改后批准：

```python
human_feedback={
    "decision": "modify",
    "comment": "降低本轮预算\napprove",
    "modifications": {"budget": 0.5, "parameters": {"steps": 100}},
}
```

也可传 `{"decision": "reject"}`。修改后的 action 会重新经过完整 validation；interactive 模式不会把非法修改静默替换为其他 action。

### 文件审批

`run_workflow(..., approval_directory="run_state/approvals")` 会按 action 和 revision 写出 `proposal-rNNN.json`、`proposal-rNNN.md` 与可填写的 `decision-rNNN.json`。程序随后正常返回，不占用终端或计算节点。编辑 decision 文件后，用相同 state、`invocation_id` 和 approval directory 重新启动：普通意见生成下一版；最终 comment 为“同意”或 `approve` 时才继续执行。proposal hash、revision 和 ID 必须匹配，防止批准过期提案。

### Autonomous、dry-run 和 replay

- `autonomous`：Policy 自动批准，然后 validation 和执行。
- `dry_run`：Policy 自动批准并执行 validation，但不调用 handler。
- `replay`：读取旧记录的 `final_action`，重新 validation 后执行。

```python
replayed = run_tool_step(
    state,
    session,
    registry=registry,
    execution_mode="replay",
    replay_record=old_record,
)
```

每次 action 在 `state["action_records"]` 保存：

```text
agent_proposal
human_feedback
final_action
execution_result
```

所有获准执行的 action 仍由原 `validate_tool_action()` 检查 config version、冻结参数、预算和 tool permission。

QBC/DFT acquisition 也使用同一 Execution Policy 语义。它在 policy 放行后使用
原有 `validate_dft_agent_decisions()` 完成候选、action 白名单、配置版本、去重和
DFT 预算校验，再进行预算预留与提交；`dry_run` 只校验，不预留也不提交。

[phase_agent/tools/policy/execution_policy.py](../../phase_agent/tools/policy/execution_policy.py)。



### 单一对话结果契约与会话验收

`decisions/agent/dialogue_contract.py`定义唯一强类型对话契约：answer（答疑）、inspect_configuration（只读设置）、configure（修订草稿）。科学动作走工具契约。模型负责语义选择；程序不依据用户措辞给这三项增加关键词路由。当前用户要求单独作为最后一条用户消息发送；持久记忆和历史对话保留为事实上下文，不能压过本轮明确要求。查看不创建草稿，答疑不显示无关等待状态；已有方案不等于所有新话语都是修订或批准。

`tests/test_conversation_contract.py`按完整会话验证提出方案、讨论、设置查看、拒绝、重启恢复、重新提案及一次审批执行。格式正确不是语义正确，真实API还要使用不同措辞复测；执行收据与方案绑定继续由现有审批层负责。

本轮过程保存总耗时、节点耗时、模型调用次数、失败与修正记录，页面可查看。父节点包含子节点时间，不能相加算总时间。普通对话先由模型依据保存的项目事实、配置和会话记忆理解；只读结果直接返回，不进入科学回收或准备。科学动作仍进入原流程校验和审批；科学事实未变化时复用本轮建议，回收结果变化时重新分析并累计模型用量。记录用于判断耗时来自模型、数据回收还是本地分析，建议复用不计为实际模型调用。没有逐节点证据时不宣称某项是卡顿原因，也不增加无界重试。


## 科学判断可见性

流程页“本轮科学判断”只显示推荐、简短理由与待审批状态；备选取舍、局限和引用默认折叠。投影来自唯一待审批方案的 raw_action、post_dft_review、round_budget_review；缺失信息不补造。多个或无待审批方案明确显示暂无唯一方案。动作图过程分别记录准备的科学判断和校验后的 selected_tool；路线选择不代表执行成功，审计结果仍是执行依据。复用既有决策契约，不新增模型调用或授权。py1 10 项针对性测试通过；ruff check及format检查通过。显示变化需现有服务加载新源码。


## 统一等待边界

wait_contract.py集中定义审批、结果、人工输入和中断核对等待。主图条件边、节点等待展示和单次动作状态判断共用；等待时结束本轮动作，主图保存interrupt，恢复从initialize重新读事实。恢复信号不授予执行权限。训练回收后禁止重复训练仍沿用已有校验，相关回归通过。23项等待、事件循环、训练完成保护及SQLite恢复测试通过。尚未集中所有科学分支的进入/完成条件；不将此修复描述为五项全部完成。


## 科学分支前置事实

branch_conditions.py复用已回收训练待评估及模型刷新状态；validate_tool_action执行前再次检查，防止旧update_mlip方案绕过提案侧禁止重复训练保护。模型刷新例外保持原行为（刷新输入、暂停和失败任务重试）。提案上下文及过程记录保存配置版本、活动模型版本与当前限制，不自动选择科学路线。22项相关测试和ruff检查通过。该检查不是完整的所有分支状态机，训练数据新增后重新训练的授权策略仍沿用现有流程。


## 动作后的结果等待与观察

动作结果分类同时检查正式 tasks 记录和旧 pending_tasks；pending、running、submitted、unknown 按 awaiting_task_result 的既有规则等待，明确放弃等待的 DFT 任务保留例外。等待结果时结束本轮，后续回合重新读取项目事实。每次动作结果保存后记录状态、配置及模型版本、事件索引、任务数和等待数；该记录不代表科学收敛。24项任务等待、事件循环和等待边界测试通过，ruff检查通过。尚未将全部科学分支的完成条件集中为统一契约。


待回收任务投影 active_pending_tasks 与 awaiting_task_result 共用判断，包含 submitted/unknown；状态展示、回收投影与动作循环采用相同等待口径。明确放弃等待的 DFT 或刷新任务仍排除，failed/completed 不伪装为等待结果。失败处理及科学分支完成仍由各自业务校验负责。


刷新完成证据检查增加空任务清单、重复任务登记、重复相图记录ID及同任务多相图记录拦截；保留原结果与 waiting_results 状态，不默默覆盖证据，不追加计算。已有版本、任务合格性及相图入账检查继续生效。17项刷新、分支前置条件及动作等待测试通过。


### Studio service shutdown and restart

The launcher and service owner notify the project-local API to interrupt all pending/running Studio runs before terminating the service. Shutdown notification is bounded; forced process termination or loss of the API may skip it. Before installing the project handler on startup, studio_run_lifecycle uses the installed in-memory dev runtime adapter to mark orphan runs interrupted and end busy/interrupted conversation checkpoints with a native END update. No graph nodes are invoked. Completed runs and conversation history remain; scientific tasks, approval databases, execution receipts and HPC jobs are unaffected. Internal dev-runtime incompatibility fails startup explicitly. Old interrupted conversations are ended rather than automatically resumed; send a new message after restart.


### 微调后统一具体动作审批

新生成的非激活方向进入 execution_plan_ready：方向判断只是依据，不先单独要求用户批准。项目主图继续复用已保存的 followup_action，校验并展示完整动作、范围、预算，再通过原生动作审批暂停。用户批准或拒绝该动作时，同一回复记录其绑定的方向决定；审批前重新核对训练回传指纹。原方向与动作均持久化，动作交付仍受执行收据和业务完成记录约束。

成功完成的 followup_action 在候选审阅中记录 followup_result，后续回收不会重新请求同一计划。旧的独立方向审批保持兼容。模型激活审批仅授权版本切换；尚未展示范围与预算的结构刷新属于新的科学动作，仍需审批。


### 审阅展示与修订

review_presentation 只读投影保存的具体方案，输出摘要卡、审批边界和 feedback_history.prior_proposal 的字段差异；不改变动作或审批哈希。控制页修改请求沿用 plan_id/state_version/proposal_hash 校验，仅传入 comment 修订，生成新版待审方案，不批准执行。旧方案与修改意见保留在反馈历史。Studio 状态提供 waiting_state 和 plan_cards，区分人工审阅、任务记录运行、等待回传和可继续分析；不将项目任务状态当作当前聊天或超算的实时状态。


### 只读执行恢复与有限刷新授权

execution_recovery_graph 按执行收据 → 已登记任务/文件/预算账本 → 恢复建议组织核对。它挂在已有 batch_recovery 节点中，不改变持久项目主图拓扑。返回的 report 同时供等待门、聊天和控制状态使用；读文件存在性、大小和小文件哈希，仅从已有记录定位文件，不遍历目录猜测产出。文件存在不能证明整个动作完成，缺失不能证明无副作用，恢复图不运行工具或自动解锁旧收据。

现有结构刷新的一次补充范围保存到 model_refresh.approved_scope：模型、配置、方案校验和、允许目标、数量和成本上限及剩余次数。补充前逐项核对，超范围/旧授权缺失即暂停，要求修订后审批。不授权提交超算。首批刷新接入原生审批；刷新执行身份使用固定动作键，并保存相应业务返回记录，避免回收时把已返回的输入准备视作未入账执行。


初始H生成模板默认 size_step=1，推荐包含矩阵不自动选中；空包含矩阵列表表示无额外周期包含限制，仍保留尺寸和几何检查。显式用户选择不覆盖，旧确认快照不自动迁移。Cr可编辑配置已改为步长1及空选择，需经既有导入确认流程更新活动边界。18项配置与超胞回归测试通过。


配置Agent必须区分用户明确要求与模板默认：未指定奇偶使用步长1，无额外包含约束；明确仅偶数时使用偶数起点和步长2，不自动选周期包含矩阵。审查需列出尺寸、奇偶、包含及几何条件。Cr/Co本次保存独立修订草稿（Cr 4..12、Co 4..16），保留已有确认快照和旧方案，待原配置导入/审核流程确认后用于后续搜索；未执行科学任务。18项相关回归通过。


用户要求新项目初始规模统一3倍：默认候选900、入选branch288、每branch4初态；策略候选总额及配额同步，预算不自动扩大。Cr重置后候选900、branch300、每branch4，旧搜索/审批/配置对话及审核标记不作为活动上下文，历史快照移至项目外备份。13项配置回归通过。


## 统一审阅与服务生命周期（2026-10-10）

`runtime/review_requests.py` 为 HTTP 审批队列、聊天绑定及 Studio 状态提供同一只读审批投影。科学动作继续走已有执行审批；候选模型激活与历史方向继续走对应原生检查点及证据校验；配置确认复用配置快照校验。队列身份与方案哈希校验后才调用各自所有者，不建立第二套科学执行器。模型切换须在本机卡片确认敏感影响，批准只切换版本，刷新与搜索另行审批。配置拒绝保留草稿，修改可用路径值 JSON 或聊天修订。

显式修改保存的训练后动作先重新审阅方向，成功后产生新方向绑定与方案修订；失败保留原方案但不执行。改为模型切换时归档旧动作，等待切换审批。批准/拒绝反馈不作为新的科研判断请求，批准仍复用存储方案。

启动器通过独立 launch.lock 串行启动，服务自身持有 service.lock。服务记录 starting/ready/stopping/stopped/failed；ready 同时要求 Studio 和控制端口就绪，读取旧记录校验进程创建时间、命令中的项目配置及健康状态。失效记录允许精确进程归属恢复；停止可以定位尚未就绪或失去健康的所属服务，必须确认所属子进程退出后才报告成功。

Studio 使用 py1 的 python -m langgraph_cli 直接启动，输出追加到项目 logs/studio_server.log，失败保留退出码与日志路径。既往偶发启动退出的具体根因尚不能由旧日志证明，诊断增强和实机启动验证不代表长期故障已被完全排除。


自然语言配置规则覆盖字段目录内全部可编辑参数；明确值写草稿，建议问题只答复，明确要求推荐并填写时标记Agent建议待确认。字段含义补充超胞、单位、预算及收敛作用范围。不编造远端事实，不加入未注册字段，不自动授权计算。24项配置回归通过；未验证真实LLM多种措辞的语义成功率。

## 执行恢复闭环与持久返回证据（2026-10-10）

`execution_reconciliation.py` 只处理已存在的完整执行身份。`POST /phase/recovery/propose` 先生成待审核的行政核对方案，保存原回执身份、完整业务指纹、文件/任务/预算检查与人工核对依据；它不解除阻塞。审批队列中的 `execution_recovery` 通过原 `/phase/decision` 的 confirm_sensitive 或 reject 处理。批准时在现有生命周期锁内重新读取状态并重查证据，文件、任务、回执或业务状态变化会拒绝旧方案。恢复确认只写一份原子业务 JSON；重复确认返回已处理。

- verified_no_effect：需明确核对无产出、无外部作业或该动作不涉及外部作业，科学计算成本为零。已有任务、工具曾返回但未登记的任务、现存文件、缺失/不可读取的已登记路径、非法预算或已有结算费用均不能通过此方案绕过。批准仅结束旧动作、按原预算模块释放该动作未结算预留，保留原回执和旧方案审计。
- verified_registered_effects：仅复用已完整登记的工作；不从文件存在推定科学完成，不恢复整份旧 state，不提交或重跑任务，预算仍由原结果回收流程结算。未登记返回任务必须先由原任务/结果所有者核对补全，当前功能不自动推断缺失科学字段。

恢复记录保存在 execution_reconciliations 和 recovery_review_history。原调用仍受执行回执保护，解除核对阻塞不授予下一项科学动作权限。未核对执行期间，审批页禁用科学批准，模型激活接口也会阻断。人工外部作业及产出核对是用户明确提供的依据，不是程序自动检查队列的证明。

`execution_receipts.sqlite` 新增 execution_events：claimed 与首次领取同事务，tool_returned 与返回状态同事务，business_saved 在 JSON 写入并 fsync 成功后记录保存文件的 SHA256。返回证据仅保留动作、状态、路径和相关任务引用，避免嵌入完整 state 或结构数组。旧数据库的只读查看不会自动升级或创建文件；下一次写入按增量建表。

日志帮助区分工具返回、业务保存及中断位置，不替代科学结果。仍不承诺 JSON、预算/相图台账与 LangGraph 多数据库间跨文件事务，也不自动修复仅审批交付后、未开始工具执行的所有历史检查点。大文件检查使用大小和修改时间，小于等于 8 MiB 的文件另外核对 SHA256；人工核对仍需覆盖原输出与外部作业。
