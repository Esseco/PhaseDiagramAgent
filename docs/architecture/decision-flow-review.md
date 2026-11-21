# 决策流程、上下文与角色交互审查（2026-11-10）

## 范围与结论

以任务开始时的磁盘工作树为基线，先核实近期上下文投影、字段索引、单次字段复用及精简意见确认均已存在，未重复实现。没有使用 Git HEAD 回退或作为修改前实现，没有新调度器、记忆库、管理 Agent、模型切换或子 Agent。未读取真实科研状态、运行真实科研模型、部署、重启服务或操作 HPC。

收益优先级：同一事实下的未接受草案修改 > 有效指导与历史分离 > 不同角色子问题 > 已知不可行路线提前拒绝 > 有界缺口澄清和检查语义。各项均沿现有模块实现。编码前已将计划保存到 TEMP_PLAN.md；本页归档实施记录。交付检查时本任务标记段已不存在，根目录现存其他任务计划保持原样。

## 九条线索核实

|线索|核实与处理|
|---|---|
|整体绑定、revise 全部重做|确实绑定强来源身份，revise 清空意见和草案。这是安全保护，并非整体缺陷。新增 replace_draft，只为事实、共同目标、指导和任务未变化且尚未接受的草案修改复用原始意见；不重标版本。|
|共同 question|原有角色提示和权限已经不同，但没有显式子问题与完成条件。ConsultationPlan 新增可选 role_tasks，共同 question 保留为 decision_goal；缺省也生成各角色不同任务。两条路径共用 specialist_task。|
|缺证据|已有 insufficient_evidence；没有可替换意见的有界澄清入口。新增 evidence_gaps 和 phase_team_clarify，只能选择该专家已声明的缺口，每角色每 revision 一次，仍用同一快照。|
|guidance 累加|原文永久追加并全部拼接进快照。新增显式 supersedes ID、来源、team 范围与 revision；当前视图仅给有效要求，原文和替代关系仍持久保存。没有文本关键词推断替代关系，也不提升为长期规则或审批。|
|已注册与可执行混淆|操作目录只证明契约注册/工具范围，有 registry 时额外过滤 handler；不证明具体输入、依赖、环境和权限已就绪。新增分维只读 operation_readiness，仅判定已知 MC 依赖和已有禁用的新恢复路线，其余保留未知。|
|Simulation checks|validate_plan 校验结构、顺序、动作和上下文绑定；字符串 checks 不执行。新增程序前检、计划检查、执行观察和科学完成的明确分离；诊断补充已有返回中的 error/reason/validation_errors。另复现模型可在新 action 私有字段中夹带未校验 _simulation_plan，使展示看似已校验；现拒绝该入口，使用已有 simulate 或经校验的 simulation_plan 参数。|
|阶段/数值规则重复|Strategic/Reliability 的恢复规则确有重复，同时包含历史周期兼容语义。缺少按版本生成提示的完整证明；本次不集中改写这些科学规则，避免历史规则被当前规则覆盖。|
|报告级引用|原先 ExpertOption 无逐项证据字段，引用主要在整份意见。新增可选 evidence_refs、counterevidence_refs、assumptions、cost_unit；两类引用都对当前快照校验。旧意见仍可读取，缺字段保持未知，不补造依据或单位。|
|回收重复|已有候选指纹、单项派生缓存、backlog/processed/feedback 检查及无新结果快路径；仍有必要的账本/导航核对与错误清理。未确认值得修改的科学重复入账缺陷；保留实现并运行无变化/新结果/中断恢复测试。|

## 行为边界与接口

### 专业意见、规划、授权

- phase_team_draft 增加 replace_draft=false。只有 planning_simulation 或 awaiting_human_input 的未接受草案可显式替换；须通过原 _fresh 强来源核验。consulting 首次起草行为兼容。
- 保留原 advice 的角色、snapshot_id、team/revision 范围和原事件；不会改成“新专家意见”。源状态、模型、配置、结果、预算、输入、权限或指导变化仍保守失效。没有跨来源版本的依赖选择性复用。
- 新草案不继承旧 Simulation 计划或 direction_review；变更动作后的旧计划因 action_hash 不符拒绝。已接受/策略确认、提交中、提交不确定和已提交状态不能用 replace_draft 改写。新的方向和执行审批仍沿既有流程。
- Simulation 计划可用已有 simulation_plan 参数内联校验；原 action 的 _simulation* 私有字段不作为新计划登记入口。历史冻结方案仍使用原 validate_frozen_plan 规则。

### 指导、子问题与旧讨论

- phase_team_feedback(revise) 可传 supersedes=[guidance_id]，只允许替代当前有效 ID；未传则保留既有累加语义。反馈原文完整写入原 guidance 和事件，结构化 guidance_records 记录来源、范围与替代关系。
- 默认 read 返回的 guidance 是当前有效文本，effective_guidance 含来源；include_history=true 的 guidance_history 和持久事件可读取全部原文。磁盘 guidance 原历史列表不删改。旧原文进入提示前只按明确替代关系过滤。
- 显式 revise 一律清空当前意见、草案、方向审阅和澄清状态；旧角色子问题保存在事件中，当前任务默认重建，也可显式传新 role_tasks。普通图的专家仍保持独立初次分析，不互传初次意见。
- 新讨论标记 decision_contract_version=2，将当前指导及 role_tasks 纳入快照。缺标记的旧讨论保留原 codex_consultation 编码和快照身份；只在显式 revise、清空旧意见后进入新契约。读取不会迁移科学状态或重标旧意见。

### 有界澄清

- 只有 consulting 中已咨询角色、已有 insufficient_evidence、有效 evidence_gaps 索引，才可请求一次澄清。该任务包含自己的原意见、明确缺口、原绑定及完整同版本角色事实。
- 不自动调用模型、不重新咨询另一个角色、不补算、不执行工具、不授予审批。登记后原意见保留在事件中；待澄清状态未完成时不得起草。精简确认也会列出其他待澄清角色。
- 若现有完整事实不足，仍返回 insufficient_evidence；需要新测量或新增事实必须走 revise。这里是同证据澄清，不是自动证据采集器。普通 Agent 既有图仍每角色一次，没有新增自动循环；它共享 gaps/task/option 契约，但没有复制原生讨论的交互流程。

### 确定性可行性与检查语义

- 原生咨询、普通角色请求及 Simulation 任务提供 operation_readiness。known_prerequisites_passed 仅表示已知 MC 门槛通过，绝不等于完整输入检查或科学可执行。
- 明确区分 registered、project_readiness（未知/等待依赖/新提案禁用）、implementation、input_readiness 和 authorization；不会把未授权误报为功能不可用。
- 原生 draft 和普通 proposal_errors 共用 prerequisite_errors；MC 输入/分配缺少已回收 Relax 与当前模型凸包时，在保存伪执行计划前拒绝。新提案的停用恢复操作及其 mode 别名共用原执行策略规则；历史冻结校验不追溯改规则。
- Simulation program_preflight 明列运行过的 action_contract_errors/prerequisite_errors 及其有限范围。validate_plan 通过只证明计划结构/动作/绑定，不证明 checks 文本已经执行。团队提交中或提交不确定时执行检查为 unknown，不能据此认定零执行。
- inspect_execution 继续只把工具状态和确定性返回作为观察；计划文字检查标为 not_evaluated，scientific_completion_claimed 始终 false。失败不自动重试、不生成任意执行逻辑。

## 可复现流程指标

tests/benchmark_decision_flow.py 使用最小 make_control 合成项目，state.json 为 {}（2 bytes），两份假专家意见，无真实模型。每轮在 TemporaryDirectory 中完整结束，重复 5 次。修改前先执行同一基准，修改后再执行；没有从 HEAD 加载旧代码。

场景：已咨询两位专家、已有尚未接受的 pause_search 草案，仅需改善解释，科学目标、证据、预算与权限不变。旧入口只能 revise → 两位重新咨询 → 起草；新入口 replace_draft。旧分支为触发原机制增加解释性 guidance，新分支不把解释文字升级为用户科研约束；这是同等流程任务比较，不是相同网络数据包或逐字相同会话。

|指标（每轮）|修改前|修改后|
|---|---:|---:|
|逻辑团队工具往返|8|5|
|假专家分析次数|4|2|
|重新咨询次数|2|0|
|缺证据澄清轮数|0|0|
|首次合法草案校验通过|5/5|5/5|
|5轮墙钟中位数（秒）|0.0696524|0.0447455|
|首次含模块预热（秒）|1.4235092|1.4097469|
|每轮累计响应 UTF-8 bytes|35890|24115|
|真实模型调用|0|0|
|失效原因|revision 清空全部意见|无需失效：同一来源、目标与指导|

“工具往返”按团队工具对应 Python 入口每调用一次计一次，未启动真实 MCP/stdin/网络客户端，未测网络等待。响应是 len(json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")) 的累计业务对象大小，含 schema 和新边界元数据；不是 token、provider 编码或费用。计时用 perf_counter，py1，BLAS/OMP/MKL 均为1线程；前后各自首轮预热已单列。tiny fixture 的耗时不外推到真实大项目。

新增负向/边界场景另记录：
- 同一快照缺口：初次分析1次 + 显式澄清最多1次；第二次澄清拒绝，真实模型仍0。
- 新事实、模型/配置/预算/输入/结果/目标/权限改变：拒绝旧版本草案复用或澄清；指导 revise 清空全部意见；未证明无关变化安全时仍保守失效。
- MC 已知依赖不就绪、停用恢复路线：draft/planning_request 提前拒绝，不保存伪计划。上述合法草案通过率保持5/5；不能据此声称真实模型草案质量提高或所有动作首次校验率提高。
- 无变化回收不再次采集；新结果/未验收 backlog 与中断提交仍经过原机制。不是以跳过账本核对来减少工具次数。

## 验证与未验证项

py1 最终针对性回归：370 passed，40.69 秒；14 条第三方 Matplotlib/Pyparsing 弃用警告。新增 test_decision_flow_boundaries 共31例，另同步 MCP 工具白名单。自带私有 Simulation 计划的负向用例先复现失败，再修复，最终通过。

运行环境：PYTHONDONTWRITEBYTECODE=1；OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1；PYTHONPATH=tests/fake_only_guard;<repo>;E:/Py-Code。pytest -q -p no:cacheprovider --basetemp E:/Py-Code/pdt --disable-warnings；仓库 conftest 按 TemporaryDirectory 隔离并清理 TEMP/TMP。基准额外设置 TEMP/TMP=E:/Py-Code/pdt。

回归范围：decision_flow_boundaries、native_team_performance、role_context_projection、codex_native_team、decision_team、simulation_agent、four_agent_contracts/closed_loop、codex_context_binding/no_project_model/offline_cold/branch_evidence_binding、native_approval、single_action_dispatch、lifecycle_recovery_replay、memory_recovery_boundaries/integrity、review_memory_lifetime、decision_evidence_catalog、branch_evidence_transport、codex_branch_flow_fixes、role_permissions_and_research_plan、transport_optimization、expert_model/workflow_transport、state_field_index、strong_state_versions、codex_mcp、return_observation、collect_only_recovery、result_recovery_commit、recovery_error_fastpath/rescan_and_branch_repair、execution_recovery_scope、abandoned_perturbation_recovery。

Ruff check phase_agent 和新增测试/基准通过；本次17个代码文件格式检查通过。全包 format --check 仍报告144个文件需格式化、761个已符合；未扩散格式化其他任务文件。按用户要求由主 agent 自审，没有独立子 agent 审查。检查了源快照兼容、调用方、有限澄清、草案授权边界和历史保留。

临时合成项目每轮由 TemporaryDirectory/finally 清理；首次默认临时目录权限错误未产生科研状态，后续验证使用短可写目录，交付前确认该失败目录已不存在，且无本次 phase-flow-* 目录残留。未清理其他任务目录。


没有测试真实模型推理质量、真实用户会话咨询次数、所有操作的环境/输入就绪、真实大项目端到端性能或部署效果。原生客户端仍需遵循新入口说明；Python 不控制其是否把完整长聊天传给子 Agent。未做跨版本专业分析依赖缓存、自动补算、自动冲突仲裁、提示科学规则版本化合并或回收框架重写，因为这些缺少安全依赖设计或没有可复现收益证据。

没有新增科学执行授权；源码完成不表示服务已经加载。真实服务继续由用户按既有生命周期和审批入口管理。

## 测量原始计数摘要

```json
{
  "after": {
    "median_seconds": 0.04474549999576993,
    "runs": [
      {
        "clarification_rounds": 0,
        "expert_calls": 2,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "none: same source, question and active guidance",
        "real_model_calls": 0,
        "reconsultations": 0,
        "response_bytes": 24115,
        "seconds": 1.409746899997117,
        "tool_roundtrips": 5
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 2,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "none: same source, question and active guidance",
        "real_model_calls": 0,
        "reconsultations": 0,
        "response_bytes": 24115,
        "seconds": 0.04474549999576993,
        "tool_roundtrips": 5
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 2,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "none: same source, question and active guidance",
        "real_model_calls": 0,
        "reconsultations": 0,
        "response_bytes": 24115,
        "seconds": 0.04474659998959396,
        "tool_roundtrips": 5
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 2,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "none: same source, question and active guidance",
        "real_model_calls": 0,
        "reconsultations": 0,
        "response_bytes": 24115,
        "seconds": 0.04458099999465048,
        "tool_roundtrips": 5
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 2,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "none: same source, question and active guidance",
        "real_model_calls": 0,
        "reconsultations": 0,
        "response_bytes": 24115,
        "seconds": 0.043808000002172776,
        "tool_roundtrips": 5
      }
    ],
    "scenario": "same-facts unaccepted draft explanation edit"
  },
  "baseline": {
    "median_seconds": 0.06965239999408368,
    "runs": [
      {
        "clarification_rounds": 0,
        "expert_calls": 4,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "revision: all opinions cleared",
        "real_model_calls": 0,
        "reconsultations": 2,
        "response_bytes": 35890,
        "seconds": 1.4235092000017175,
        "tool_roundtrips": 8
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 4,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "revision: all opinions cleared",
        "real_model_calls": 0,
        "reconsultations": 2,
        "response_bytes": 35890,
        "seconds": 0.06965239999408368,
        "tool_roundtrips": 8
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 4,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "revision: all opinions cleared",
        "real_model_calls": 0,
        "reconsultations": 2,
        "response_bytes": 35890,
        "seconds": 0.06569370000215713,
        "tool_roundtrips": 8
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 4,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "revision: all opinions cleared",
        "real_model_calls": 0,
        "reconsultations": 2,
        "response_bytes": 35890,
        "seconds": 0.06881050000083633,
        "tool_roundtrips": 8
      },
      {
        "clarification_rounds": 0,
        "expert_calls": 4,
        "final_status": "awaiting_human_input",
        "first_draft_passed": true,
        "invalidation_reason": "revision: all opinions cleared",
        "real_model_calls": 0,
        "reconsultations": 2,
        "response_bytes": 35890,
        "seconds": 0.07061460000113584,
        "tool_roundtrips": 8
      }
    ],
    "scenario": "same-facts unaccepted draft explanation edit"
  }
}
```

修改前关键模块 SHA256（首次编辑前读取，仅用于识别工作树基线）：

- agents/contracts.py: c76afeb527995717997475cd230325da642d9237b9160074e2ae1e3ccb869b8e
- agents/strategic/codex_team.py: a9b5047fd7102dd45f1ebe727ceb6e922a09919ca6c62cd3124b65788c563d1e
- runtime/codex/team.py: db7af87790391e96e6d6d38445a7ff6b336cf1651d122c210447d5491aefd63c
- tools/dispatch/scientific_operations.py: e8c86eb629c06e3fbc55d9f2f28d4bce90e9ee8592e7d59e39737d9784b57437
