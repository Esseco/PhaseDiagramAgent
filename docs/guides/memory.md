# 上下文、记忆与分析客户端

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [LLM 决策与 token 管理](#doc-llm_decision_context)
- [可选 Deep Agents 分析适配](#doc-deep_agents_analysis)
- [可选混合分析客户端](#doc-hybrid_proposal)
- [Agent token policy](#doc-token_policy)
- [决策记忆](#doc-decision_memory)

<a id="doc-llm_decision_context"></a>

## LLM 决策与 token 管理

LLM 综合持久人工建议、物理先验、冻结参数、搜索规则、已批准知识、近期科学行动与收益、最新版本现状和本次要求做决策。明确人工约束不被旧经验覆盖，跨模型收益未经验证不比较。

传输时去除重复候选、原子坐标与原始日志，保留所有候选 ID 和科学指标。长期约束不截断；近期记录之外保留各科学动作的最近证据，知识按当前动作检索。压缩只作用于请求副本，不修改持久记忆。记录字符大小用于诊断，字符数不等同实际 token；实际用量以 API 为准。

普通请求使用小输出额度；DFT 根据方案规模调整输出额度，最多重试一次截断响应。理由和证据引用简短，展示用 Na、相、Ehull 与成本由本地读取，不让模型重复抄写。

代码仅核验身份、重复任务、真实成本和已确认硬约束。相/Na覆盖为软提醒，最终取舍由 LLM 解释。DFT 方案不合规最多请求 LLM 修订一次，不由代码重新采点或改计算类型；修订后重新预览并等待人工批准。失败不执行任务或暂停搜索。

### 实现入口

[phase_agent/analysis/state/build_decision_context.py](../../phase_agent/analysis/state/build_decision_context.py)。

<a id="doc-deep_agents_analysis"></a>

## 可选 Deep Agents 分析适配

主流程始终由 LangGraph 编排。`deepseek.proposal_harness=deepagents` 显式选择此模型分析适配器；默认 `legacy` 标签表示直接 DeepSeek JSON 客户端，沿用已有配置名称，不代表旧执行框架。

适配器只接收摘要白名单、已审核证据及输出契约，使用调用期 StateBackend。它不能执行科学工具、提交、激活或接管项目记忆；结果仍经过同一 proposal 图校验和人工审批。适配器失败由调用方记录，不静默调用另一条付费分析路径。

本地假模型测试不能证明真实模型质量、提供商用量或超算可用性。配置中的步数/响应 token 限制也不等于总费用限额。

[phase_agent/decisions/agent/deepagents_proposal.py](../../phase_agent/decisions/agent/deepagents_proposal.py)。

<a id="doc-hybrid_proposal"></a>

## 可选混合分析客户端

`deepseek.proposal_harness=hybrid` 显式启用此分析适配。配置对话、普通问答和常规方案使用直接 JSON 客户端；只在 proposal 模式且请求明确指定 `analysis_harness=deepagents`，或预算证据含至少两份训练报告时使用 Deep Agents。

`hybrid_proposal.py` 的选择条件控制分析客户端，不选择科学动作、不批准工具，也不创建第二套项目状态。普通候选数量大不会自动切换分析适配器。`_llm_usage` 保存实际报告的调用数/token，`_analysis_route` 保存分析路径、原因和耗时；缺少真实费用凭证时费用保持未知。

`deepagents_max_tokens` 和 `deepagents_recursion_limit` 限制单响应规模及分析步数。分析结果继续进入同一 LangGraph proposal 校验和审批路径。真实项目是否启用取决于其运行时配置，本说明不修改项目配置。

[phase_agent/decisions/agent/hybrid_proposal.py](../../phase_agent/decisions/agent/hybrid_proposal.py)。

<a id="doc-token_policy"></a>

## Agent token policy

普通配置沟通、生成和文件准备关闭深度思考；具有相图、QBC 或模型过期证据的科学决策启用思考。明确的文件准备指令优先使用轻量模式。策略、DFT 选点、模型更新、收敛和预算分配请求使用重要决策模式。

本地运行时 JSON 的 `deepseek.routine_max_tokens` 默认 1600，`deepseek.reasoning_max_tokens` 默认 8192。它们是单次输出上限，不是每次实际消耗；`thinking=disabled` 可禁止所有深度思考。旧 `thinking=enabled` 改为仅允许重要决策使用，不再让普通对话全部思考。

请求发送前去掉重复状态、候选列表和坐标/占位大数组；本地台账、版本和科学结果不变。保留对象编号、科学指标、覆盖、预算和人工记忆。普通“继续/下一步/然后呢”在已有确定的文件准备流程时使用规则建议，仍走人工审批。

解析失败最多重试一次。思考耗尽输出上限时以非思考模式重试；失败响应的已知 API 用量也计入搜索预算，金额未知时不估造。网络故障无服务端用量证据时无法推断收费。

修改代码后重启本地 Agent 服务生效；无需改动已确认的科学配置。

[phase_agent/decisions/agent/prepare_llm_request.py](../../phase_agent/decisions/agent/prepare_llm_request.py)。

<a id="doc-decision_memory"></a>

## 决策记忆

### 有界检索与可解释使用

经验候选现在包含decision_outcome：仅按动作结果显式返回的task_id关联任务状态、原模型版本和实际成本。未返回任务ID时不猜关联；未完成任务标partial。候选仍不自动变成长期指令。新记录明确supersedes后，人工批准才将同作用范围/类别/适用条件的旧记录标为superseded并保留历史；跨适用条件替代会报错。相同条件的不同陈述进入冲突审核，不用关键词猜测科学矛盾。

检索保留人工审核与体系/动作/模型适用条件，增加相关词排序、去重、已替代记录过滤，以及显式model_dependent记录的模型版本隔离。决策上下文包含retrieval_reason，说明匹配词与适用条件。token_budget目前采用保守JSON字符上限，不是模型分词器的精确token数；超预算记录不截断，跳过以保留完整语义。尚未引入向量语义检索，也不自动证明或升级经验。

人工长期知识和短期运行经验分开传给 Agent。状态文件是存储来源，不需要新增数据库。

`decision_memory.long_term` 包含四类显式知识：`human_system_knowledge`、
`physical_priors`、`frozen_parameter_advice`、`search_rules`。审批反馈可以通过
`long_term_memory` 字典修订其中任意类别。`decision_memory.short_term` 则由
state manager 每轮替换，保存近期 action、收益、失败任务、QBC 和当前搜索状态。

### 初始化长期建议

在正式 `run_workflow(...)` 调用中增加：

```python
initial_long_term_advice=[
    "优先 DFT-SP，DFT-relax 需要说明晋级理由",
    "关注覆盖不足的组成区域，并保留全局探索",
]
```

该参数仅在 state 尚无 decision_memory 时初始化；恢复运行不会覆盖已经修订的建议。建议不改变冻结参数、预算上限或 BOHB 所有权。

### 每轮文件修订

保留 decision-rNNN.json 原有 proposal_id、revision、proposal_hash，填写：

```json
{
  "decision": "comment",
  "comment": "本轮请解释推荐范围，并参考更新后的长期建议",
  "long_term_advice": [
    "优先 DFT-SP，DFT-relax 需要说明晋级理由",
    "重点关注覆盖不足的区域"
  ]
}
```

`comment` 为本次意见；只有显式填写 `long_term_advice` 才更新长期建议。该列表完整替换旧列表，`null` 或省略表示不修改，`[]` 表示清空。旧版本及修改来源保存在 decision_memory.history 中。

长期建议一旦变化，Agent 先生成新版 proposal 并暂停，即使该次同时写了“同意”，也不会立即执行旧建议。检查新版后，在新版 decision 文件填写 `comment: "同意"` 或 `comment: "approve"` 继续。更新长期建议本身不授予 action 执行许可。

### Agent 每次参考的内容

- `long_term_human_advice`：有效人工建议、版本和来源，持续保留。
- `current_phase_diagram`：MLIP/DFT 分开的当前相图版本、稳定条目数及最低 Ehull 条目（最多十项）。
- `recent_experience`：最近五条收益观测、最近五条 action 的状态及人工意见。条数不等于轮数；一轮可能产生多条记录。
- `coverage_gaps`：调用方当前提供的覆盖缺口。

近期经验从现有 action_records/rewards 提取，不由 AI 自动写成长久规则。历史数值按原值呈现，不把相关性描述成因果结论。缺失的相图和收益数据保持缺失，不生成虚构观测。新收益记录附加前后相图版本与能量基准，但相图版本本身不足以证明模型可比，跨模型比较仍需核对计算来源。

通用 action、人工修订、轮次策略和 DFT 决策均接入统一上下文。独立调用轮次/DFT接口时需要传入含 decision_memory、rewards 和 phase_diagrams（或 phase_diagram_state）的状态；正式主动学习调用会转交这些字段。通用 proposal JSON/MD 同时展示长期建议、当前相图与近期经验，便于审查。

### 结构化记忆与体系经验包

新增 `decision_memory.records`（人工审核的长期记录）和 `memory_candidates`（自动收集的事实候选），不改变旧状态格式。记录包含作用范围、适用条件、证据引用、成熟度和有效状态。Agent 每次只读取少量相关且已审核记录；候选事实不能直接成为建议。

本地控制接口：`GET /phase/memory` 查看记录与审核队列；`POST /phase/memory/propose` 提议记录；`POST /phase/memory/review` 逐条批准或拒绝。体系 Skill 在用户确认收敛后生成于项目 `knowledge_export/<体系>-draft`，包括 `SKILL.md`、`profile.json`、`evidence.json` 和 `CHANGELOG.md`。草稿不自动发布；设置运行时 `knowledge_library_root` 或环境变量 `PHASE_SEARCH_KNOWLEDGE_ROOT` 后，使用 `POST /phase/memory/skills/publish` 明确批准发布。`GET /phase/memory/skills` 查看新项目匹配结果，`POST /phase/memory/skills/import` 只创建逐条审核提案，不改配置硬约束。

这些 Skill 是经验记录，不含可执行代码。预算耗尽、缺少收敛证据或仅有未经审核的候选观察不能发布。旧项目可调用 `backfill_memory_candidates(state_path, dry_run=True)` 预览，再以 `dry_run=False` 写回；写回前自动创建 `.memory-backup`，不删除原文件。

[phase_agent/persistence/memory/decision_memory.py](../../phase_agent/persistence/memory/decision_memory.py)。



## 参数、预算与结果经验

`analysis/state/budget_experience.py` 提供纯读经验视图。普通科学动作也纳入，无须先具有 `round_budget_review`。只按任务ID或 parent_decision_id 关联任务，不用批次巧合、时间邻近或数组顺序猜测归因。保存参数、预算上限、完成/失败数、可追溯实际成本、耗时、GPU核时及带模型/能量基准版本的收益。

耗时复用已有任务回收的 `runtime_observation` 和按任务ID关联的 `cost_history`；来源须为实际计时或可靠调度记录。GPU核时复用已有 accounting 接口；同一共享作业的总核时不按结构重复记账。相对成本、计时和GPU核时分别保留，缺失字段为null；估算和已申请预算不冒充实测，串行任务时间之和不等于并行批次墙钟时间。Slurm分区只是报告的硬件类别，不能证明具体GPU型号相同。

上下文按阶段、模型/配置版本、硬件类别、资源数、参数和相近原子数分组，附样本数、中位数、范围和真实任务引用。当前模型已知时不将旧模型样本混入可比组。最多8组/8个近期案例，共享12000字符的样本摘要额度，另保留覆盖计数、检索数量与约束说明；字符数不是供应商token数。单次观察不称为已验证规则，统计摘要不证明最优参数、因果节省或反事实收益。

经验接入 `round_budget_evidence.budget_experience`，直接DeepSeek及可选Deep Agents均可读取。`evidence_catalog`注册budget_experience和budget_cohort引用。事实自动进入 memory_candidates，可随任务回收更新；候选只保存摘要与任务索引，完整回执仍保留在tasks，重复收集不重复写入相同观察。自动事实积累不自动变成长久规则或跨项目结论，长期知识仍走审核与适用范围检查。

Agent结合经验提出下一轮预算、结构规模、MC步数和筛选安排，并说明缺失证据及不确定性。原有预算/冻结条件与审批继续生效，观察本身不能增加额度或执行任务。后续回收带有计时/accounting的真实结果才能扩充实测样本；此功能不主动查询超算，也不生成历史缺失数据。
