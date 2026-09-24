# 决策记忆

人工长期知识和短期运行经验分开传给 Agent。状态文件是存储来源，不需要新增数据库。

`decision_memory.long_term` 包含四类显式知识：`human_system_knowledge`、
`physical_priors`、`frozen_parameter_advice`、`search_rules`。审批反馈可以通过
`long_term_memory` 字典修订其中任意类别。`decision_memory.short_term` 则由
state manager 每轮替换，保存近期 action、收益、失败任务、QBC 和当前搜索状态。

## 初始化长期建议

在正式 `run_workflow(...)` 调用中增加：

```python
initial_long_term_advice=[
    "优先 DFT-SP，DFT-relax 需要说明晋级理由",
    "关注覆盖不足的组成区域，并保留全局探索",
]
```

该参数仅在 state 尚无 decision_memory 时初始化；恢复运行不会覆盖已经修订的建议。建议不改变冻结参数、预算上限或 BOHB 所有权。

## 每轮文件修订

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

## Agent 每次参考的内容

- `long_term_human_advice`：有效人工建议、版本和来源，持续保留。
- `current_phase_diagram`：MLIP/DFT 分开的当前相图版本、稳定条目数及最低 Ehull 条目（最多十项）。
- `recent_experience`：最近五条收益观测、最近五条 action 的状态及人工意见。条数不等于轮数；一轮可能产生多条记录。
- `coverage_gaps`：调用方当前提供的覆盖缺口。

近期经验从现有 action_records/rewards 提取，不由 AI 自动写成长久规则。历史数值按原值呈现，不把相关性描述成因果结论。缺失的相图和收益数据保持缺失，不生成虚构观测。新收益记录附加前后相图版本与能量基准，但相图版本本身不足以证明模型可比，跨模型比较仍需核对计算来源。

通用 action、人工修订、轮次策略和 DFT 决策均接入统一上下文。独立调用轮次/DFT接口时需要传入含 decision_memory、rewards 和 phase_diagrams（或 phase_diagram_state）的状态；正式主动学习调用会转交这些字段。通用 proposal JSON/MD 同时展示长期建议、当前相图与近期经验，便于审查。
