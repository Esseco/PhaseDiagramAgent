# 四 Agent

| 目录 | 输入与输出 | 主要入口 |
| --- | --- | --- |
| exploration/ | 同一事实快照 → 搜索充分性、覆盖、多样性与 Branch/MC 建议 | agent.py、generation_contracts.py、mc_contracts.py |
| reliability/ | 同一事实快照 → 模型误差、QBC/OOD、相稳定性风险与 DFT/训练建议 | agent.py、dft_contracts.py、selection/ |
| strategic/ | 两专家意见、目标、预算和人工意见 → 一个最终 Action 与可选路线 | agent.py、research_plan.py、human_review.py、codex_team.py |
| simulation/ | 冻结 Action → 执行计划、实际结果检查和证据诊断 | agent.py、contracts.py |

contracts.py 定义专家 JSON，permissions.py 限制各角色合法操作；common/ 保存公共 Action 与对话契约。专家不直接调用计算或授予审批。Strategic 不直接执行；Simulation 不改变科研目标。

推理在 graphs/decision_team，Simulation 规划/检查在 graphs/actions。实际计算复用 tools/science，按已配置阶段推进的执行工具在 tools/state/decide_next_calculation.py。每次只有一个已授权动作，没有第二套 scheduler。

普通项目模型与 Codex 原生接管共用契约和科学事实；旧 callable、冻结方案及保存模块描述符保留兼容。详见 [四 Agent 架构](../../docs/architecture/four-agents.md)。
