# Strategic Agent

agent.py 选择咨询范围并形成最终提案，prompts.py 定义策略取舍；research_plan.py 表达有界路线，human_review.py 检查风险和已确认授权；codex_team.py 保存原生 Codex 讨论、意见、修订与提交身份。

propose_tool_action/revise_tool_proposal 保留现有工作流提案接口；round_budget_review 与 scientific_decisions 汇总已有科学决策契约；strategy/ 和 choose_debug_next_action 保留显式离线工具。它们不是在线语义的第二个调度器。

Strategic 综合专家意见，不重做其专业报告；只批准方向或提出动作，不直接计算。未来 research_plan 步骤没有执行权。Simulation 和现有工具执行经校验、授权并冻结的当前动作。
