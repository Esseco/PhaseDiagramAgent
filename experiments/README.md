# 离线研究实验

`bohb/` 比较固定预算、随机 Hyperband 与 BOHB；`branch_surrogate/` 比较代理模型和特征方案。实验显式接收输入，与生产图的聊天、审批和任务状态分开。

生产入口见 [Graph 架构](../docs/architecture/graph.md#doc-graph_architecture)。已移除依赖旧规则调度器的 compare_rule_and_agent 实验，避免从实验重新引入另一套动作执行流程。
