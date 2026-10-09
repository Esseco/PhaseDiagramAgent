# 科学主图可读性方案（保留运行结构）

当前已确认方案：保留生命周期 → 有界循环 → 工具图三层执行结构，采用只读总览与真实子图帮助阅读。原生产主图扁平化目标不再推进。

已完成：run_tool_step 抽出 tool_step_stages；run_event_loop 抽出 ActionLoopSession 和 prepare_event_loop，保留原公开入口；生命周期动作准备与结果合并分别提取为 prepare_workflow_actions、complete_workflow_actions。回调顺序、审批、持久化与停止规则保持不变。事件循环相关回归 31 项通过，生命周期相关回归 20 项通过。

未实施生产主图展开；现有审批、防重、预算、入账和恢复路径保持不变。上述接口提取不代表运行图已经扁平化。

现有三层图已有明确名称：scientific_lifecycle（回收与分析）、bounded_action_iteration（有界循环与入账）、approved_scientific_action（提案、审批、校验和工具分流）。名称仅帮助辨认，不改变路由，不表示三层已经扁平化。

总览见 [科学流程总览](SCIENTIFIC_FLOW_OVERVIEW.md)，由 orchestration.graph_overview 读取真实编译图生成，工具节点仅在说明中合并，完整工具名称单独列出。没有新增可执行图，也不自动改写工作区文件。

验收：文档与生成器严格一致；投影读取真实节点与连线且不调用 invoke；现有审批、循环、子图发现和持久化回归保持通过。未来图定义变化后，重新运行生成器并更新本文链接的总览，测试会检测过期内容。
