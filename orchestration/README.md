# LangGraph 编排导航

本包集中生命周期与工具图定义，不实现科学计算。

| 文件 | 职责 |
| --- | --- |
| scientific_graph.py | 缓存实际生产子图层级；ContextVar 只绑定当前消息调用 |
| studio_chat_graph.py | 普通消息直达、防重与实际科学子图的可发现入口 |
| search_workflow_graph.py | 回收、分析、等待、评估导出、动作和收尾的连接 |
| tool_action_graph.py | 提案、审批、校验、工具分支和入账的连接 |
| event_loop_graph.py | 有界动作迭代与每步持久化顺序 |
| proposal_graph.py / proposal_nodes.py | 科学建议、提案校验及其节点 |
| approval_graph.py | SQLite持久审批、interrupt/Command和单次交付防重放 |
| lifecycle_nodes.py / tool_nodes.py | 节点适配器，调用既有业务实现 |
| state.py | 两个图的状态定义 |
| contracts.py | 审批与执行身份的严格持久边界校验 |
| runtime_context.py | 调用期依赖与上下文检查 |

其他模块按业务职责保留：

- run/main.py：公开运行入口与装配；生命周期业务节点见execution_layer/workflows/lifecycle_*.py。
- execution_layer/workflows/run_event_loop.py：事件业务回调、幂等与业务快照；图定义位于本包event_loop_graph.py。
- execution_layer/workflows/run_tool_step.py：工具图节点实现。
- decision_layer/agent/langgraph_decision.py：兼容决策入口；图定义位于本包proposal_graph.py。
- execution_layer：科学工具、审批、预算和执行协议。
- data_layer：数据与记忆。
- run/studio_service.py：聊天服务。

原 workflows 下两个同名图导入包装已移除，直接从本包导入图定义。
approval_graph.py是原生持久审批子图；其他科学图不启用检查点重放，仍以state/ledger恢复。

## 调用期上下文

runtime_context.WorkflowRuntime 保存每次调用独立的业务回调 frame（含 client/manager）。
生产主图状态只记录阶段、选中工具与简要返回状态，不携带运行对象或完整业务结果；完整结果由当前 context 交回业务调用方。底层测试可通过 expose_response=True 检查注入回调的完整返回值。
正式 run_* 入口自动创建 context；直接 build_* 后 invoke/stream 时，必须传入新的 context=WorkflowRuntime()。
不得跨并发调用共享该实例。此分离不等于已具备持久节点恢复：业务回调仍需读取原 state/ledger。

实际层级：Studio confirmed_local_chat → 科学生命周期 → bounded_action_graph → execute_validated_action 内的工具图。通过闭包显式引用同一 compiled graph，并传递给原业务调用；静态发现与实际执行引用同一子图，不维护单独的展示图。
EventLoopRuntime 独立保留当前步骤完整 outcome，持久化回调接收完整结果，图只流出状态。审批等待与校验拒绝仍终止派发；新注册工具必须同步进入编译图，不静默回退。
