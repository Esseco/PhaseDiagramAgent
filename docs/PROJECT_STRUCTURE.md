# 项目架构索引

采用官方示例中图构建、节点、状态、工具分离的原则，而非强制所有业务塞进单个agent目录。
参考：https://docs.langchain.com/oss/python/langgraph/application-structure

## 从哪里开始读

1. run/studio_service.py：服务入口；run/local_project_launcher.py：本地启动。
2. run/main.py：装配运行依赖和生命周期业务回调。
3. orchestration/search_workflow_graph.py：生命周期节点连接和条件分支。
4. orchestration/event_loop_graph.py：有界动作图；execution_layer/workflows/run_event_loop.py提供持久业务回调。
5. orchestration/tool_action_graph.py：审批、校验、工具执行的连接。
6. orchestration/proposal_graph.py与proposal_nodes.py：模型建议与提案校验；decision_layer保留科学判断与兼容入口。

## 编排包

```text
orchestration/
  search_workflow_graph.py  生命周期图构建与路由
  tool_action_graph.py      工具图构建与路由
  event_loop_graph.py       有界事件图
  proposal_graph.py         决策图
  proposal_nodes.py         模型请求与校验节点
  lifecycle_nodes.py       生命周期节点适配器
  tool_nodes.py            工具节点适配器
  state.py                 图可见状态定义
  runtime_context.py       每次调用独立的运行依赖
```

节点适配器连接现有业务回调，不复制科学算法。
run/main.py仅保留公开签名与回调装配。生命周期业务实现已拆分为：

| execution_layer/workflows下的模块 | 职责 |
| --- | --- |
| lifecycle_initialization.py | 确认配置、版本迁移校验与运行依赖初始化 |
| lifecycle_recovery.py | 回收对账、科学反馈、等待屏障与本轮分析导出 |
| lifecycle_finalization.py | 调用动作循环、准备输入与最终状态保存 |
| lifecycle_support.py | 状态读写、事实摘要与迁移说明 |
| tool_outcomes.py | 工具审计记录与结果状态合并 |
| tool_proposal_revision.py | 人工意见修订、预算预览与重新待审；不执行工具 |
| initial_tool_proposal.py | 初次科学提案选择、阶段约束与成本预览 |

动作循环通过显式event_loop参数注入finalization，不依赖反向导入run/main.py。
run_tool_step.py保留公开接口、工具节点实现与提案准备，结果审计/合并已分离。
原workflows下同名graph导入包装已移除，图定义统一从orchestration导入。

## 功能边界

| 目录 | 内容 |
| --- | --- |
| config_layer | 默认参数、校验、配置确认与版本 |
| decision_layer | 提案、科学判断、采点与策略 |
| execution_layer | 工具注册/执行、审批、预算、状态协议和工作流业务回调 |
| scientific_layer | 结构、MC、DFT、MLIP与训练算法 |
| analysis_layer | 相识别、相图、误差、收益和成本分析 |
| data_layer | 数据、模型记录、台账和记忆 |
| run | CLI/Web、运行装配和展示 |
| tests | 单元与跨层集成测试 |
| experiments | 独立方法实验，不写正式运行状态 |

依赖方向：入口装配 → 图与节点适配 → 业务实现。科学算法不反向依赖聊天页面。
工具注册见execution_layer/dispatch/create_tool_registry.py；无需再新增一份重复tools清单。

## 代码与业务工作区不同

本索引描述代码仓库。epoch、提交、结果、输出和记忆文件的布局见WORKSPACE_LAYOUT.md，不能因为代码整理而自动移动已有任务文件。
当前使用本地Python服务，不是LangSmith Agent Server部署；因此不放一个不能直接运行的langgraph.json。
持久审批子图已使用interrupt/Command和SQLite checkpointer；科学执行仍以state/ledger恢复。细节见NATIVE_APPROVAL_RECOVERY.md。

状态边界：orchestration/contracts.py严格校验审批/执行身份；execution_layer/state/execution_receipts.py记录执行开始与返回，公共工具适配器在业务完成记录缺失时阻止重复执行。收据不替代state/ledger。

## 尚需继续拆分的边界

- 初次提案已移入initial_tool_proposal.py，人工修订在tool_proposal_revision.py；run_tool_step.py保留已有审批更新与节点装配。阶段约束仍有明确顺序，不用隐式动态模块代理。
- run/chat_application.py拥有聊天handler；chat_intent_routing.py归一化语义意图，chat_review.py校验审批页决定，chat_execution.py组装工作流调用。open_webui_api.py保留兼容HTTP入口，不再定义handler。重生成/配置等既有命令仍由handler顺序协调，未复制实现。
- 事件图和决策图定义已迁入orchestration，不存在第二份实现。事件循环业务回调仍通过调用期闭包持有入账状态，不具备原生持久恢复。
- 原生持久恢复需先定义可序列化业务状态与副作用恢复协议。Runtime分离不等于检查点恢复完成。

这些是明确的后续架构工作，不属于已完成的重构，也不影响当前回归覆盖下的运行接口。

本次结构拆分验证：py1隔离回归820项测试、5项子测试通过（52.06秒），排除暂不使用的BOHB集成测试；未进行生产数据分析或提交作业。

生命周期业务与工具结果模块拆分后的复验：820项测试、5项子测试通过（176.10秒）；仍排除BOHB集成测试。原公开运行入口、审批、回收和结果入账保持测试覆盖。
