# Graph 架构与审批恢复

<a id="doc-graph_architecture"></a>

## 实际入口与职责

| 层次 | 实现 | 职责 |
| --- | --- | --- |
| Studio 网关 | runtime/studio_service.py；graphs/studio_chat_graph.py | 服务、消息接收和答复，不代替科学流程 |
| 对话 | graphs/dialogue/graph.py | 读取项目、上下文、语义决策、配置、审阅、科研调用与展示 |
| 科研主图 | graphs/project/graph.py | initialize → observe → wait → analyze → react → finalize |
| 专家咨询 | graphs/decision_team/graph.py | 不可变证据、按需 Exploration/Reliability、Strategic 最终提案 |
| 唯一动作 | graphs/actions/graph.py | 提案、Simulation 规划、授权、验证、一次工具调用、检查及审计 |
| 恢复子图 | graphs/recovery/ | 已有批次、执行核对、方向审阅和训练交接 |

[编译图](flow.md) 直接由 graph_overview.render_overview 生成，测试检查文档与实际拓扑一致。[四 Agent](four-agents.md) 定义角色与权限；具体函数仍在 tools/science/analysis，不为展示另建调度器。

## 状态与执行边界

图 State 只含可序列化事实。客户端、manager、锁和回调放 Runtime context。科学事实以原 state/ledger 为准；模型说明、图追踪和检查点不构成计算成功或授权证据。

持久项目节点名及线程身份保留。旧 bounded_action_graph 追踪名称表示单次动作，不是多步执行循环。每次科学调用 max_steps=1；等待、拒绝与问答不创建科学任务。一次获批封装内部的计算范围仍受原模块契约限制，不批准额外科学路线。

<a id="doc-execution_policy"></a>

## 提案、授权与唯一执行链

Action 使用原 tool/action_type/operation/parameters 契约。scientific_operations 描述科学操作，create_tool_registry 注册工具；最终动作先归一化和验证，再通过 execute_tool_action 派发。专家只能输出建议，Simulation 是执行角色，Strategic 不能直接调用计算。

Simulation 计划在审批前冻结，包含前置检查、一次注册工具调用和结果验证；失败诊断只引用实际观察，不能自行重试或改变参数。批准复用冻结动作，不再次请求模型；非法修改不消耗原审批。旧未携带新计划的冻结方案保留历史契约。

普通人工动作走原执行审批；重大决策、严重冲突、敏感操作及配置高成本风险按需先审阅方向。接受方向不等于批准执行。自动授权仅来自已确认 workflow.auto_approved_operations（默认空列表），运行时覆盖、模型及记忆不能授予权限。dry-run 不执行工具；replay 仍检查身份、版本与回执，不重放副作用。

<a id="doc-native_approval_recovery"></a>

## 持久审批和暂停恢复

项目主图在 wait_for_project_input 使用原生 interrupt，reconcile 信号以新事实恢复。无检查点的动作图不直接 interrupt；其待审方案复用外层持久审批、身份和数据库。

训练交接拥有独立 SQLite 检查点与线程；manifest、验证回传、缺配置及激活审核可暂停。恢复时重新读取当前状态和产物，不把旧检查点当作事实权威。参数、配置、模型或证据改变时旧方案需重新审阅，已有完成任务保留。

审批与执行持久层记录原动作、配置/模型绑定、领取与返回证据。异常不能证明没有副作用；响应不确定时先核对回执和已登记任务，不重复执行、不删除收据解除保护。行政核对仍需精确身份与人工审核，不创建科学任务。并发写入共用项目生命周期锁。

<a id="doc-batch_recovery_and_trace"></a>

## 观察、回收与等待

observe 内依次 collect_and_reconcile、batch_recovery、training_lifecycle、edge_direction_review、scientific_feedback；这些阶段先核对已登记结果，再进入等待判断和本轮分析。准备成功、上传、提交、运行和科学结果验收是不同状态。

回收验证批次/任务、来源模型、参数、校验和和结果完整性。失败或缺文件保持可解释等待，不重算已完成结果；行政结果接纳及输入恢复复用现有入口。node_trace、progress_events 只提供观测，不授予权限，也不替代结果校验。

同消息完成答复可重送，但科学动作不得重放。未结清执行、预算预留或审批歧义仍阻止新动作。历史检查点的隔离和路由继续由唯一 project runner 负责，不建立平行恢复器。

## 维护与验证

图连线在 graph.py，业务适配在 nodes.py，科学逻辑放原业务包。更改持久节点名、线程或副作用协议前须检查历史恢复测试。入口和操作见 [开发指南](development.md)、[恢复](../guides/recovery.md)、[训练](../guides/training.md)、[Codex](../guides/codex-team.md)。

运行 py1 中 test_graph_overview、test_application_layout、单动作/原生审批、生命周期与训练跨进程测试；真实 API/HPC 需要独立验证。源码维护不自动重启服务。
