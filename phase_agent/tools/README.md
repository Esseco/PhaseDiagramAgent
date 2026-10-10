# 动作执行与约束

工具消费通过校验并获得所需批准的动作，核对版本与预算，生成输入、派发或回收结果并保存回执。在线路径每轮最多一个科学动作；独立超算 step_runner 保留同样的约束。

## 从哪里读

- [dispatch/create_tool_registry.py](dispatch/create_tool_registry.py)
- [workflows/run_tool_step.py](workflows/run_tool_step.py)
- [policy/validate_tool_action.py](policy/validate_tool_action.py)
- [workflows/create_dft_selection_handler.py](workflows/create_dft_selection_handler.py)

职责和审批边界见 [Graph 架构](../../docs/architecture/graph.md#doc-graph_architecture)，使用说明见 [文档索引](../../docs/README.md)。
