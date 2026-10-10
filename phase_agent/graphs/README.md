# LangGraph 流程入口

图负责状态传递、节点连线、条件路由与暂停恢复；节点调用业务服务和工具。State 保存可序列化事实，运行依赖放入 Runtime context。

## 从哪里读

- [dialogue/graph.py](dialogue/graph.py)
- [project/graph.py](project/graph.py)
- [actions/graph.py](actions/graph.py)
- [state.py](state.py)
- [runtime_context.py](runtime_context.py)

职责和审批边界见 [Graph 架构](../../docs/architecture/graph.md#doc-graph_architecture)，使用说明见 [文档索引](../../docs/README.md)。
