# LLM 决策与输出契约

模型接收当前事实、用户原始请求和有限记忆，返回答复、配置请求或一个合法科学动作。程序校验结果并交给审批图；本包不提交计算、不写科学结果。

## 从哪里读

- [agent/propose_tool_action.py](agent/propose_tool_action.py)
- [agent/revise_tool_proposal.py](agent/revise_tool_proposal.py)
- [agent/proposal_validation.py](agent/proposal_validation.py)

职责和审批边界见 [Graph 架构](../../docs/architecture/graph.md#doc-graph_architecture)，使用说明见 [文档索引](../../docs/README.md)。
