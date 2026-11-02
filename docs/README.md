# 文档导航

当前目录只区分 guides（使用）与 architecture（维护）；被取代的开发日志与旧验收已外置备份。

| 目的 | 入口 |
| --- | --- |
| 仓库文件在哪里、哪些必须保留 | [目录架构](architecture/repository-layout.md) |
| 四个 Agent 如何协作 | [四 Agent](architecture/four-agents.md) |
| 新材料与两个 epoch 怎样运行 | [科研生命周期](architecture/workflow.md) |
| 启动、对话与调试 | [启动指南](guides/start.md) |
| 科学配置、边界与预算 | [配置](guides/configuration.md) |
| 输入、回传、模型和 epoch 文件 | [项目文件](guides/project-files.md) |
| 搜索、DFT 路线与模型刷新 | [搜索](guides/search.md) |
| DFT 输入与回收 | [DFT](guides/dft.md) |
| 训练、独立验证和模型激活 | [训练](guides/training.md) |
| 相图、误差与分析结果 | [分析](guides/analysis.md) |
| 状态诊断与恢复 | [恢复](guides/recovery.md) |
| Codex 接管与原生团队 | [Codex 四 Agent](guides/codex-team.md) |
| Codex 命令和控制协议 | [控制入口](guides/codex-control.md) |
| 短期事实与审核知识 | [记忆](guides/memory.md) |
| 研究提案与执行范围契约 | [科研决策](architecture/scientific-decisions.md) |
| 实际节点、审批和恢复 | [Graph 架构](architecture/graph.md) |
| 源码定位、修改约束 | [开发指南](architecture/development.md) |
| 实际编译图 | [流程图](architecture/flow.md) |
| 四 Agent 架构重构验收 | [验收记录](architecture/four-agents-validation.md) |

默认本地环境 py1。真实项目独立于应用源码；开发验证使用假模型、假超算和最小合成数据。源码总入口见 [README](../README.md)。
