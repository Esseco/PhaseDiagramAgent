# 仓库目录与维护入口

当前应用只有一个生产包 phase_agent，保留现有 LangGraph、科学算法和审批恢复协议。科研数据项目不包含应用代码。

```text
Process_PhaseDiagram/
├── phase_agent/
│   ├── agents/
│   │   ├── exploration/    探索充分性与 Branch/Relax/MC 建议
│   │   ├── reliability/    MLIP/DFT 与相稳定性可信度建议
│   │   ├── strategic/      咨询、路线取舍、人工意见及最终提案
│   │   ├── simulation/     执行规划、结果检查与异常诊断
│   │   ├── common/         公共提案和对话契约
│   │   ├── contracts.py    专家 JSON 契约
│   │   └── permissions.py  角色权限
│   ├── graphs/             dialogue/project/decision_team/actions/recovery
│   ├── evidence/           当前事实与模型/项目绑定的证据
│   ├── models/             模型客户端、能力、传输与用量
│   ├── policies/           参数、预算、授权及版本校验
│   ├── tools/              注册、输入准备、执行、远端交接及回收
│   ├── science/            MC、MLIP、DFT、训练和结构算法
│   ├── analysis/           凸包、误差、成本与分析输出
│   ├── configuration/      模板、草稿和确认配置
│   ├── persistence/        台账、模型登记与审核记忆
│   ├── compatibility/      旧模块描述符集中映射
│   └── runtime/            启动器、Studio/HTTP 和 Codex 接口
├── .codex/agents/           Codex 原生专家角色定义
├── docs/
│   ├── guides/             当前操作与科学使用说明
│   └── architecture/       当前架构、编译图和验收记录
├── tests/                  单元、审批、恢复及闭环测试
├── experiments/            独立研究实验，部分有现有测试依赖
├── settings/               本地运行配置与示例
├── local/                  既有本地状态，非可删除源码
├── .langgraph_api/         既有运行状态，不参与源码清理
├── langgraph.json          Studio 图和 HTTP 注册
├── codex-agent.cmd         原服务控制入口
└── codex-mcp.cmd           原 MCP 入口
```

## 阅读顺序

1. [四 Agent](four-agents.md)：职责、输入输出、权限、人工意见和记忆。
2. [科研生命周期](workflow.md)：边界配置、同模型内迭代、跨 epoch 更新。
3. [Graph 架构](graph.md)：实际节点、执行边界与持久恢复；[编译图](flow.md) 来自真实图。
4. [开发指南](development.md)：源码定位与必须保留的不变量。
5. [文档导航](../README.md)：启动、文件、训练、恢复和 Codex 接管。

## 文件归属

Agent 负责分析、策略或执行规划；具体科学计算在 science，带副作用的任务准备和交接在 tools。按已配置阶段推进单结构任务的 decide_next_calculation 位于 tools/state，不属于 Reliability 的科研判断。它保留现有逻辑，不授予任务执行权。

旧顶层执行包和 coordinator 包不恢复。历史模块名经 compatibility/module_references.py 与 module_moves.json 解析；legacy 命名的台账、结果和知识读取器仍有实际用途，保留。公开启动命令、MCP 名称、历史审批及项目数据格式继续使用。

源码外置备份在同级 _Process_PhaseDiagram_source_backups，不加入 PYTHONPATH。已被取代的三角色验收、旧开发计划和一次性维护/诊断记录从当前仓库删除，原文保存在本轮源码 ZIP；当前指南不再链接这些历史文件。

## 安全整理范围

整理不迁移或清除 local、settings、.langgraph_api、真实项目 structures/parameters/workflow_state/submissions、结果或模型。运行服务由用户手动管理；源码更新不自动重启服务或提交科研任务。

## 本轮整理验证（2026-11-01）

- 从原仓库更新 23 个文件，删除 60 个已取代或迁走的文件。计算阶段工具内容保持不变，只迁移位置并增加旧模块映射；没有改变 epoch 路由、科学策略或执行行为。
- py1 针对性回归 63 passed（16.99 秒）：计算阶段、可配置执行、旧描述符、应用布局、图投影、四 Agent 闭环、Codex 合成新项目及 Simulation/人工审阅。
- 全仓库 4,144 项测试收集通过；本轮没有重跑此前已通过的全量执行。静态检查通过，806 个生产 Python 文件格式检查通过。
- 当前 Markdown 本地目标链接检查通过；原 settings/local/.langgraph_api 共 4 个文件哈希未变。保留 experiments 及仍有效的 legacy 读取器。
- NiCo、Ni、Cr、Co、FeMn 五项目只读检查均 ready、issues 为空；没有启动真实服务、模型 API 或 HPC 计算。

外置源码/历史文档备份：E:/jupyter notebook/1-AL_for_PhaseDiagram/_Process_PhaseDiagram_source_backups/layout-cleanup-20261101-212647.zip。测试临时目录由现有 TemporaryDirectory 清理；维护脚本和临时计划在交付前移除。
