# Codex 适配包

- agent.py：本地项目服务生命周期。
- control.py：现有 HTTP 控制服务的薄客户端。
- mcp.py：stdio 工具发现、项目选择和协议分发。
- context.py / reports.py：共享科学上下文、规则审计与方案展示。
- offline.py：复用同进程科研工作流。
- team.py：原生专家、人的方向反馈与记忆审核的 MCP 契约。

科学角色在 agents/，决策讨论在 agents/strategic/codex_team.py，LangGraph 执行与恢复在 graphs/。本包不实现第二套科学调度。

根目录 codex-agent.cmd、codex-mcp.cmd 及原 runtime.codex_agent/control/mcp 的 -m 命令保持可用；三个旧路径文件只有 main 入口，不保留重复实现。其他历史模块描述符由 compatibility 统一转换。

[使用指南](../../../docs/guides/codex-team.md)
