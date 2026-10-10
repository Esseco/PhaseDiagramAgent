# 应用启动与服务适配

提供项目选择器、Studio 生命周期、HTTP 控制接口及依赖组装。消息进入 dialogue 图，确认配置后使用 project 图；本包不另建科学决策循环。

## 启动与独立入口

在源码根目录使用 py1：

```powershell
python -m phase_agent.runtime.local_project_launcher
python -m phase_agent.runtime.studio_service --runtime-config '项目目录/agent_runtime.json'
```

两条命令选择一种启动同一项目。`studio_service` 绑定项目并管理 Studio/控制服务；直接使用 LangGraph CLI 的环境设置见 [Studio 说明](../../docs/guides/start.md#doc-studio_local)。

无常驻服务的超算分步工具为 `python -m phase_agent.runtime.step_runner --help`，输入使用已确认配置和经过审批的方案；它不建立第二套聊天决策流程。成本报告为 `python -m phase_agent.runtime.estimate_calculation_cost --help`，只读取文件并生成报告。

## 从哪里读

- [local_project_launcher.py](local_project_launcher.py)
- [studio_service.py](studio_service.py)
- [studio_runtime.py](studio_runtime.py)
- [runtime_composition.py](runtime_composition.py)
- [chat_application.py](chat_application.py)

职责和审批边界见 [Graph 架构](../../docs/architecture/graph.md#doc-graph_architecture)，使用说明见 [文档索引](../../docs/README.md)。
