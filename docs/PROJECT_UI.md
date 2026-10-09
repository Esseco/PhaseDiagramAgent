# 相图搜索项目 UI

从正式项目根目录，在 py1 中运行 `python -m run.local_project_launcher`。
也可使用 py1 的 pythonw.exe 打开 `run/start_projects.pyw`。

1. 点击“选择文件夹”。项目名称就是文件夹名；已有 agent_runtime.json 会自动加载，新目录会建立独立配置草稿。
2. 点击“启动服务”。服务会准备好该文件夹独立的 Studio 配置和端口，在下方显示 Studio 网址，不自动打开浏览器，也不发送计算指令。
3. 复制 Studio 网址到浏览器，在 Studio 中完成配置、开始或继续搜索、处理审批。仅关闭网页不会停止服务。
4. 点击“查看状态”查看所选项目摘要和 Studio 网址；点击“停止服务”停止本窗口启动的项目服务。可同时启动多个文件夹，各项目独立运行。

主界面只有“选择文件夹”“启动服务”“查看状态”“停止服务”，项目列表每 3 秒自动刷新。“运行中”表示本地服务在线，计算进度以项目摘要和 Studio 为准。

每个项目独立保存：

```text
agent_runtime.json                    本地运行设置
parameters/                           科学配置草稿及会话
workflow_state/                       搜索状态、台账、审批、执行记录
workflow_state/studio/.langgraph_api/  Studio 对话和 checkpoint
workflow_state/studio/service.json     本地服务身份和端口
logs/agent_server.log                  服务日志
structures/                           候选结构
analysis_outputs/                     相图等分析结果
submissions/                          计算输入批次
```

项目写入路径必须在其文件夹内。导入旧项目如果提示路径冲突，先核对目录；程序不会迁移或覆盖旧结果。
代码共用，但每个项目拥有独立进程、端口和状态。操作系统锁阻止同一项目重复启动。
本地服务与 Studio 页面分开：关页面不会停止服务；“停止服务”或退出管理窗口会停止本窗口启动的本地服务，超算作业不会自动取消。

文件隔离不分配 GPU。多个计算项目仍需在各自配置中指定合适的 GPU/超算资源；现有预算与调度限制继续生效。

