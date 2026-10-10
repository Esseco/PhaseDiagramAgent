# 相图 Agent 工作区

这里保存本项目的数据和配置，程序源码独立存放。先看下表找文件，再按正常审批流程操作。

| 目的 | 位置 |
| --- | --- |
| 启动项目 | [agent_runtime.json](agent_runtime.json) |
| 修改体系、环境、边界 | [初始配置](parameters/search_config.project.json) |
| 修改预算、MC、DFT和训练设置 | [运行配置](parameters/run_config.project.json) |
| 查看产物 | [结果索引](analysis_outputs/output_index.md) |
| 查状态与审批 | workflow_state/ |
| 查母结构与候选 | structures/reference_structures/、candidate_structures/ |
| 查提交与回传 | submissions/，轮次内 inputs/ 与 results/ |
| 查记忆 | agent_memory/，阅读视图不授予审批 |
| 查恢复资料 | history_backups/ |

在程序源码目录激活 py1，运行 `python -m phase_agent.runtime.local_project_launcher`，选择这里的 agent_runtime.json。不要在数据项目目录运行源码模块。

配置可以直接写入草稿。手动编辑后让 Agent 读取检查；确认版本与历史任务参数分别保存。普通表达由 LLM 理解；“继续”用于分析当前状态与回收结果，不等于批准计算或模型激活。

计算、输入准备、提交与激活按审批规则进行。Relax/MC 提交 inputs 内批次 GPU.sh；DFT 提交单任务脚本；微调提交训练轮 inputs/GPU.sh 一次。回传对应 results 后继续。模型和大轨迹按项目协议留在超算，独立验证不能用训练数据或K折成绩代替。

文件对应关系见 [目录说明](documentation/FILE_LAYOUT.md)。项目服务由源码中的 Studio 入口管理；状态、账本与检查点保持原登记，不手动改任务编号或完成标记。
