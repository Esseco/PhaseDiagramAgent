# 相图 Agent 工作区

这里保存本项目的数据和配置，程序源码独立存放。先看下表找文件，再按正常审批流程操作。

| 目的 | 位置 |
| --- | --- |
| 打开本项目 | `python run_project.py`，再点击启动服务 |
| 运行入口配置 | [agent_runtime.json](agent_runtime.json) |
| 修改体系、环境、边界 | [初始配置](parameters/search_config.project.json) |
| 修改预算、MC、DFT和训练设置 | [运行配置](parameters/run_config.project.json) |
| 查看产物 | [结果索引](analysis_outputs/output_index.md) |
| 查状态与审批 | workflow_state/ |
| 查母结构与候选 | [母结构](structures/reference_structures/) · [候选结构](structures/candidate_structures/) |
| 查提交与回传 | Agent 先读 [机器索引](submissions/submission_index.json)，目录说明见 [提交索引](submissions/README.md) |
| 查交互记录 | interaction_records/epochN/，见 [交互索引](interaction_records/README.md) |
| 查共享数据 | shared_dft/records.sqlite3（共享 DFT 数据库，按配置读取） |
| 查对话信息 | workflow_state/dialogue_memory/ 与 dialogue_process/（哈希 JSON，按会话和处理批次保存） |
| 查记忆 | agent_memory/，阅读视图不授予审批 |
| 查恢复资料 | history_backups/ |

新项目可以在这里执行 `python run_project.py` 打开已选中本项目的管理界面；`python run_project.py --check` 只读检查，不启动服务或计算。入口使用创建时的 py1 和共享源码，不在每个数据项目里复制程序。

也可以在源码目录激活 py1，运行 `python -m phase_agent.runtime.local_project_launcher` 选择这里的 agent_runtime.json。首次确认科学边界后使用原有审批流程；每轮最多一个科学动作。

配置可以直接写入草稿。手动编辑后让 Agent 读取检查；确认版本与历史任务参数分别保存。普通表达由 LLM 理解；“继续”用于分析当前状态与回收结果，不等于批准计算或模型激活。

计算、输入准备、提交与激活按审批规则进行。优先使用 Agent 交付且通过输入验收的完整 `_input` 文件夹，执行其中 `sbatch submit_all.sh`；远端环境通过预检后才提交子作业。将配对 `_output` 文件夹整体回传原登记位置后继续。历史任务沿用其登记脚本、目录和回执，已提交任务不重复提交。模型和大轨迹按项目协议留在超算，独立验证不能用训练数据或K折成绩代替。

文件对应关系见 [目录说明](documentation/FILE_LAYOUT.md)。项目服务由源码中的 Studio 入口管理；状态、账本与检查点保持原登记，不手动改任务编号或完成标记。

根目录 agent_runtime.json 是日常入口。diagrams/diagram_default/agent_runtime.json 仅作历史兼容配置入口，科学状态和结果仍归项目根。候选结构和记忆在初始化后建立，未开始的项目可能尚无这些目录。
