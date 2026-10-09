# 相图 Agent 工作区

本目录保存一个独立项目的配置、结构、提交文件、结果和记忆。程序源码另存于代码仓库；启动服务后，在 Studio 中与本项目 Agent 对话。

## 启动与日常操作

在 **源码目录** 激活 `py1` 后运行 `python -m run.local_project_launcher`，选择本项目的 [agent_runtime.json](agent_runtime.json)。不要在只有结构和数据的项目目录中运行这个模块。

| 你要做什么 | 对 Agent 说什么 |
| --- | --- |
| 查看进度，不推进任务 | 查看状态 |
| 回收新结果并推进到下一处等待 | 继续 |
| 手动修改文件后先检查 | 读取配置 JSON |
| 检查配置，通过后继续 | 读取配置 JSON 并继续 |
| 修改长期使用的参数 | 修改运行配置：把……改为…… |
| 批准独立验证通过的模型 | 激活候选 `<版本>` 原因：`<审阅理由>` |

“继续”用于回收与推进，不等于批准模型激活。具体科学计算由你在超算提交；Agent 准备输入并说明提交位置和回传文件。

## 两份可编辑配置

- [初始配置](parameters/search_config.project.json)：本地与超算环境、搜索边界、体系组成、母结构、模型路径和必要计算设置。
- [运行配置](parameters/run_config.project.json)：预算、采样、MC、微调、验证和收敛参数。旧项目未拆分时仍使用原科学配置文件。
- 两份均可手动编辑，也可告诉 Agent 要修改的值。保存后重新读取检查；通过后生成新版本快照，历史计算仍归属原配置版本。

初始必要信息齐全即可进入搜索建议；缺项或冲突会再次询问。后续阶段缺少的验证标准等信息，到使用该阶段时再补齐。

## 超算提交与回传

上传完整轮次目录，在对应 `inputs` 中提交作业。计算、模型读取和 SHA256 计算均通过计算节点作业执行；登录节点只负责提交与查询。

| 阶段 | 提交位置与文件 | 回传内容 |
| --- | --- | --- |
| Relax / MC | `inputs` 内对应批次的 `GPU.sh` | 对应 `results` |
| DFT | 对应单任务目录中的提交脚本 | 对应 `results` |
| 微调训练 | 训练轮 `inputs/GPU.sh`，提交一次 | K折及逐点CSV、`models.json`、完成标记 |
| 补模型清单 | 原训练轮 `inputs/GPU_manifest.sh` | `results/models.json` |
| 独立模型验证 | 原训练轮 `inputs/GPU_validation.sh` | 指定的 `results/validation-<request_id>.json` |

回传后说“继续”。模型和大日志保留在超算，不必下载；独立验证数据必须与训练数据区分，K折结果不能替代最终主模型独立测试。

## 等待与恢复

Studio 主图可展开 `training_lifecycle`，查看回收、清单检查、作业准备、验证和等待节点。训练交接通过 LangGraph 原生中断暂停，检查点保存在 `workflow_state/langgraph_training.sqlite`。重启后“继续”会带入最新业务状态并重新核对文件。

[运行状态](workflow_state/state.json) 是业务状态事实源，建议通过 Agent 修改。检查点记录恢复位置，不替代业务状态或激活审批。其他搜索阶段沿用各自现有恢复机制。

## 文件导航

- [结果与分析索引](analysis_outputs/output_index.md)
- [跨epoch流程总览](analysis_outputs/round_summary.csv)
- [提交与结果布局说明](documentation/FILE_LAYOUT.md)
- [项目记忆](agent_memory/)：事实与建议经审阅管理，文件视图不直接反向导入状态。

| 目录 | 内容 |
| --- | --- |
| parameters | 初始/运行配置、配置会话和版本快照 |
| structures/reference_structures | 初始母结构 |
| structures/candidate_structures | 候选结构池 |
| submissions/epochN_model | 提交输入与原始回传结果 |
| analysis_outputs/epochN_model | 相图、误差、训练数据和诊断 |
| workflow_state | 状态、台账、缓存、审批和检查点 |
| agent_memory | 项目记忆视图 |
| logs | 服务日志 |
| history_backups | 历史备份 |
| documentation | 项目使用说明 |

`epoch` 表示已激活模型的代际；同一模型下的多个搜索轮次不增加 epoch。每个 Search-group 对应一批 branch，Relax、MC 和 DFT 按该组关联；微调轮属于原模型 epoch。每轮 `inputs` 放提交文件，`results` 放原始回传，`analysis_outputs` 放本地处理产品。
