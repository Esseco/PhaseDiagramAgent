# 相图 Agent 工作区

- [初始配置](parameters/search_config.project.json)：本地/超算环境、搜索边界、母结构与计算设置。
- [运行配置](parameters/run_config.project.json)：预算、采样、MC、微调与验证；旧项目未拆分时仍使用原科学配置文件。
- 两份配置均可手动编辑，保存后说“读取配置 JSON 并继续”；也可直接告诉 Agent 修改参数。新配置检查通过后保存版本快照，已有计算仍归属原配置版本。
- [查看提交、回传与分析文件索引](analysis_outputs/output_index.md)。
- [跨epoch流程总览](analysis_outputs/round_summary.csv)。
- [运行状态](workflow_state/state.json)：状态唯一事实源，建议通过Agent修改。
- [记忆视图](agent_memory/)：经Agent审阅批准后修改，不反向导入视图。
- [启动绑定](agent_runtime.json)：启动器继续选择此文件，无需另建项目。

| 目录 | 内容 |
| --- | --- |
| parameters | 可编辑参数、配置会话、冻结快照 |
| workflow_state | 活状态、台账、缓存、审批、提交清单 |
| agent_memory | 可读记忆视图 |
| structures/reference_structures | 原始母结构与相参考 |
| structures/candidate_structures | 生成的候选结构池 |
| submissions/epochN_model | 本epoch提交输入与原始results |
| analysis_outputs/epochN_model | 本epoch相图、误差、图、训练数据和诊断 |
| logs | 服务日志 |
| history_backups | 有独立内容的历史备份；已迁移重复副本删除 |
| documentation | 说明文档 |

epoch表示模型代际，模型名使用实际版本。微调round按有效轮次编号，未执行且已废弃的旧草稿不占编号。

每个Search-group对应一批branch：Relax-0001与Relax-0001_MC-round-0001/0002配套；DFT-round按实际轮次归属；MLIP-finetune-round放在其原模型epoch下。

每轮inputs放提交文件，results放本地必需回传文件，analysis_outputs放本地处理产品。上传完整轮目录；Relax/MC在inputs中的批次目录提交GPU.sh，DFT在单任务目录提交，微调在训练轮inputs提交GPU.sh一次。回传同级results；模型和大日志留在超算。历史快照与当前相图各有用途，不按内容相同删除。
