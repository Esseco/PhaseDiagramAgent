# 项目文件管理

生产应用源码位于 Process_PhaseDiagram/phase_agent；科学项目与源码独立，保留在各自工作区。新项目使用同一模板，不复制应用代码。现有入口 `python run_project.py` 不变。

```text
项目/
├── run_project.py               打开此项目
├── agent_runtime.json           启动绑定
├── parameters/                  初始配置、运行配置和确认版本
├── structures/                  母结构、候选结构
├── workflow_state/              科学状态、账本、审批和执行凭据
│   ├── dialogue_memory/         哈希命名的对话记忆摘要
│   └── dialogue_process/        哈希命名的对话处理记录
├── shared_dft/                  共享 DFT 数据读取库 records.sqlite3
├── interaction_records/
│   ├── epoch0/                  本轮交互；英文阶段名_YYYYMMDD_序号.md
│   ├── epoch1/
│   ├── project/                 项目级维护说明
│   └── unassigned/              无法可靠确定轮次的旧记录
├── submissions/
│   ├── README.md                提交与回传索引
│   ├── epoch0/
│   │   ├── README.md            完整模型版本与各目录用途
│   │   ├── layout_manifest.json  直接子目录的层级、登记状态和未分类项
│   │   ├── calculations/        登记的计算工作目录；每个 operation 保留 inputs 与 results
│   │   ├── delivery/
│   │   │   ├── transfers/       普通计算、训练的配对上传和回传
│   │   │   └── relax_mc/        弛豫与蒙特卡洛配对批次
│   │   └── evidence/
│   │       ├── verified_results/ 已校验读取的原始结果
│   │       ├── recovery/         仅回收资料和故障凭据
│   │       └── storage_archives/ 完整归档包
│   └── epoch1/
├── analysis_outputs/
│   ├── latest/                  当前模型结果与状态
│   │   ├── README.md
│   │   ├── phase_diagram/       MLIP 总相图、D/H 相图及信息
│   │   └── dft_error/           当前模型同帧 DFT 误差及信息（若可用）
│   ├── epoch0/
│   │   ├── README.md
│   │   └── phase_diagram/       本轮唯一相图目录，按实际事件顺序编号
│   │       ├── 0_update_phasediagram/
│   │       ├── 1_relaxmc-1/     若发生；包含对应图和说明
│   │       └── 2_dft-selection-1/  若发生；包含选点图和说明
│   ├── epoch1/                  同样按 phase_diagram、dft、dft_error、diagnostics 分类
│   ├── out_data/                当前跨轮表格与附属信息
│   └── history/                 完整历史、内部快照、旧格式及失败轮归档
├── workflow_state/analysis/     内部索引、跨轮汇总、未分配证据和整理记录
└── history_backups/             人工维护快照
```

编号历史从采用新版发布入口的当前已核验相图开始；更早的旧图与引用继续保留在原路径，不依据文件时间推测顺序。DIRECT 可在相图不可用时作为首个步骤。`latest` 中如有仍被登记状态引用的历史证据，Agent 会保留并在整理报告中标出，直到可通过正式入口安全迁移。只创建实际需要的目录；实际 epoch 数按模型登记决定。正式 MLIP 主图包含全部有效点，D/H 对照单独绘制；已选 DFT 时生成独立选点图。每次已核验的当前模型凸包发布还会在同一 epoch bundle 和 `latest/phase_diagram/` 中生成 `voltage_curve.png`、`voltage_curve.csv` 与 `voltage_curve_status.json`。电压按相邻稳定 hull 段的能量斜率计算，并使用已确认配置中的 `energy_conventions.voltage_references.Na` 作为 Na 化学势参考；缺少显式参考时保留带原因的 not_ready CSV 和图，不猜测参考能量。历史凸包或帧不足时标记不可用，不以其他数据代替。

## 日常使用

1. 先读 `submissions/README.md`、`submission_index.json`，再读目标 `epochN/README.md` 与 `layout_manifest.json`。阅读本轮答复中的实际路径。上传完整 `_input` 文件夹，在超算执行其中的 `sbatch submit_all.sh`。
2. 将远端对应 `_output` 文件夹整体放回原登记目录，再告诉 Agent“已回传”。程序沿登记的回传路径校验和读取，不扫描其他项目寻找结果。
3. `calculations` 只保存登记的本地计算；上传和回传只看 `delivery/`；`evidence/` 是已校验科学结果与凭据。分析图表看 `analysis_outputs/epochN/`。历史已提交任务不重投。
4. 交互详细记录在 `interaction_records/epochN/`；答复保留本轮文件的完整地址。项目级维护文档放 `project`，无法可靠确定轮次的旧记录放 `unassigned`。共享 DFT 数据由 `shared_dft/records.sqlite3` 读取；对话缓存和处理记录分别在 `workflow_state/dialogue_memory/`、`workflow_state/dialogue_process/`。

## 维护实现

模型与 epoch 的对应仍由 `upload_layout.model_rounds` 登记。目录采用简短 epoch 名称，完整模型版本与任务、批次身份不变。重复编号对应不同模型会显式报错，防止目录合并。

- `tools/remote/submission_architecture.py`：四层提交契约和每个 epoch 的 `layout_manifest.json`。
- `tools/remote/model_upload_directory.py`：轮次目录的共享命名。
- `runtime/interaction_report_paths.py`：交互目录和独占序号；未知模型不会猜测 epoch。
- `tools/remote/submission_handoff.py`：普通计算和训练的 transfers 配对包。
- `tools/relax_mc_batch/prepare.py`、`results.py`：Relax/MC 配对批次和 verified_results。
- `analysis/feedback/workspace_guide.py` 与 templates：新项目和现有项目共用说明。
- `tools/local/organize_epoch_folders.py`：显式迁移、冲突预检、字节核验与归档代码副本清理。

迁移直接使用真实 epoch 目录。运行记录与索引更新为新路径；历史 manifest 与冻结配置保持原始内容，读取时使用显式路径迁移记录。旧目录链接与重复文件名核对后移除。

清理仅删除 SHA256 与保留 inputs.zip 对应成员一致的解压运行代码；删除清单记录归档路径、成员名和哈希。内容不同的文件、原始结果和恢复资料保留。修改源码后由用户手动重新启动服务加载，维护工具不控制服务或提交计算。

目录被占用而无法移动时，迁移明确报错并保留原数据；不再创建目录链接。

旧路径的读取转换复用 `configuration/session/relocate_workspace_paths.py`，支持多次搬迁与循环检测；冻结配置保持原样。当前状态持久化新路径，工作流控制记录保留迁移版本，旧检查点在读取时转换，过期状态写入会被拒绝。迁移不依赖目录 junction 或文件硬链接。

未登记的提交目录、旧式回传包和其他历史材料统一放在 `submissions/previously_used/`，禁止从该目录提交。

## 分析版本与 latest

每次核验结果更新（包括 Relax、MC 分段、DFT 同帧预测补充）、排除/恢复与模型刷新，自动发布当前 MLIP 双图及各已登记 epoch 的累计 DFT 误差。普通产物索引只将 MLIP 作为正式相图；旧 DFT/combined 文件不删除。

版本由模型/epoch、输入内容、绘图和质量规则决定；重复输入复用版本，不覆盖历史。manifest.json 保存输入指纹、模型绑定、父版本、必需文件和 SHA256 校验和。最新完整副本同时更新 epoch 内 latest 与项目级 latest。两张相图或能量/受力图任一失败，不发布不完整版本；*_status.json / latest_status.json 明确标记 not_ready，保留的上一版不能称为当前模型结果。图中的 HD 打叉仅表示既有规则的假设排除对照，不自动删除正式数据。

DFT 误差按原轮冻结模型、同帧有效配对累计，用 data_id 或 task_id+frame 去重，冲突身份显式失败。能量按结构等权，受力汇总全部原子 Cartesian 分量，重新计算 MAE/RMSE，不平均各轮指标。dataset_manifest.json 与逐轮旧产物保留来源。缺少当前模型的实际预测时保持待评估，不将旧模型预测重命名为新 epoch，也不自动宣称独立测试集误差。
