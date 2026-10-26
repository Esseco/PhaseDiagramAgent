# 项目目录与提交回传

按下方目录定位需要的说明；源码入口链接保留在对应章节。项目科学参数以确认配置为准。

- [工作区布局](#doc-workspace_layout)
- [每轮提交、回传与处理结果分离](#doc-round_transfer_layout)
- [Epoch 与数据类型归档](#doc-epoch_output_layout)
- [最小结果回传](#doc-lightweight_result_transfer)

<a id="doc-workspace_layout"></a>

## 工作区布局

代码仓库与体系工作区分离。新项目默认采用以下名称；已有项目仅在用户明确要求后迁移。

```text
agent_runtime.json                启动入口（不改名）
parameters/                           参数、配置会话、冻结快照
workflow_state/                       state、台账、缓存、审批、提交清单
agent_memory/                         记忆阅读视图
structures/
  reference_structures/               母结构与相参考
  candidate_structures/               生成的候选结构
submissions/epochN_实际模型版本/
  Search-group-0001/                  同一次branch搜索链
    Relax-0001/
    Relax-0001_MC-round-0001/
    Relax-0001_MC-round-0002/
    DFT-round-0001_方案标识/
  MLIP-finetune-round-0001/            本epoch原模型的微调输入
analysis_outputs/
  output_index.md                     提交、结果、分析的文件索引
  round_summary.csv                   跨epoch总览
  epochN_实际模型版本/
    phase_diagrams/mlip|dft|combined/  当前相图与history快照
    Search-group-0001/DFT-round-…/
      comparisons/                    能量、受力CSV及plots
      training/                       可训练数据
      diagnostics/                    磁矩检查
logs/                                 日志
history_backups/                      唯一旧版本、恢复资料及整理清单
documentation/                        工作区说明
```

epoch表示激活的模型代际，不等于MC、DFT或微调子轮次。实际模型版本、任务ID、远端路径不因本地整理而改变。未执行且已明确废弃的微调草稿不占轮次；真实提交、完成或激活的历史不重编号。

当前交付统一为登记的完整 `_input` 文件夹及配对 `_output`。上传整个 `_input`，在其中执行 `sbatch submit_all.sh` 一次；结果整体回传登记的 `_output`，再由 Agent 校验和导入。内部 GPU.sh 由交付脚本派发，不另行提交。历史任务沿原登记脚本与路径回收，已提交任务不重复提交。训练方案以 training_plan.json 为准。

阶段轮下的 inputs/results 是机器索引和历史任务的内部布局，不是当前人工上传入口。上方树用于定位原始科学记录；实际上传与回传以 [机器索引](project-files.md) 和下方 [每轮传输布局](workspace.md#doc-round_transfer_layout) 为准。

记忆通过Agent审阅/批准修改，阅读视图不反向导入，state是唯一事实源。冻结配置内容、版本与哈希保留；本地移动通过runtime_storage_override和local_path_relocations解析。相图、误差、训练数值不重新计算，CSV只更新路径。当前相图和历史快照用途不同，即使相同也不删除。

维护模块（默认仅预览，apply=True才执行）：

- rename_workspace_directories：旧名称改为明确名称，服务停止后执行；目标冲突、链接、并发编辑均拒绝。
- remove_migrated_duplicates：只删除history_backups中与有效文件同名、字节一致的旧副本，保留有效文件和删除清单，不删除唯一历史。
- verify_workspace_layout：只读检查当前绑定、母结构与必需文件。

以上位于phase_agent/tools/local。config/runtime/upload_batches/outputs等旧入口仍由已有配置显式支持，不自动迁移。旧迁移器仅作为v2历史升级工具保留；升级后不会再次创建旧名称目录。重启仍选工作区根目录agent_runtime.json。

### 实现入口

[phase_agent/configuration/session/resolve_workspace_paths.py](../../phase_agent/configuration/session/resolve_workspace_paths.py)。

<a id="doc-round_transfer_layout"></a>

## 每轮提交、回传与处理结果分离

人工交付层与内部科学记录分开。不要按旧目录名推断当前提交命令；先读 submissions/submission_index.json 的已登记路径。

```text
submissions/epochN/
  transfers/
    批次_input/                     上传整个目录
      inputs.zip
      submit_all.sh                 在此执行一次
      dispatch_tasks.py             普通任务/训练包
      run_upload.py                 Relax/MC 配对包使用此入口
    批次_output/                    整体回传至登记位置
  阶段轮/inputs、results/            内部与历史科学记录，保持登记路径
analysis_outputs/epochN/             本地相图、误差、训练数据和诊断
```

普通任务/训练包与 Relax/MC 包的派发入口不同，不要求同时存在两个 Python 入口。交付包自行检查环境并派发内部任务，用户不再逐一提交内部 GPU.sh。模型、大轨迹和原始大日志按协议留在超算；只回传本地所需的小型产物与完成凭据。

旧任务继续从原登记 results 路径回收；已提交作业不得重新提交。旧迁移工具仅供显式离线维护，不是运行或回收的必经步骤。冻结 manifest、任务身份、审批和科学结果保持原样。目录职责与当前入口见 [项目文件管理](project-files.md)。

[phase_agent/tools/remote/submission_handoff.py](../../phase_agent/tools/remote/submission_handoff.py) 与 [Relax/MC 交付](../../phase_agent/tools/relax_mc_batch/handoff.py)。

<a id="doc-epoch_output_layout"></a>

## Epoch 与数据类型归档

submissions 与 analysis_outputs 共用 state.upload_layout.model_rounds 的模型映射。epoch 表示模型代际，不是 MC、DFT 或微调尝试次数。模型版本不变时 epoch 不变；不按文件夹个数推测。

新版 Agent 输出目录：

    analysis_outputs/epoch0_mace-mh-1/
      phase_diagrams/mlip/phase_diagram.csv
      Search-group-0001/DFT-round-0001_<operation>/
        comparisons/{energy_comparison,force_comparison,metrics}.csv
        comparisons/plots/
        diagnostics/magnetic_moments.csv
        training/training.json
      round_metrics.csv
    analysis_outputs/round_summary.csv
    analysis_outputs/output_index.md

output_index.md 是查找入口：列出已登记的相图、误差、磁矩、绘图、训练输入路径及文件是否存在；不扫描原始计算数据、不计算新结果。round_summary.csv 用于跨 epoch 总览，round_metrics.csv 用于单 epoch 多 DFT 轮误差比较。索引在 Agent 发布 DFT 分析产品时更新，内容相同不重写。

正式相图仅使用当前模型的 MLIP 能量并保留 history。历史纯 DFT 与综合相图保留为有来源标注的诊断文件；默认回收不重建它们，也不把 DFT 能量混入正式 MLIP 凸包。DFT 同帧比较、相竞争证据和训练数据继续按轮次保存。

CSV 内增加 epoch；误差与磁矩表同时保留模型版本、search group 和 DFT 轮次。超算训练结果 CSV 保留 epoch、原模型版本、training_round 和 fold，不混同本轮原模型同帧误差。

历史 csv_path 仍可读；导出已有 CSV 不重算、不搬迁。科学数据未变时沿用当前文件；后续产生新版数据由 Agent 按新版目录发布。目录内名称简洁，完整归属由路径与 CSV 列共同说明。用户明确要求整理工作区时可运行可恢复迁移器，科学数据不重新计算，旧索引归档。

训练方案状态必须与实际目录一致。旧目录缺失或同模型已有未完成方案而配置/数据变化时，返回 confirmation_required，明确请求核对或重生成，不通过记录数量直接追加训练轮次。不自动删除、恢复或重命名历史任务；提交过的作业也不自动取消。新轮次使用已有合法轮次最大编号加一，不以记录条数代替编号。

[phase_agent/analysis/state/model_epoch.py](../../phase_agent/analysis/state/model_epoch.py)。

<a id="doc-lightweight_result_transfer"></a>

## 最小结果回传

默认原则：只传本地Agent分析、训练数据准备和回收校验必须读取的内容；远端保留的大产物用清单登记位置，不删除原文件。

|阶段|必须回传|留在超算|
|---|---|---|
|Relax/MC|result.json、task.finished.json、结果引用的最终结构；当前重启协议需要的checkpoint；批次runtime统计|原始日志、trace/status/settings/pool诊断文件，位置记入remote_artifacts.json|
|DFT|result.json、task.finished.json、training.json（全部训练帧）、mlip_result.json（同帧原模型预测）；旧结果引用的结构文件|OUTCAR、vasprun.xml、WAVECAR、CHGCAR及原始日志|
|微调|kfold_metrics.csv、energy_comparison.csv、force_comparison.csv、models.json、training.finished.json|正式committee/主模型、K折模型、训练检查点和大日志|

DFT结构、能量、受力、应力和磁矩不能只传路径：本地分析与构建训练集仍要实际读取。模型可只登记远端路径，因为训练和后续科学预测在超算执行。路径和SHA256仅标识模型，不替代独立验证，更不构成激活授权。

新收集脚本不再把模型复制到results/models。models.json包含storage=remote、绝对remote_model_path、model_path、大小、SHA256及主模型/committee身份。保持超算训练目录稳定，否则清单路径失效。

已完成训练如需只刷新模型清单，将项目新版phase_agent/tools/remote/collect_training_results.py替换到超算训练inputs目录（旧布局在训练轮根），然后在该目录运行 `python collect_training_results.py --manifest-only`。不训练、不重算K折，不自动删除旧results/models；下载时只选择表中必需文件，旧模型副本无需下载。

更改仅对使用新版生成/导出脚本的作业生效，已提交旧脚本不会自动更新。当前远端训练结果自动登记到候选模型的完整入口仍未实现；不能据此声称回传清单后会自动验证/激活。现有本地分析输出不参与下载，按需由Agent再生成。

检查点仍保留，因为现有续跑协议会读取它；未经远端续跑适配验证，不把它贸然改为纯路径。原始JSON/CSV压缩传输和历史results清理未在此变更中实施。

[phase_agent/tools/remote/collect_training_results.py](../../phase_agent/tools/remote/collect_training_results.py)。

