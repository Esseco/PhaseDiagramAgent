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

各阶段results随提交目录回传，不拆到analysis_outputs。Relax/MC提交批次GPU.sh，DFT提交单任务GPU.sh，微调提交训练轮GPU.sh一次。训练方案以training_plan.json为准，不重复发布finetune_report.json。

新生成Relax/MC/DFT批次使用阶段轮目录下的inputs/批次与results/任务分离；微调也采用inputs和results同级，GPU.sh在inputs。上方旧树中批次实际置于inputs。详细目录与上传层级见 [每轮传输布局](workspace.md#doc-round_transfer_layout)。旧目录仅通过显式离线迁移整理。

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

新生成的Relax、MC、DFT批次采用以下布局，旧任务按已有登记路径继续回收；本次不移动现有文件。

```text
submissions/epoch0_模型版本/Search-group-0001/
  Relax-0001/
    inputs/Relax-submission-0001_remote-000001/  上传、提交及远端工作目录
    results/任务目录/                           只放本地回收必需结果
  Relax-0001_MC-round-0001/
    inputs/MC-sampling-0001_remote-000002/
    results/任务目录/
  DFT-round-0001_方案标识/DFT-single-point/
    inputs/DFT-single-point-submission-0001_remote-000003/
    results/任务目录/
analysis_outputs/epoch0_模型版本/
  Search-group-0001/DFT-round-0001_方案标识/
    comparisons/                              本地处理的误差与图
    training/                                 本地整合的训练数据
    diagnostics/                              本地磁矩等诊断
  phase_diagrams/                              本地相图
```

上传整个阶段轮目录，保留inputs与results的相对层级；无需上传analysis_outputs。Relax/MC只提交inputs下每个批次根GPU.sh一次；DFT提交对应单任务GPU.sh。下载同一轮results，放回本地对应results目录。Agent依照state/manifest登记路径读取，不要求下载远端工作目录。

inputs在超算运行后会产生原始大文件，不能理解为不可变目录；这些文件留在超算，不回传。results只含必需科学结果、校验标记和必要远端路径清单，清单见LIGHTWEIGHT_RESULT_TRANSFER.md。模型刷新任务也使用inputs/results分离。

分析输出始终在analysis_outputs，不混入results。本次仅调整新批次生成规则；任务ID、epoch含义、审批与科学执行逻辑不变。

微调也已统一为MLIP-finetune-round-0001/inputs（GPU.sh、训练数据、成员配置、收集脚本）和同级results（指标与远端模型清单）。完整上传该轮目录，在inputs内sbatch GPU.sh一次。原轮重生成保留轮根登记及编号，结果存在时仍阻止自动替换。

现有工作区可通过phase_agent.tools.local.migrate_round_inputs预览并显式应用迁移；要求服务关闭，备份state和修改脚本，可回滚。已完成计算的task.json/manifest内容保持不变，不将它们改为新的科学任务。新生成任务使用新的相对结果路径；历史输入仅用于记录，不建议把旧manifest重新提交。原始回传results不移动、不重算；analysis_outputs仍为独立处理产品。

[phase_agent/tools/remote/build_upload_batch_directory.py](../../phase_agent/tools/remote/build_upload_batch_directory.py)。

<a id="doc-epoch_output_layout"></a>

## Epoch 与数据类型归档

submissions 与 analysis_outputs 共用 state.upload_layout.model_rounds 的模型映射。epoch 表示模型代际，不是 MC、DFT 或微调尝试次数。模型版本不变时 epoch 不变；不按文件夹个数推测。

新版 Agent 输出目录：

    analysis_outputs/epoch0_mace-mh-1/
      phase_diagrams/mlip/phase_diagram.csv
      phase_diagrams/dft/phase_diagram.csv
      phase_diagrams/combined/phase_diagram.csv
      Search-group-0001/DFT-round-0001_<operation>/
        comparisons/{energy_comparison,force_comparison,metrics}.csv
        comparisons/plots/
        diagnostics/magnetic_moments.csv
        training/training.json
      round_metrics.csv
    analysis_outputs/round_summary.csv
    analysis_outputs/output_index.md

output_index.md 是查找入口：列出已登记的相图、误差、磁矩、绘图、训练输入路径及文件是否存在；不扫描原始计算数据、不计算新结果。round_summary.csv 用于跨 epoch 总览，round_metrics.csv 用于单 epoch 多 DFT 轮误差比较。索引在 Agent 发布 DFT 分析产品时更新，内容相同不重写。

三个相图各自保留 history。DFT 相图是该 epoch 时刻累计的合格 DFT 凸包，不表示只含该 epoch 的 DFT 数据；具体来源仍由记录列追踪。综合相图明确保留能量来源和校正状态，不与纯 DFT/MLIP 相图混同。

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

