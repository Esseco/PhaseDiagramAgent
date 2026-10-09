# 每轮提交、回传与处理结果分离

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

现有工作区可通过execution_layer.local.migrate_round_inputs预览并显式应用迁移；要求服务关闭，备份state和修改脚本，可回滚。已完成计算的task.json/manifest内容保持不变，不将它们改为新的科学任务。新生成任务使用新的相对结果路径；历史输入仅用于记录，不建议把旧manifest重新提交。原始回传results不移动、不重算；analysis_outputs仍为独立处理产品。
