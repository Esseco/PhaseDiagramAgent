# 工作区布局

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

新生成Relax/MC/DFT批次使用阶段轮目录下的inputs/批次与results/任务分离；微调也采用inputs和results同级，GPU.sh在inputs。上方旧树中批次实际置于inputs。详细目录与上传层级见 [每轮传输布局](ROUND_TRANSFER_LAYOUT.md)。旧目录仅通过显式离线迁移整理。

记忆通过Agent审阅/批准修改，阅读视图不反向导入，state是唯一事实源。冻结配置内容、版本与哈希保留；本地移动通过runtime_storage_override和local_path_relocations解析。相图、误差、训练数值不重新计算，CSV只更新路径。当前相图和历史快照用途不同，即使相同也不删除。

维护模块（默认仅预览，apply=True才执行）：

- rename_workspace_directories：旧名称改为明确名称，服务停止后执行；目标冲突、链接、并发编辑均拒绝。
- remove_migrated_duplicates：只删除history_backups中与有效文件同名、字节一致的旧副本，保留有效文件和删除清单，不删除唯一历史。
- verify_workspace_layout：只读检查当前绑定、母结构与必需文件。

以上位于execution_layer/local。config/runtime/upload_batches/outputs等旧入口仍由已有配置显式支持，不自动迁移。旧迁移器仅作为v2历史升级工具保留；升级后不会再次创建旧名称目录。重启仍选工作区根目录agent_runtime.json。
