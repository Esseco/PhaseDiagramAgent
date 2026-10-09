# 上传目录按模型代际组织

upload_batches/epoch0_<初始模型>/Search-group-0001 下保留 Branch-0001、Relax-0001、Relax-0001_MC-round-0001/0002、DFT-round-0001 等阶段。新模型激活后的模型编号对应 epoch1，追加搜索组不增加epoch。内部remote编号和task ID不变。

微调输入在来源模型的epoch目录下，MLIP-finetune-round-0001；committee、main_cv、main_final现有内部命名不拆分，防止破坏已有相对数据路径和提交脚本。该目录是从epoch0训练新模型的输入，不能提前归属epoch1。

旧branch结构位于内部结构池时不复制为新任务；未来生成直接写Search-group/Branch。迁移仅移动已存在任务目录，更新state、结构台账、能量池及任务JSON的本地路径；远端路径、历史审批与日志保留原样。metadata备份与migration.json存放backups/epoch-layout-*。迁移没有重生成数据，也不会补齐丢失训练输入。

使用migrate_epoch_upload_layout先dry-run再执行；登记中有活动远端作业、目标冲突或state变化时阻止迁移。运行Agent前应重启以重载登记，远端已经上传的原目录仍可计算，结果需放回新本地对应目录。
# 当前轮次目录规则

当前每轮采用inputs/（上传提交）和results/（必需回传）同级，微调也相同；完整规则见 [每轮传输布局](ROUND_TRANSFER_LAYOUT.md)。下方旧版示例若未包含inputs，仅用于迁移前历史路径说明。

