# 提交、结果与分析文件

## 对应关系

文件夹只用 epoch0、epoch1 等简短名称；模型完整版本保留在各轮 README 和机器记录中。epoch对应实际激活的模型版本；Search-group对应一次branch生成与搜索链。
Relax-0001与Relax-0001_MC-round-0001、0002为同一组的浅/深MC。
submissions 在 epoch 下直接列出 G0001_Relax-0001、G0001_Relax-0001_MC-round-0001 和 G0001_DFT-0001_DFT-single-point 等计算目录；G 表示搜索组。每个计算目录的 inputs/results 为同级目录。MLIP-finetune-round 位于原模型 epoch 下。提交和回收入口见 submissions/README.md；运行记录使用当前目录，不保留旧目录链接。
未执行且已明确废弃的微调草稿不占编号，不重编号真实运行历史。

## 回传到哪里

submissions 保存输入、各阶段原始 results 和配对交付目录。新交付使用完整 `_input` 文件夹内的 `sbatch submit_all.sh`，对应 `_output` 整体回传登记位置。历史任务保留原脚本与结果位置，已提交的作业不重新提交。
不把原始results放到analysis_outputs，不手动修改task_id或完成标记。

## 分析文件在哪里

analysis_outputs/epoch…/phase_diagrams：MLIP、DFT、combined三种相图及history。
analysis_outputs/epoch…/Search-group…/DFT-round…：

- comparisons：能量、受力逐点CSV，误差指标；plots是对角线图。
- training：可供MLIP训练的结构和标签。
- diagnostics：逐原子磁矩与检查信息。

当前版本不因重复请求重新分析；有新有效数据时由Agent更新。
母结构在structures/reference_structures，候选池在structures/candidate_structures。
参数经Agent复核确认后生效，记忆视图不直接反向导入state。

## 数据读取与对话信息

- `shared_dft/records.sqlite3`：共享 DFT 数据库，由配置指定读取，不手动拆分或移动。
- `workflow_state/dialogue_memory/`：按哈希文件保存的对话记忆摘要。
- `workflow_state/dialogue_process/`：按哈希文件保存的对话处理记录。
- `interaction_records/`：面向人阅读的交互 Markdown，文件名统一使用英文 ASCII。

完整文件位置见[文件索引](../analysis_outputs/output_index.md)。

## 每轮目录速查

| 路径 | 用途 |
| --- | --- |
| interaction_records/epochN/ | 本轮用户消息、完整答复和维护说明 |
| submissions/epochN/calculations/ | 本地计算工作目录，按操作名分组 |
| submissions/epochN/delivery/transfers/ | 普通计算与训练的上传 `_input` / 回传 `_output` |
| submissions/epochN/delivery/relax_mc/ | Relax/MC 配对批次及其上传、回传入口 |
| submissions/epochN/evidence/verified_results/ | 已校验入账的原始结果和凭据 |
| submissions/epochN/evidence/recovery/ | 仅回收与故障核对资料 |
| submissions/epochN/evidence/storage_archives/ | 完整归档包，程序按需恢复 |
| submissions/previously_used/ | 未登记或旧式材料，仅供追溯 |
| analysis_outputs/epochN/phase_diagrams/ | mlip、dft、combined 相图；history 保存历史快照 |
| analysis_outputs/epochN/Search-group-*/DFT-round-*/ | 每次 DFT 的 comparisons、training、diagnostics |

submissions 中其余 G0001_* 为按搜索组编号的计算工作目录，inputs 是原始输入，results 是计算结果。
目录直接保存真实文件，不保留旧目录链接；verified_results/recovery 的有效结果与恢复凭据保留。未登记任务和旧式回传包移入 submissions/previously_used/。
跨轮汇总 round_summary.csv、gpu_usage.csv 保留在结果目录根部，避免每轮重复。尚无 epoch 的记录放 unassigned，项目级维护说明放 project。
