# 提交、结果与分析文件

## 对应关系

epoch对应实际激活的模型版本；Search-group对应一次branch生成与搜索链。
Relax-0001与Relax-0001_MC-round-0001、0002为同一组的浅/深MC。
DFT-round归入该Search-group；MLIP-finetune-round位于原模型epoch下。
未执行且已明确废弃的微调草稿不占编号，不重编号真实运行历史。

## 回传到哪里

submissions保存提交输入和各阶段原始results。Relax/MC提交批次GPU.sh，
DFT提交单任务脚本，微调在训练轮inputs提交GPU.sh一次。回传该轮必要results后由Agent读取。
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
完整文件位置见[文件索引](../analysis_outputs/output_index.md)。
