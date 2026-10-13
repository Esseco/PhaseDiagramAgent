# Relax/MC/DFT 统一交付包

已将统一交付包收尾接入三个准备入口，包括 MC 已准备任务的恢复出口。打包失败明确返回阻塞原因和已有状态，不丢失任务、不重新分配预算。微调准备流程原本已调用统一 prepare_handoff，保留该路径。

完整交互文件夹必须包含 inputs.zip、submit_all.sh、dispatch_tasks.py。简洁回复逐包检查三件文件；不完整时报告需补齐交互包，不再把内部epoch目录当作提交位置。准备或重建包不等于提交任务；已提交的不应重复提交。

包的复用保护：保留已发布的交付包和现有批次/任务身份。当前检查没有修改 FeMn/Cr 的 state.json、任务、预算、审批或已有压缩包；核对ZIP CRC、提交脚本数、实际task_id及不包含模型权重。失败任务不会自动重试，已豁免等待任务不会恢复等待或重新发布。

Cr：现有 Cr_MC_Round02_MC1 包完整，当前1个MC任务。
FeMn：现有 FeMn_DFTSP_Round01_DFTSP3 包完整，但3个旧DFT任务已 recovery_wait_waived=True，不重新引导提交；保存原包和标记。失败刷新也不自动重试。展示逻辑在刷新没有实际等待任务时允许显示其他尚在等待的交互包，并继续尊重所有豁免。

55项相关回归通过，涵盖Relax/MC/DFT打包、打包失败任务保留、重复准备复用、预算不足选择、微调及等待显示。旧测试夹具将MC预算100改为900：当前最低段需要200步，原夹具的100步应进入预算选择而不能直接分配；科学预算算法没有修改。目录计数断言只统计批次目录，避免把同名归档ZIP重复算为新批次。

未调用真实LLM/HPC，未提交或重启服务。源码更新需用户手动重启Agent加载。

O3-1ele-NaCrO2: E:\0-PhaseDiagram\O3-1ele-NaCrO2\diagrams\diagram_default\submissions\archives\01-upload\Cr_MC_Round02_MC1

O3-2ele-NaFeMn: E:\0-PhaseDiagram\O3-2ele-NaFeMn\submissions\archives\01-upload\FeMn_DFTSP_Round01_DFTSP3
