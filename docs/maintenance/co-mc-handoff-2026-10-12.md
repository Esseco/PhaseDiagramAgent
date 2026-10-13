# Co MC 交互包修复

根因：MC输入准备返回已准备状态前漏掉统一交付包创建；简洁显示在无handoff时退回内部epoch目录，给出错误上传指引。
已补齐MC任务准备后的统一打包；简洁提示只接受存在inputs.zip和submit_all.sh的交付包，缺包明确报告阻塞，不展示内部epoch目录作为提交位置。
当前Co的24个MC任务已统一打包为3个GPU批次，原task.json字节、任务记录、预算和审批保持不变。仅新增批次handoff元数据及提交索引。状态修改前备份、持有原生命周期锁并做写前比对，没有提交作业或管理服务。

提交文件夹：E:\0-PhaseDiagram\O3-1ele-NaCoO2\diagrams\diagram_default\submissions\archives\01-upload\Co_MC_Round01_MC24
回传文件夹：E:\0-PhaseDiagram\O3-1ele-NaCoO2\diagrams\diagram_default\submissions\epoch0_mace-mh-1\return-results
尚未提交的上传完整提交文件夹，在超算执行 sbatch submit_all.sh。已经提交的任务不要重交。回传远端return-results文件夹中的结果后告诉Agent已回传。
28项MC/等待回复/交互包回归通过，生产全包ruff lint通过，两个修改源码文件格式通过。当前包已核对ZIP完整性、3个脚本及24个实际任务ID，不含模型权重。
旧服务需要用户手动重启加载源码；当前状态中的交互包已补齐。
