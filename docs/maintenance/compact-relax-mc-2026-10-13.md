# Relax/MC 三件式交互包（2026-10-14）

取代上一版两个文件的设计。上传目录严格只有 submit_all.sh、run_upload.py、inputs.zip。提交脚本保留原 Slurm 资源和环境配置，通过 sbatch 分配计算节点后运行独立 Python 文件。脚本和 Python 都检查 SLURM_JOB_ID；Python 校验并解压单个压缩包，保留已有工作状态，随后在同一计算作业内运行原协调器。其余结构、配置和运行代码都在 inputs.zip 中。

已更新 Co、Cr 当前 epoch0_RelaxMC_Batch01/upload。原 inputs.zip SHA256 不变；每包所有载荷校验通过。未修改 state.json、任务、预算或审批，未启动服务或提交 HPC。

生产验证：12 passed，1 deselected（前一版已确认的基线提示文字断言失败）；修改源码 ruff 通过，生产文件与已测暂存文件哈希一致。未执行真实超算任务。
