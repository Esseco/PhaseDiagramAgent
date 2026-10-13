# DFT 对照预测执行边界

已删除本地推理分支；即使配置 dft_comparison_location=local，也只接受超算回传的同帧预测。缺少 mlip_result.json 时报告需要在超算补算，不启动本地计算。
超算 write_prediction 使用 device=cuda，保留原轮次模型版本、指纹、最终帧和单位校验。无 CPU 回退。
未修改 Co、Cr、FeMn 数据、进度、审批或服务。
本地 Agent 需手动重启；超算现有代码需更新 predict_dft_final_frame.py 后才使用 GPU。remote-code-update.zip 是源码覆盖包，不是任务输入包；解压到超算的 Process_PhaseDiagram 源码根目录，先保留原文件备份。它不提交计算。旧 DFT 成功结果保留，必要时在超算补算预测。
