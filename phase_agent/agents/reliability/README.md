# Reliability Agent

agent.py 和 prompts.py 分析 MLIP/DFT 误差、QBC/OOD 及相稳定性风险；dft_contracts.py、selection/ 保留已有候选、采样和数据充分性工具。

只提出可靠性建议，不决定 Branch 扩展或全局研究路线，不执行 DFT/训练。按配置阶段推进计算的工具位于 tools/state/decide_next_calculation.py；旧模块描述符由 compatibility 集中映射。
