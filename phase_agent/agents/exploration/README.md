# Exploration Agent

agent.py 接收共享证据和当前问题，prompts.py 限定探索分析；generation_contracts.py、mc_contracts.py 与 generation_plan.py 描述已有 Branch/MC 约束。branch_pool_review 负责补充前的已有池审阅。

只提出探索建议，不选择 DFT/训练、不评估 MLIP 误差、不执行任务或决定全局终止。具体结构生成、Relax 和 MC 算法仍在 science，派发仍走唯一审批链。
